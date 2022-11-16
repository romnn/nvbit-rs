#![allow(warnings)]
use std::path::PathBuf;

pub fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("cargo out dir"))
        .canonicalize()
        .expect("canonicalize path")
}

pub fn manifest_path() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir"))
        .canonicalize()
        .expect("canonicalize path")
}

fn main() {
    // rerun if the build script changes
    println!("cargo:rerun-if-changed=build.rs");

    // rerun if the instrumentation changes
    println!("cargo:rerun-if-changed=instrumentation");
    println!("cargo:rerun-if-changed=instrumentation/inject_funcs.cu");

    if true {
        // compile the injection functions
        let tool_obj = output_path().join("tool.o");
        let result = std::process::Command::new("nvcc")
            .args([
                "-I../../nvbit-sys/nvbit_release/core",
                "-Xcompiler",
                "-fPIC",
                "-dc",
                "-c",
                "instrumentation/tool.cu",
                "-o",
                // instr_obj.as_os_str().to_str().unwrap(), // .display().as_str(),
                tool_obj.to_string_lossy().to_string().as_str(), // .display().as_str(),
            ])
            .output()
            .expect("nvcc failed");
        println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
        assert!(result.status.success());

        // compile the injection functions
        let inject_obj = output_path().join("inject_funcs.o");
        let result = std::process::Command::new("nvcc")
            .args([
                // "-D_FORCE_INLINES",
                "-I../../nvbit-sys/nvbit_release/core",
                // "-rdc=true",
                "-maxrregcount=24",
                "-Xptxas",
                "-astoolspatch",
                "--keep-device-functions",
                // "-c",
                "-Xcompiler",
                "-fPIC",
                // "-dc",
                "-c",
                "instrumentation/inject_funcs.cu",
                "-o",
                // instr_obj.as_os_str().to_str().unwrap(), // .display().as_str(),
                inject_obj.to_string_lossy().to_string().as_str(), // .display().as_str(),
            ])
            .output()
            .expect("nvcc failed");
        println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
        assert!(result.status.success());

        let dev_obj_link = output_path().join("dev_link.o");
        let result = std::process::Command::new("nvcc")
            .args([
                // "-D_FORCE_INLINES",
                // "-O3",
                // tracer_tool.o
                // "-L../nvbit_release/core",
                // "-rdc=true",
                // "-lcudart",
                "-Xcompiler",
                "-fPIC",
                "-dlink",
                &tool_obj.to_string_lossy(),
                &inject_obj.to_string_lossy(),
                // "-lnvbit"
                // "-lcuda",
                // "-shared",
                "-o",
                &dev_obj_link.to_string_lossy(),
                // "tracer_tool.so",
            ])
            .output()
            .expect("nvcc failed");
        println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
        assert!(result.status.success());

        // println!("cargo:rustc-link-arg=-Wl,-export-dynamic");
        // println!("cargo:rustc-link-arg=-Wl,-fvisibility=extern");
        // println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        // println!("cargo:rustc-link-arg={}", inject_obj.display());
        // println!("cargo:rustc-link-arg={}", inject_obj_link.display());
        // println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

        if true {
            let use_static = true;
            let lib = if use_static {
                let lib = output_path().join("libinstrumentation.a");
                // ar r library.a lib_source.o
                std::process::Command::new("ar")
                    .args([
                        // "r",
                        "cru",
                        lib.to_string_lossy().to_string().as_str(),
                        tool_obj.to_string_lossy().to_string().as_str(),
                        inject_obj.to_string_lossy().to_string().as_str(),
                        dev_obj_link.to_string_lossy().to_string().as_str(),
                    ])
                    .output()
                    .expect("linking failed");
                println!(
                    "linking result: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert!(result.status.success());

                // println!("cargo:rustc-link-arg=-Wl,--whole-archive,-linstrumentation,-Wl,--no-whole-archive");
                println!("cargo:rustc-link-search=native={}", output_path().display());
                // println!("cargo:rustc-link-arg=-L{}", output_path().display());
                // println!("cargo:rustc-link-arg=-Wl,-Bstatic");
                // println!("cargo:rustc-link-arg=-Wl,--whole-archive");
                // println!("cargo:rustc-link-arg=-linstrumentation");

                println!("cargo:rustc-link-lib=static:+whole-archive=instrumentation");

                // println!("cargo:rustc-link-lib=static=instrumentation");
                // println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

                lib
            } else {
                let lib = output_path().join("libinstrumentation.so");
                let result = std::process::Command::new("clang++")
                    .args([
                        tool_obj.to_string_lossy().to_string().as_str(),
                        inject_obj.to_string_lossy().to_string().as_str(),
                        dev_obj_link.to_string_lossy().to_string().as_str(),
                        "-lcuda",
                        "-lcudart",
                        "-shared",
                        "-o",
                        lib.to_string_lossy().to_string().as_str(),
                        // "tracer_tool.so",
                        // "-D_FORCE_INLINES",
                        // "-O3",
                        // tracer_tool.o
                        // "-L../nvbit_release/core",
                        // "-rdc=true",
                        // "-lcudart",
                        // "-Xcompiler",
                        // "-fPIC",
                    ])
                    .output()
                    .expect("linking failed");
                println!(
                    "linking result: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert!(result.status.success());

                // println!("cargo:rustc-link-arg=-Wl,--whole-archive");
                // println!("cargo:rustc-link-search=native={}", output_path().display());
                // println!("cargo:rustc-link-lib=dylib=instrumentation");

                // ENV RUSTFLAGS="-C link-arg=-L/usr/local/share/dpdk/x86_64-native-linux-gcc/lib -C link-arg=-Wl,--whole-archive -C link-arg=-ldpdk -C link-arg=-Wl,--no-whole-archive -C link-arg=-lnuma -C link-arg=-lm -C link-arg=-lc"

                lib
            };
        }

        // cc::Build::new()
        //     // .compiler("nvcc")
        //     .object(&inject_obj)
        //     .object(&inject_obj_link)
        //     .shared_flag(true)
        //     .compile("instrumentation.so");

        // println!("cargo:rustc-link-lib=dylib={}", shared_lib.display());
        // println!("cargo:rustc-link-lib=dylib={}", shared_lib.display());
        // println!("cargo:rustc-link-lib=dylib={}", );
        // println!("cargo:rustc-link-lib=dylib=instrumentation");

        // cc::Build::new()
        //     .compiler("nvcc")
        //     .include("../../nvbit-sys/nvbit_release/core")
        //     .include("/usr/lib/x86_64-linux-gnu")
        //     .no_default_flags(true)
        //     // .define("FORCE_INLINES", "")
        //     .flag("-D_FORCE_INLINES")
        //     .flag("-maxrregcount=24")
        //     .flag("-Xptxas")
        //     // .flag("-cloning=no")
        //     // Compile patch code for CUDA tools. Implies --keep-device-functions.
        //     // May only be used in conjunction with --ptx or --cubin or --fatbin.
        //     // Shall not be used in conjunction with -rdc=true or -ewp.
        //     // Some PTX ISA features may not be usable in this compilation mode.
        //     .flag("-astoolspatch")
        //     // In whole program compilation mode,
        //     // preserve user defined external linkage __device__ function definitions
        //     // in generated PTX.
        //     .flag("--keep-device-functions")
        //     // .flag("-shared")
        //     // Compile each .c, .cc, .cpp, .cxx, and .cu input file into
        //     // an object file that contains relocatable device code.
        //     // It is equivalent to --relocatable-device-code=true --compile.
        //     // .flag("-dc") // MUST BE DISABLED
        //     // Compile each .c, .cc, .cpp, .cxx, and .cu input file into an object file.
        //     // .flag("-c")
        //     .flag("-dlink")
        //     .file("instrumentation/inject_funcs.cu")
        //     .warnings(false)
        //     // .shared_flag(true)
        //     // .compile("instrumentation");
        //     .compile("instrumentation.o");

        // link the tool into a shared lib

        // .object("/path/to/my/o/file/foo.o")

        // this looks very promising, let nvcc perform the device linking as well
        // because the final linking will be done by rust
        //
        //
        // https://stackoverflow.com/questions/22115197/dynamic-parallelism-undefined-reference-to-cudaregisterlinkedbinary-linking
        //
        // nvcc -arch=sm_35 -rdc=true -c file.cu
        // nvcc -arch=sm_35 -dlink -o file_link.o file.o -lcudadevrt -lcudart
        // g++ file.o file_link.o main.cpp -L<path> -lcudart -lcudadevrt
    }

    // --export-dynamic-symbol-list
    println!(
        "cargo:rustc-link-search=native={}",
        "/usr/lib/x86_64-linux-gnu"
    );
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=cudadevrt");
    // println!("cargo:rustc-link-lib=dylib=stdc++");

    // println!("cargo:rustc-link-lib=dylib=instrumentation");
    // println!("cargo:rustc-link-lib=dylib=instrumentation");
}
