#![allow(warnings)]

use bindgen::EnumVariation;
use std::path::{Path, PathBuf};

fn create_dirs(path: impl AsRef<Path>) {
    let dir = path.as_ref().parent().expect("get parent dir");
    std::fs::create_dir_all(&dir);
}

pub fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("cargo out dir"))
        .canonicalize()
        .expect("canonicalize")
}

pub fn manifest_path() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir"))
        .canonicalize()
        .expect("canonicalize")
}

fn rerun_if(path: impl AsRef<Path>) {
    if path.as_ref().is_dir() {
        for entry in std::fs::read_dir(path.as_ref()).expect("read_dir") {
            rerun_if(&entry.expect("entry").path());
        }
    } else {
        println!("cargo:rerun-if-changed={}", path.as_ref().display());
    }
}

fn generate_utils_bindings() {
    let mut builder = bindgen::Builder::default()
        .clang_args(["-x", "c++", "-std=c++11"])
        // avoid difficulties with C++ std::vector for example
        // .opaque_type("std::.*")
        // .blocklist_type("std::.*")
        .allowlist_type("ChannelDev")
        .allowlist_type("ChannelHost")
        .generate_comments(false)
        .rustified_enum("*")
        .prepend_enum_name(false)
        .derive_eq(true)
        .derive_default(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: false,
        })
        // .header("nvbit_release/core/utils/channel.hpp")
        // .header("nvbit_release/core/utils/utils.h");
        .header("nvbit/utils.h");

    let bindings = builder.generate().expect("generating bindings failed");

    let bindings_path = output_path().join("bindings/utils.rs");
    create_dirs(&bindings_path);
    bindings
        .write_to_file(&bindings_path)
        .expect("writing bindings failed");

    let debug_bindings_path = manifest_path().join("debug/utils_bindings.rs");
    create_dirs(&debug_bindings_path);
    bindings
        .write_to_file(&debug_bindings_path)
        .expect("writing bindings failed");
}

fn generate_nvbit_bindings(includes: impl IntoIterator<Item = PathBuf>) {
    let mut clang_args: Vec<String> = vec![
        "-x".to_string(),
        "c++".to_string(),
        "-std=c++11".to_string(),
    ];
    for inc in includes.into_iter() {
        clang_args.push(format!("-I{}", inc.display()));
    }
    dbg!(&clang_args);
    let mut builder = bindgen::Builder::default()
        .clang_args(clang_args)
        // .detect_include_paths(false)
        // .allowlist_type("Instr")
        .allowlist_var("NVBIT_VERSION")
        .allowlist_type("basic_block_t")
        .allowlist_type("CFG_t")
        // avoid difficulties with C++ std::vector for example
        // .blocklist_type("Instr")
        // .opaque_type("Instr")
        .opaque_type("std::.*")
        .blocklist_type("std::.*")
        // .allowlist_type("ipoint_t")
        // .allowlist_type("nvbit_api_cuda_t")
        // .allowlist_type("nvbit_*")
        // .allowlist_type("nvbit_api_cuda_t")
        .allowlist_function("^nvbit_.*")
        // .allowlist_function("nvbit_at_init")
        // .allowlist_function("nvbit_at_term")
        // .allowlist_function("nvbit_at_ctx_init")
        // .allowlist_function("nvbit_at_ctx_term")
        // .allowlist_function("nvbit_at_cuda_event")
        // .allowlist_function("nvbit_get_related_functions")
        // .allowlist_function("nvbit_get_related_functions")
        // .allowlist_type("CUcontext")
        // .allowlist_type("CUcontext")
        // .allowlist_type("CUresult")
        .allowlist_type("imm_uint64")
        .allowlist_type("imm_double")
        .allowlist_type("reg")
        .allowlist_type("pred")
        .allowlist_type("cbank")
        .allowlist_type("generic")
        .allowlist_type("mref")
        .allowlist_type("operand_t")
        .allowlist_type("RegModifierType")
        .allowlist_type("RegModifierTypeStr")
        .allowlist_type("OperandType")
        .allowlist_type("OperandTypeStr")
        .allowlist_type("MemorySpace")
        .allowlist_type("MemorySpaceStr")
        .allowlist_function("^cu.*")
        .allowlist_type("^cu.*")
        .allowlist_type("^CU.*")
        .generate_comments(false)
        .rustified_enum("*")
        .prepend_enum_name(false)
        .derive_eq(true)
        .derive_default(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: false,
        })
        // .header("nvbit_release/core/utils/channel.hpp")
        // .header("nvbit_release/core/utils/utils.h")
        // this tool here gives problems, because it uses __device__ etc.
        // not really used but must be included?
        // .header("nvbit_release/core/nvbit_tool.h")
        // .header("nvbit_release/core/generated_cuda_meta.h")
        // .header("nvbit_release/core/tools_cuda_api_meta.h")
        // .header("nvbit_release/core/instr_types.h")
        // .header("nvbit_release/core/nvbit.h");
        .header("nvbit/bindings/nvbit_bindings.h");

    // .header("nvbit_release/core/nvbit_tool.h")
    // .header("nvbit_release/core/instr_types.h")
    // .header("nvbit_release/core/nvbit_reg_rw.h");

    // generate bindings
    let bindings = builder.generate().expect("generating bindings failed");

    // write generated bindings to $OUT_DIR/bindings.rs
    let bindings_path = output_path().join("bindings/nvbit.rs");
    create_dirs(&bindings_path);
    bindings
        .write_to_file(&bindings_path)
        .expect("writing bindings failed");

    let debug_bindings_path = manifest_path().join("debug/nvbit_bindings.rs");
    create_dirs(&debug_bindings_path);
    bindings
        .write_to_file(&debug_bindings_path)
        .expect("writing bindings failed");
}

static NVBIT_RELEASES: &'static str = "https://github.com/NVlabs/NVBit/releases/download";
static NVBIT_VERSION: &'static str = "1.5.5";

pub fn decompress_tar_bz2(src: impl AsRef<Path>, dest: impl AsRef<Path>) {
    let compressed = std::fs::File::open(src).expect("open file");
    let stream = bzip2::read::BzDecoder::new(compressed);
    let mut archive = tar::Archive::new(stream);
    archive.unpack(&dest).expect("unpack tar");
}

fn download_nvbit(version: impl AsRef<str>, arch: impl AsRef<str>) -> PathBuf {
    use std::fs::File;

    let nvbit_release_name = format!("nvbit-Linux-{}-{}", arch.as_ref(), version.as_ref());
    let nvbit_release_archive_name = format!("{}.tar.bz2", nvbit_release_name);
    let nvbit_release_archive_url = reqwest::Url::parse(&format!(
        "{}/{}/{}",
        NVBIT_RELEASES,
        version.as_ref(),
        nvbit_release_archive_name,
    ))
    .expect("parse url");
    println!("cargo:warning={}", nvbit_release_archive_url);

    let archive_path = output_path().join(nvbit_release_archive_name);
    let nvbit_path = output_path().join(nvbit_release_name);

    // check if the archive already exists
    let force = false;
    if force || !nvbit_path.is_dir() {
        println!("cargo:warning={}", archive_path.display());
        let _ = std::fs::remove_file(&archive_path);
        let mut nvbit_release_archive_file = File::create(&archive_path).expect("create file");
        reqwest::blocking::get(nvbit_release_archive_url)
            .expect("get nvbit request")
            .copy_to(&mut nvbit_release_archive_file)
            .expect("copy nvbit archive");

        let _ = std::fs::remove_file(&nvbit_path);
        decompress_tar_bz2(&archive_path, &nvbit_path);
    }

    nvbit_path
}

fn build_tool() {
    let nvbit_obj = output_path().join("nvbit.o");
    let nvbit_obj_link = output_path().join("nvbit_link.o");

    let result = std::process::Command::new("nvcc")
        .args([
            "-Invbit_release/core",
            // "-G",
            // "-lineinfo",
            "-Xcompiler",
            "-fPIC",
            "-c",
            // "nvbit-sys/nvbit_release/core/nvbit_tool.h",
            // "nvbit-sys/nvbit_release/core/nvbit_tool.h",
            "nvbit/nvbit_tool.cu",
            // "nvbit/nvbit.cu",
            "-o",
            &nvbit_obj.to_string_lossy(),
        ])
        .output()
        .expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());

    let result = std::process::Command::new("nvcc")
        .args([
            // "-G",
            // "-lineinfo",
            "-Xcompiler",
            "-fPIC",
            "-dlink",
            &nvbit_obj.to_string_lossy(),
            "-o",
            &nvbit_obj_link.to_string_lossy(),
        ])
        .output()
        .expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());
}

fn build_utils_bridge() {
    let utils_obj = output_path().join("utils.o");
    let result = std::process::Command::new("nvcc")
        .args([
            // "-I../", // nvbit-sys/nvbit_release/core",
            "-Invbit_release/core",
            // "-G",
            // "-lineinfo",
            // "-Invbit-sys/nvbit_release/core",
            // "-maxrregcount=24",
            // "-Xptxas",
            // "-astoolspatch",
            // "--keep-device-functions",
            // "-c",
            "-Xcompiler",
            "-fPIC",
            // "-dc",
            "-c",
            // "instrumentation/inject_funcs.cu",
            "nvbit/utils_test.cu",
            // "nvbit_release/core/channel.hpp",
            // "nvbit_release/core/utils.h",
            "-o",
            &utils_obj.to_string_lossy(),
        ])
        .output()
        .expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());

    let dev_obj_link = output_path().join("dev_link.o");
    let result = std::process::Command::new("nvcc")
        .args([
            // "-G",
            // "-lineinfo",
            "-Xcompiler",
            "-fPIC",
            "-dlink",
            &utils_obj.to_string_lossy(),
            // "-lnvbit"
            // "-lcuda",
            // "-shared",
            "-o",
            &dev_obj_link.to_string_lossy(),
        ])
        .output()
        .expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());

    cxx_build::bridge("src/utils.rs")
        .compiler("nvcc")
        // include CUDA stuff
        // .include("/usr/lib/x86_64-linux-gnu")
        .no_default_flags(true)
        .warnings(false)
        .flag("-x")
        .flag("cu")
        // .flag("-G")
        // .flag("-lineinfo")
        .flag("-Xcompiler")
        .flag("-fPIC")
        .file("nvbit/utils.cu")
        .object(&utils_obj)
        .object(&dev_obj_link)
        .static_flag(true)
        .compile("utilsbridge");
    println!("cargo:rustc-link-lib=static=utilsbridge");
    // println!("cargo:rustc-link-lib=static:+whole-archive=utilsbridge");
}

fn build_nvbit_bridge() {
    cxx_build::bridge("src/nvbit.rs")
        .compiler("nvcc")
        // .include("nvbit_release/core")
        // include CUDA stuff
        // .include("/usr/lib/x86_64-linux-gnu")
        // "nvbit_release/core")
        // .file("nvbit/inject_funcs.cu")
        // .file(&manifest_path().join("nvbit/nvbit.cu"))
        .no_default_flags(true)
        // .file("nvbit/nvbit.cu")
        // .flag("-lnvbit")
        // .flag("-lcuda")
        // .flag("-lcudart_static")
        .warnings(false)
        // .flag("-G")
        // .flag("-lineinfo")
        .flag("-Xcompiler")
        .flag("-fPIC")
        // .flag_if_supported("-std=c++14")
        // .flag("-dlink")
        // .file("nvbit/channel.h")
        .file("nvbit/nvbit.cu")
        // .file("nvbit-sys/nvbit_release/core/channel.hpp")
        // .object(&nvbit_obj)
        // .object(&nvbit_obj_link)
        // .link_lib_modifier("+whole-archive")
        .compile("nvbitbridge");
}

fn main() {
    // rerun if the build script changes
    rerun_if("build.rs");

    // rerun if the cxx bridge definition changes
    rerun_if("src/");

    // rerun if the shims change
    rerun_if("nvbit/");

    // rerun if the nvbit source changes (TODO: for dev only)
    rerun_if("nvbit_release/");

    // rerun if the CUDA library path changes
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");

    let utils = std::env::var("CARGO_FEATURE_UTILS").is_ok();
    if utils {
        println!("cargo:rustc-cfg=nvbit_utils");
    }

    let nvbit_version = std::env::var("NVBIT_VERSION").unwrap_or(NVBIT_VERSION.to_string());
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("cargo target arch");

    // check if the target architecture supports nvbit
    let supported = vec!["x86_64", "aarch64", "ppc64le"];
    if !supported.contains(&target_arch.as_str()) {
        panic!(
            "unsupported target architecture {} (nvbit supports {:?})",
            target_arch, supported
        );
    }

    let vendored_nvbit = download_nvbit(nvbit_version, target_arch);

    // communicate the includes of nvbit to other crates
    println!("cargo:root={}", output_path().display());
    let nvbit_include_path = vendored_nvbit.join("nvbit_release/core/");

    let nvbit_include_path = manifest_path().join("nvbit_release/core/");
    println!(
        "cargo:rustc-link-search=native={}",
        nvbit_include_path.display()
    );
    println!("cargo:include={}", nvbit_include_path.display());

    generate_nvbit_bindings([nvbit_include_path]);
    // generate_utils_bindings();

    build_nvbit_bridge();
    build_utils_bridge();

    // nvcc -ccbin=g++ -D_FORCE_INLINES -dc -c -std=c++11 -I../nvbit_release/core -Xptxas
    // -cloning=no -Xcompiler -Wall -arch=sm_35 -O3 -Xcompiler -fPIC tracer_tool.cu -o tracer_tool.o

    // let nvbit_obj_link = output_path().join("nvbit_link.o");
    // let result = // std::process::Command::new("nvcc")
    //     cc::Build::new()
    //     .compiler("nvcc")
    //     .no_default_flags(true)
    //     .warnings(false)
    //     .flag("-Xcompiler")
    //     .flag("-fPIC")
    //     .flag("-dlink")
    //     .file(&fuck)
    //         // "-o",
    //         // &nvbit_obj_link.to_string_lossy(),
    //     // ])
    //     .compile("nvbitbridge_dev");
    //
    // .output()
    // .expect("nvcc failed");
    // println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    // assert!(result.status.success());

    // cc::Build::new()
    //     // .cuda(true)
    //     // .compiler("g++")
    //     .compiler("nvcc")
    //     .no_default_flags(true)
    //     .flag("-ccbin=g++")
    //     .warnings(false)
    //     // .flag("-std=c++11")
    //     .flag("-lnvbit")
    //     .flag("-lcuda")
    //     .flag("-lcudart_static")
    //     // .flag("-cudart=shared")
    //     // .flag("-gencode")
    //     // .flag("arch=compute_61,code=sm_61")
    //     .include("nvbit_release/core")
    //     .file("nvbit/nvbit.cu")
    //     .compile("nvbit_tool.a");

    // println!(
    //     "cargo:rustc-link-search=native={}",
    //     &manifest_path()
    //         .join("nvbit_release/core/")
    //         .to_string_lossy()
    // );

    // if cfg!(target_os = "windows") {
    //     println!(
    //         "cargo:rustc-link-search=native={}",
    //         find_cuda_windows().display()
    //     );
    // } else {
    //     for path in find_cuda() {
    //         println!("cargo:rustc-link-search=native={}", path.display());
    //     }
    // };

    // println!(
    //     "cargo:rustc-link-search=native={}",
    //     manifest_path().join("nvbit_release/core/").display()
    // );
    // println!("cargo:rustc-link-lib=static:+whole-archive=nvbit");
    // println!("cargo:rustc-link-lib=static:+whole-archive=nvbitbridge");

    println!(
        "cargo:rustc-link-search=native={}",
        "/usr/lib/x86_64-linux-gnu"
    );

    println!("cargo:rustc-link-lib=static=nvbit");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
