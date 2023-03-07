use nvbit_build;

#[derive(Debug)]
struct ParseCallbacks {}

impl bindgen::callbacks::ParseCallbacks for ParseCallbacks {
    fn add_derives(&self, info: &str) -> Vec<String> {
        if info == "inst_trace_t" {
            vec![
                "serde::Serialize".to_string(),
                "serde::Deserialize".to_string(),
            ]
        } else {
            vec![]
        }
    }
}

fn generate_bindings() {
    let builder = bindgen::Builder::default()
        .clang_args([
            "-x",
            "c++",
            "-std=c++11",
            &format!("-I{}", nvbit_build::nvbit_include().display()),
        ])
        .generate_comments(false)
        .rustified_enum(".*")
        .rustfmt_bindings(true)
        .parse_callbacks(Box::new(ParseCallbacks {}))
        .header("instrumentation/common.h");

    let bindings = builder.generate().expect("generating bindings");
    bindings
        .write_to_file(nvbit_build::output_path().join("bindings.rs"))
        .expect("writing bindings failed");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rerun-if-changed=instrumentation");

    generate_bindings();

    // the lib name must be unique per example to avoid name conflicts
    let lib = format!(
        "{}instrumentation",
        nvbit_build::manifest_path()
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap()
    );

    nvbit_build::Build::new()
        .include(nvbit_build::nvbit_include())
        .include(nvbit_build::manifest_path().join("instrumentation"))
        .instrumentation_source("instrumentation/instrument_inst.cu")
        .source("instrumentation/tool.cu")
        .compile(lib)
        .unwrap();

    // compile the instrumentation functions
    // let instrument_inst_obj = nvbit_build::output_path().join("instrument_inst.o");
    // let result = std::process::Command::new("nvcc")
    //     .args([
    //         &format!("-I{}", nvbit_build::nvbit_include().display()),
    //         "-maxrregcount=24",
    //         "-Xptxas",
    //         "-astoolspatch",
    //         "--keep-device-functions",
    //         "-Xcompiler",
    //         "-fPIC",
    //         "-c",
    //         "instrumentation/instrument_inst.cu",
    //         "-o",
    //         &instrument_inst_obj.to_string_lossy(),
    //     ])
    //     .output()
    //     .expect("nvcc failed");
    // println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    // assert!(result.status.success());

    // compile the helper functions for flushing the channel etc.
    // let tool_obj = nvbit_build::output_path().join("tool.o");
    // let result = std::process::Command::new("nvcc")
    //     .args([
    //         &format!("-I{}", nvbit_build::nvbit_include().display()),
    //         &format!(
    //             "-I{}",
    //             nvbit_build::manifest_path()
    //                 .join("instrumentation")
    //                 .display()
    //         ),
    //         "-Xcompiler",
    //         "-fPIC",
    //         "-dc",
    //         "-c",
    //         "instrumentation/tool.cu",
    //         "-o",
    //         &tool_obj.to_string_lossy(),
    //     ])
    //     .output()
    //     .expect("nvcc failed");
    // println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    // assert!(result.status.success());

    // let dev_obj_link = nvbit_build::output_path().join("dev_link.o");
    // let result = std::process::Command::new("nvcc")
    //     .args([
    //         &format!("-I{}", nvbit_build::nvbit_include().display()),
    //         "-Xcompiler",
    //         "-fPIC",
    //         "-dlink",
    //         &tool_obj.to_string_lossy(),
    //         &instrument_inst_obj.to_string_lossy(),
    //         "-o",
    //         &dev_obj_link.to_string_lossy(),
    //     ])
    //     .output()
    //     .expect("nvcc failed");
    // println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    // assert!(result.status.success());

    // the static lib name must be unique per example to avoid name conflicts
    // let static_lib = format!(
    //     "{}instrumentation",
    //     nvbit_build::manifest_path()
    //         .file_name()
    //         .and_then(std::ffi::OsStr::to_str)
    //         .unwrap()
    // );
    // std::process::Command::new("ar")
    //     .args([
    //         "cru",
    //         &nvbit_build::output_path()
    //             .join(format!("lib{static_lib}.a"))
    //             .to_string_lossy(),
    //         &tool_obj.to_string_lossy(),
    //         &instrument_inst_obj.to_string_lossy(),
    //         &dev_obj_link.to_string_lossy(),
    //     ])
    //     .output()
    //     .expect("linking failed");
    // println!(
    //     "linking result: {}",
    //     String::from_utf8_lossy(&result.stderr)
    // );
    // assert!(result.status.success());

    // println!(
    //     "cargo:rustc-link-search=native={}",
    //     nvbit_build::output_path().display()
    // );
    // println!("cargo:rustc-link-lib=static:+whole-archive={static_lib}");
}
