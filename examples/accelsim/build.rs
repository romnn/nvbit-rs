use std::path::PathBuf;

#[must_use]
pub fn nvbit_include() -> PathBuf {
    PathBuf::from(std::env::var("DEP_NVBIT_INCLUDE").expect("nvbit include path"))
        .canonicalize()
        .expect("canonicalize path")
}

#[must_use]
pub fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("cargo out dir"))
        .canonicalize()
        .expect("canonicalize path")
}

#[must_use]
pub fn manifest_path() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir"))
        .canonicalize()
        .expect("canonicalize path")
}

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
            &format!("-I{}", nvbit_include().display()),
        ])
        .generate_comments(false)
        .rustified_enum(".*")
        .rustfmt_bindings(true)
        .parse_callbacks(Box::new(ParseCallbacks {}))
        .header("instrumentation/common.h");

    let bindings = builder.generate().expect("generating bindings");
    bindings
        .write_to_file(output_path().join("bindings.rs"))
        .expect("writing bindings failed");
}

fn main() {
    // for (key, value) in std::env::vars() {
    //     println!("cargo::warning={key}: {value}");
    //     if key.to_lowercase().contains("dep_") {
    //         println!("cargo::warning={key}: {value}");
    //     }
    // }

    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rerun-if-changed=instrumentation");

    generate_bindings();

    // compile the instrumentation functions
    let instrument_inst_obj = output_path().join("instrument_inst.o");
    let result = std::process::Command::new("nvcc")
        .args([
            &format!("-I{}", nvbit_include().display()),
            "-maxrregcount=24",
            "-Xptxas",
            "-astoolspatch",
            "--keep-device-functions",
            "-Xcompiler",
            "-fPIC",
            "-c",
            "instrumentation/instrument_inst.cu",
            "-o",
            &instrument_inst_obj.to_string_lossy(),
        ])
        .output()
        .expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());

    // compile the helper functions for flushing the channel etc.
    let tool_obj = output_path().join("tool.o");
    let result = std::process::Command::new("nvcc")
        .args([
            &format!("-I{}", nvbit_include().display()),
            &format!("-I{}", manifest_path().join("instrumentation").display()),
            "-Xcompiler",
            "-fPIC",
            "-dc",
            "-c",
            "instrumentation/tool.cu",
            "-o",
            &tool_obj.to_string_lossy(),
        ])
        .output()
        .expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());

    let dev_obj_link = output_path().join("dev_link.o");
    let result = std::process::Command::new("nvcc")
        .args([
            &format!("-I{}", nvbit_include().display()),
            "-Xcompiler",
            "-fPIC",
            "-dlink",
            &tool_obj.to_string_lossy(),
            &instrument_inst_obj.to_string_lossy(),
            "-o",
            &dev_obj_link.to_string_lossy(),
        ])
        .output()
        .expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());

    // the static lib name must be unique per example to avoid name conflicts
    let static_lib = format!(
        "{}instrumentation",
        manifest_path()
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap()
    );
    std::process::Command::new("ar")
        .args([
            "cru",
            &output_path()
                .join(format!("lib{static_lib}.a"))
                .to_string_lossy(),
            &tool_obj.to_string_lossy(),
            &instrument_inst_obj.to_string_lossy(),
            &dev_obj_link.to_string_lossy(),
        ])
        .output()
        .expect("linking failed");
    println!(
        "linking result: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.status.success());

    println!("cargo:rustc-link-search=native={}", output_path().display());
    println!("cargo:rustc-link-lib=static:+whole-archive={static_lib}");
}
