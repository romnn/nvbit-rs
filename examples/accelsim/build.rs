#![allow(warnings)]
use std::path::PathBuf;

pub fn nvbit_include() -> PathBuf {
    PathBuf::from(std::env::var("DEP_NVBIT_INCLUDE").expect("nvbit include path"))
        .canonicalize()
        .expect("canonicalize path")
}

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

fn generate_bindings() {
    let mut builder = bindgen::Builder::default()
        .clang_args([
            "-x",
            "c++",
            "-std=c++11",
            &format!("-I{}", nvbit_include().display()),
        ])
        .header("instrumentation/common.h");

    let bindings = builder.generate().expect("generating bindings");
    bindings
        .write_to_file(output_path().join("bindings.rs"))
        .expect("writing bindings failed");
}

fn main() {
    // rerun if the build script changes
    println!("cargo:rerun-if-changed=build.rs");

    // rerun if the instrumentation changes
    println!("cargo:rerun-if-changed=instrumentation");

    generate_bindings();

    // compile the injection functions
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

    // compile the injection functions
    let inject_obj = output_path().join("inject_funcs.o");
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
            "instrumentation/inject_funcs.cu",
            "-o",
            &inject_obj.to_string_lossy(),
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
            &inject_obj.to_string_lossy(),
            "-o",
            &dev_obj_link.to_string_lossy(),
        ])
        .output()
        .expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());

    let lib = output_path().join("libinstrumentation.a");
    std::process::Command::new("ar")
        .args([
            "cru",
            &lib.to_string_lossy(),
            &tool_obj.to_string_lossy(),
            &inject_obj.to_string_lossy(),
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
    println!("cargo:rustc-link-lib=static:+whole-archive=instrumentation");
}
