fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rerun-if-changed=instrumentation");

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
}
