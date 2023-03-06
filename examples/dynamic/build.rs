use std::path::PathBuf;
// use regex::Regex;
// use std::process::Command;

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
        if info == "mem_access_t" {
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
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rerun-if-changed=instrumentation");

    generate_bindings();

    // compile the instrumentation function
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

    let lib = output_path().join("libinstrumentation.a");
    std::process::Command::new("ar")
        .args([
            "cru",
            &lib.to_string_lossy(),
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
    println!("cargo:rustc-link-lib=static:+whole-archive=instrumentation");

    
    // let cuobjdump_bin_path: PathBuf = std::env::var("CUDA_INSTALL_PATH")
    //     .ok()
    //     .map(|cuda_install| [&cuda_install, "bin", "cuobjdump"].iter().collect())
    //     .and_then(|bin: PathBuf| if bin.exists() { Some(bin) } else { None })
    //     .unwrap_or(PathBuf::from("cuobjdump"));
    // let cuobjdump_ptx_files = Command::new(&cuobjdump_bin_path)
    //     .arg("-lptx")
    //     .arg(&inject_obj)
    //     .output()
    //     .unwrap();
    // let cuobjdump_ptx_files = String::from_utf8(cuobjdump_ptx_files.stdout).unwrap();

    // lazy_static::lazy_static! {
    //     static ref PTX_FILE_REG: Regex = Regex::new(r"PTX file\s*\d*:\s*(?P<ptx_file>\S*)\s*").unwrap();
    // }

    // let ptx_files: Vec<&str> = PTX_FILE_REG
    //     .captures_iter(&cuobjdump_ptx_files)
    //     .filter_map(|cap| cap.name("ptx_file"))
    //     .map(|cap| cap.as_str())
    //     .collect();

    // // println!("cargo:warning={:?}", &ptx_files);

    // for ptx_file in ptx_files {
    //     Command::new(&cuobjdump_bin_path)
    //         .current_dir(output_path())
    //         .arg("-xptx")
    //         .arg(&ptx_file)
    //         .arg(&inject_obj)
    //         .output()
    //         .unwrap();
    // }
}
