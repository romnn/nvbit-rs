mod find_cuda;

use std::env;
use std::path::{Path, PathBuf};

static NVBIT_RELEASES: &str = "https://github.com/NVlabs/NVBit/releases/download";
static NVBIT_VERSION: &str = "1.5.5";

fn create_dirs(path: impl AsRef<Path>) {
    let dir = path.as_ref().parent().expect("get parent dir");
    std::fs::create_dir_all(dir).expect("create dirs");
}

fn output_path() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").expect("cargo out dir"))
        .canonicalize()
        .expect("canonicalize")
}

#[allow(dead_code)]
fn manifest_path() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir"))
        .canonicalize()
        .expect("canonicalize")
}

fn generate_nvbit_bindings<P: AsRef<Path>>(includes: impl IntoIterator<Item = P>) {
    let mut clang_args: Vec<String> = vec![
        "-x".to_string(),
        "c++".to_string(),
        "-std=c++11".to_string(),
    ];
    for inc in includes {
        clang_args.push(format!("-I{}", inc.as_ref().display()));
    }
    let builder = bindgen::Builder::default()
        .clang_args(clang_args)
        .allowlist_var("NVBIT_VERSION")
        .allowlist_type("basic_block_t")
        .allowlist_type("CFG_t")
        .opaque_type("std::.*")
        .blocklist_type("std::.*")
        .allowlist_function("^nvbit_.*")
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
        .rustified_enum(".*")
        .prepend_enum_name(false)
        .derive_eq(true)
        .derive_default(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .header("nvbit/bindings/nvbit_bindings.h");

    // generate bindings
    let bindings = builder.generate().expect("generating bindings failed");

    let bindings_path = output_path().join("bindings/nvbit.rs");
    create_dirs(&bindings_path);
    bindings
        .write_to_file(&bindings_path)
        .expect("writing bindings failed");

    // let debug_bindings_path = manifest_path().join("debug/nvbit_bindings.rs");
    // create_dirs(&debug_bindings_path);
    // bindings
    //     .write_to_file(&debug_bindings_path)
    //     .expect("writing bindings failed");
}

fn decompress_tar_bz2(src: impl AsRef<Path>, dest: impl AsRef<Path>) {
    let compressed = std::fs::File::open(src).expect("open file");
    let stream = bzip2::read::BzDecoder::new(compressed);
    let mut archive = tar::Archive::new(stream);
    archive.unpack(&dest).expect("unpack tar");
}

fn download_nvbit(version: impl AsRef<str>, arch: impl AsRef<str>) -> PathBuf {
    use std::fs::File;

    let nvbit_release_name = format!("nvbit-Linux-{}-{}", arch.as_ref(), version.as_ref());
    let nvbit_release_archive_name = format!("{nvbit_release_name}.tar.bz2");
    let nvbit_release_archive_url = reqwest::Url::parse(&format!(
        "{}/{}/{}",
        NVBIT_RELEASES,
        version.as_ref(),
        nvbit_release_archive_name,
    ))
    .expect("parse url");

    let archive_path = output_path().join(nvbit_release_archive_name);
    let nvbit_path = output_path().join(nvbit_release_name);

    // check if the archive already exists
    let force = false;
    if force || !nvbit_path.is_dir() {
        std::fs::remove_file(&archive_path).ok();
        let mut nvbit_release_archive_file = File::create(&archive_path).expect("create file");
        reqwest::blocking::get(nvbit_release_archive_url)
            .expect("get nvbit request")
            .copy_to(&mut nvbit_release_archive_file)
            .expect("copy nvbit archive");

        std::fs::remove_file(&nvbit_path).ok();
        decompress_tar_bz2(&archive_path, &nvbit_path);
    }

    nvbit_path
}

#[allow(dead_code)]
fn build_tool<P: AsRef<Path>>(includes: impl IntoIterator<Item = P>) {
    let nvbit_obj = output_path().join("nvbit.o");
    let nvbit_obj_link = output_path().join("nvbit_link.o");

    let mut cmd = std::process::Command::new("nvcc");
    for inc in includes {
        cmd.arg(format!("-I{}", inc.as_ref().display()));
    }
    cmd.args([
        "-Xcompiler",
        "-fPIC",
        "-c",
        "nvbit/nvbit_tool.cu",
        "-o",
        &nvbit_obj.to_string_lossy(),
    ]);
    let result = cmd.output().expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());

    let result = std::process::Command::new("nvcc")
        .args([
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

fn build_utils_bridge<P: AsRef<Path>>(includes: impl IntoIterator<Item = P>) {
    let includes: Vec<P> = includes.into_iter().collect();

    let utils_obj = output_path().join("utils.o");
    let mut cmd = std::process::Command::new("nvcc");
    for inc in &includes {
        cmd.arg(format!("-I{}", inc.as_ref().display()));
    }
    cmd.args([
        "-Xcompiler",
        "-fPIC",
        "-c",
        "nvbit/utils.cu",
        "-o",
        &utils_obj.to_string_lossy(),
    ]);
    let result = cmd.output().expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());

    let dev_obj_link = output_path().join("dev_link.o");
    let result = std::process::Command::new("nvcc")
        .args([
            "-Xcompiler",
            "-fPIC",
            "-dlink",
            &utils_obj.to_string_lossy(),
            "-o",
            &dev_obj_link.to_string_lossy(),
        ])
        .output()
        .expect("nvcc failed");
    println!("nvcc result: {}", String::from_utf8_lossy(&result.stderr));
    assert!(result.status.success());

    let mut cmd = cxx_build::bridge("src/utils.rs");
    for inc in &includes {
        cmd.flag(&format!("-I{}", inc.as_ref().display()));
    }
    cmd.compiler("nvcc")
        .no_default_flags(true)
        .warnings(false)
        .flag("-x")
        .flag("cu")
        .flag("-Xcompiler")
        .flag("-fPIC")
        .file("nvbit/utils_bridge.cu")
        .object(&utils_obj)
        .object(&dev_obj_link)
        .static_flag(true)
        .try_compile("utilsbridge")
        .expect("compile utils bridge");
    println!("cargo:rustc-link-lib=static=utilsbridge");
}

fn build_nvbit_bridge<P: AsRef<Path>>(includes: impl IntoIterator<Item = P>) {
    let mut cmd = cxx_build::bridge("src/nvbit.rs");
    cmd.compiler("nvcc");
    for inc in includes {
        cmd.flag(&format!("-I{}", inc.as_ref().display()));
    }

    cmd.no_default_flags(true)
        .warnings(false)
        .flag("-Xcompiler")
        .flag("-fPIC")
        .file("nvbit/nvbit_bridge.cu")
        .try_compile("nvbitbridge")
        .expect("compile nvbit bridge");
}

fn main() {
    // rerun if the build script changes
    println!("cargo:rerun-if-changed=build.rs");

    // rerun if the cxx bridge definition changes
    println!("cargo:rerun-if-changed=src/");

    // rerun if the shims change
    println!("cargo:rerun-if-changed=nvbit/");

    // rerun if the CUDA library path changes
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_UTILS");

    let utils = env::var("CARGO_FEATURE_UTILS").is_ok();
    if utils {
        println!("cargo:rustc-cfg=nvbit_utils");
    }

    let nvbit_version = env::var("NVBIT_VERSION").unwrap_or_else(|_| NVBIT_VERSION.to_string());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("cargo target arch");

    // check if the target architecture supports nvbit
    let supported = vec!["x86_64", "aarch64", "ppc64le"];
    assert!(
        supported.contains(&target_arch.as_str()),
        "unsupported target architecture {target_arch} (nvbit supports {supported:?})"
    );

    // download nvbit
    let vendored_nvbit = download_nvbit(nvbit_version, target_arch);

    // communicate the includes of nvbit to other crates
    println!("cargo:root={}", output_path().display());
    let nvbit_include_path = vendored_nvbit.join("nvbit_release/core/");
    // let nvbit_include_path = manifest_path().join("nvbit_release/core/");

    println!(
        "cargo:rustc-link-search=native={}",
        nvbit_include_path.display()
    );
    println!("cargo:include={}", nvbit_include_path.display());

    generate_nvbit_bindings([&nvbit_include_path]);

    build_nvbit_bridge([&nvbit_include_path]);
    build_utils_bridge([&nvbit_include_path]);

    let cuda_paths = find_cuda::find_cuda();
    // println!("cargo:warning=cuda paths: {cuda_paths:?}");
    for cuda_path in &cuda_paths {
        println!("cargo:rustc-link-search=native={}", cuda_path.display());
    }

    println!("cargo:rustc-link-lib=static=nvbit");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
}
