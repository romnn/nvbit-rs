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

fn generate_channel_bindings() {
    let mut builder = bindgen::Builder::default()
        .clang_args(["-x", "c++", "-std=c++11"])
        // avoid difficulties with C++ std::vector for example
        .opaque_type("std::.*")
        .blocklist_type("std::.*")
        .allowlist_type("ChannelDev")
        .allowlist_type("ChannelHost")
        .generate_comments(true)
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
        .header("nvbit/channel.h");

    let bindings = builder.generate().expect("generating bindings failed");

    let bindings_path = output_path().join("bindings/channel.rs");
    create_dirs(&bindings_path);
    bindings
        .write_to_file(&bindings_path)
        .expect("writing bindings failed");

    let debug_bindings_path = manifest_path().join("debug/channel_bindings.rs");
    create_dirs(&debug_bindings_path);
    bindings
        .write_to_file(&debug_bindings_path)
        .expect("writing bindings failed");
}

fn generate_nvbit_bindings() {
    let mut builder = bindgen::Builder::default()
        .clang_args(["-x", "c++", "-std=c++11"])
        .allowlist_type("Instr")
        .allowlist_type("basic_block_t")
        .allowlist_type("CFG_t")
        // avoid difficulties with C++ std::vector for example
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
        .generate_comments(true)
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
        .header("nvbit_release/core/nvbit.h");

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
    if true || !nvbit_path.is_dir() {
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

fn main() {
    let utils = std::env::var("CARGO_FEATURE_UTILS").is_ok();

    if utils {
        println!("cargo:rustc-cfg=nvbit_utils");
    }

    // rerun if the build script changes
    rerun_if("build.rs");

    // rerun if the cxx bridge definition changes
    rerun_if("src/");

    // rerun if the shims change
    rerun_if("nvbit/");

    // rerun if the CUDA library path changes
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");

    // communicate the includes of nvbit to other crates
    // println!("cargo:root={}", dst.to_str().unwrap());
    // println!("cargo:rustc-link-search=native={}", lib.to_str().unwrap());
    // println!("cargo:include={}/include", dst.to_str().unwrap());

    generate_nvbit_bindings();
    generate_channel_bindings();

    let nvbit_version = std::env::var("NVBIT_VERSION").unwrap_or(NVBIT_VERSION.to_string());

    // supported: x86_64, aarch64, ppc64le
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("cargo target arch");
    let supported = vec!["x86_64", "aarch64", "ppc64le"];
    if !supported.contains(&target_arch.as_str()) {
        panic!(
            "unsupported target architecture {} (nvbit supports {:?})",
            target_arch, supported
        );
    }
    // match target_arch {
    //     "x86_64" | "aarch64" | "ppc64le" => {},
    //     _ => {
    //         panic!("unsupported nvbit architecture
    //     },
    // }

    let vendored_nvbit = download_nvbit(nvbit_version, target_arch);

    // todo: download the correct nvbit into cargo dir
    // find out the host architecture
    // get the version via env?
    // create workspaces: nvbit-sys and nvbit-rs
    // add example binary tracer tool with options

    // nvcc -ccbin=g++ -D_FORCE_INLINES -dc -c -std=c++11 -I../nvbit_release/core -Xptxas
    // -cloning=no -Xcompiler -Wall -arch=sm_35 -O3 -Xcompiler -fPIC tracer_tool.cu -o tracer_tool.o

    // panic!("{}", &manifest_path().join("src/lib.rs").display());
    // cxx_build::bridge(&manifest_path().join("src/lib.rs"))

    let nvbit_obj = output_path().join("nvbit.o");
    let nvbit_obj_link = output_path().join("nvbit_link.o");

    if false {
        let result = std::process::Command::new("nvcc")
            .args([
                "-Invbit_release/core",
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

    // let fuck = output_path().join("libnvbitbridge.a");
    cxx_build::bridge("src/lib.rs")
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

    println!(
        "cargo:rustc-link-search=native={}",
        &manifest_path()
            .join("nvbit_release/core/")
            .to_string_lossy()
    );
    println!("cargo:rustc-link-lib=static=nvbit");

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
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
