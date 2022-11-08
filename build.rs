#![allow(warnings, dead_code)]

use bindgen::EnumVariation;
use std::path::PathBuf;

pub fn output() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

fn main() {
    // todo: force cpp with clang args
    // todo: download the correct nvbit into cargo dir
    // find out the host architecture
    // get the version via env?
    // create workspaces: nvbit-sys and nvbit-rs
    // add example binary tracer tool with options

    // nvcc -ccbin=g++ -D_FORCE_INLINES -dc -c -std=c++11 -I../nvbit_release/core -Xptxas -cloning=no -Xcompiler -Wall -arch=sm_35 -O3 -Xcompiler -fPIC tracer_tool.cu -o tracer_tool.o

    // cxx_build::bridge("src/lib.rs")
    //     .file("nvbit/nvbit.cu")
    //     // .flag_if_supported("-std=c++14")
    //     .compile("cxxbridge-demo");

    cc::Build::new()
        // .cuda(true)
        // .compiler("g++")
        .compiler("nvcc")
        .no_default_flags(true)
        .flag("-ccbin=g++")
        .warnings(false)
        // .flag("-std=c++11")
        .flag("-lnvbit")
        .flag("-lcuda")
        .flag("-lcudart_static")
        // .flag("-cudart=shared")
        // .flag("-gencode")
        // .flag("arch=compute_61,code=sm_61")
        .include("nvbit_release/core")
        .file("nvbit/nvbit.cu")
        .compile("nvbit_tool.a");

    // println!("cargo:rustc-link-search=native={}", "nvbit_release/core/");
    // println!("cargo:rustc-link-lib=static=nvbit");

    let clang_args = vec!["-x", "c++"];
    let mut builder = bindgen::Builder::default()
        .clang_args(&clang_args)
        .allowlist_type("Instr")
        .allowlist_type("basic_block_t")
        .allowlist_type("CFG_t")
        // avoid difficulties with C++ std::vector for example
        .opaque_type("std::.*") 
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
        .allowlist_type("^CU.*")
        // .blocklist_function("acoshl")
        .generate_comments(false)
        .rustified_enum("*")
        .prepend_enum_name(false)
        .derive_eq(true)
        .derive_default(true)
        .derive_hash(true)
        .derive_ord(true)
        // .size_t_is_usize(true)
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
    bindings
        .write_to_file(output().join("bindings.rs"))
        .expect("writing bindings failed");

    // so we dont have to find the bindings in the temp output folder each time
    bindings
        .write_to_file("./bindings_debug.rs")
        .expect("writing bindings failed");

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

    // println!("cargo:rustc-link-lib=static=nvbit_release/core/libnvbit.a");
    println!("cargo:rustc-link-search=native={}", "nvbit_release/core/");
    println!("cargo:rustc-link-lib=static=nvbit");

    println!("cargo:rustc-link-search=native={}", "/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");

    // println!("cargo:rustc-link-lib=static=cuda");
    // println!("cargo:rustc-link-lib=static=cudart");
    
    // .flag("-lnvbit")
    // .flag("-lcuda")
    // .flag("-lcudart_static")
    // .flag("-cudart=shared")

    // println!("cargo:rustc-link-lib=static=nvbit_tool");
    // 
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=nvbit/nvbit.cu");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");

    // --whitelist-type="^cuda.*" \
    // --whitelist-type="^surfaceReference" \
    // --whitelist-type="^textureReference" \
    // --whitelist-var="^cuda.*" \
    // --whitelist-function="^cuda.*" \
    // --default-enum-style=rust \
    // --no-doc-comments \
    // --with-derive-default \
    // --with-derive-eq \
    // --with-derive-hash \
    // --with-derive-ord \
    // /opt/cuda/include/cuda_runtime.h
}
