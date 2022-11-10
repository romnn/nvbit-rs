fn main() {
    // nvcc -ccbin=g++ -D_FORCE_INLINES -dc -c -std=c++11 -I../nvbit_release/core -Xptxas -cloning=no -Xcompiler -Wall -arch=sm_35 -O3 -Xcompiler -fPIC tracer_tool.cu -o tracer_tool.o
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=inc/nvbit.h");
    // println!("cargo:rerun-if-changed=nvbit/nvbit.h");

    cxx_build::bridge("src/lib.rs")
        .cpp(true)
        // .include("nvbit_release/core")
        // include CUDA stuff
        // .include("/usr/lib/x86_64-linux-gnu")
        // "nvbit_release/core")
        .file("inc/nvbit.h")
        // .file("nvbit/nvbit.cu")
        // .compiler("nvcc")
        // .no_default_flags(true)
        // .flag("-lnvbit")
        // .flag("-lcuda")
        // .flag("-lcudart_static")
        .warnings(false)
        // .flag_if_supported("-std=c++14")
        .compile("nvbitbridge");
}
