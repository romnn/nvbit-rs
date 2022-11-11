fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=inc/nvbit.h");

    cxx_build::bridge("src/lib.rs")
        .cpp(true)
        .file("inc/nvbit.h")
        .warnings(false)
        .compile("nvbitbridge");
}
