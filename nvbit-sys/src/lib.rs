pub mod nvbit;
pub mod utils;

#[allow(warnings, dead_code)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings/nvbit.rs"));
}
