pub mod nvbit;
pub mod utils;

#[allow(
    warnings,
    clippy::all,
    clippy::pedantic,
    clippy::restriction,
    clippy::nursery
)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings/nvbit.rs"));
}

pub use bindings::*;
