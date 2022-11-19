#![allow(warnings)]

pub mod instr;
pub mod cfg;
pub mod cuda;
pub mod nvbit;

pub use cfg::*;
pub use nvbit::*;
pub use cuda::*;
pub use instr::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {}
}
