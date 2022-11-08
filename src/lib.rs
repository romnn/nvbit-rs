#![allow(warnings, dead_code)]

#[allow(warnings, dead_code)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::*;

extern "C" {
    fn init_nvbit();
}

pub fn init() {
    unsafe { init_nvbit() }
}

// #[test]
// fn included_nvbit_lib() {
//     unsafe {
//         assert_eq!(this_must_be_present(), 42);
//     }
// }

// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" {
        fn this_must_be_present() -> i64;
    }

    #[test]
    fn included_nvbit_lib() {
        unsafe {
            assert_eq!(this_must_be_present(), 42);
        }
    }
}
