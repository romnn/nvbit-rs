#[cxx::bridge]
mod ffi {
    // Rust types and signatures exposed to C++.
    // we do not want to expose anything to C++
    extern "Rust" {
        // type CUfunction;
        // type MultiBuf;

        // fn next_chunk(buf: &mut MultiBuf) -> &[u8];
    }

    unsafe extern "C++" {
        include!("cxx-test/inc/nvbit.h");
        // include!("nvbit.h");
        // include!("nvbit/nvbit.cc");
        // include!("nvbit-sys/nvbit/nvbit.h");
        // include!("nvbit-sys/nvbit/nvbit.h");
        // include!("nvbit_sys/nvbit/nvbit.cc");
        // include!("nvbit_release/core/nvbit.h");

        // type CUfunction = super::bindings::CUfunction;
        // type CUfunc_st = super::bindings::CUfunc_st;
        // type CUctx_st = super::bindings::CUctx_st;
        // type Instr = super::bindings::Instr;

        // typedef enum cudaError_enum { ... } CUresult
        // type cudaError_enum;
        // type CUresult;

        // type nvbit_api_cuda_t;

        // type CUfunction;
        // type CUcontext;
        // type CUfunction = cxx::UniquePtr<CUfunc_st>;
        // type CUfunction = *mut CUfunc_st;
        // type CUcontext = *mut CUctx_st;
        // type CUresult;
        // type Instr;

        fn rust_nvbit_get_related_functions() -> usize;

        // long unsigned int (*)(CUctx_st*, CUfunc_st*)’} to ‘std::size_t (*)(const CUctx_st&, const CUfunc_st&)’ {aka ‘long unsigned int (*)(const CUctx_st&, const CUfunc_st&)
        // unsafe fn rust_new_nvbit_get_related_functions(
        // unsafe fn rust_new_nvbit_get_related_functions(
        //     // ctx: &CUctx_st,
        //     ctx: *mut CUctx_st,
        //     // ctx: TestCUcontext,
        //     // func: &CUfunc_st,
        //     func: *mut CUfunc_st,
        //     // func: TestCUfunction,
        //     // ) -> cxx::UniquePtr<cxx::CxxVector<CUfunction>>;
        // ) -> Vec<TestCUfunction>;

        // returns non mutable vec of instructions
        // const rust::Vec<uint8_t> &c_return_ref_rust_vec(const C &c);
        // fn unique_ptr_works() -> UniquePtr<u8>;

        fn rust_nvbit_get_instrs(// ctx: &CUctx_st,
            // ctx: CUcontext,
            // ctx: *mut CUctx_st,
            // ctx: TestCUcontext,
            // func: &CUfunc_st,
            // func: CUfunction,
            // func: *mut CUfunc_st,
            // func: TestCUfunction,
            // ) -> cxx::UniquePtr<cxx::CxxVector<CUfunction>>;
        // ) -> cxx::UniquePtr<cxx::CxxVector<usize>>;
        // // ) -> &cxx::String; // &cxx::CxxVector<u8>;
        ) -> UniquePtr<CxxVector<u8>>;
        // ) -> &cxx::CxxVector<u8>;
        // ) -> &cxx::CxxVector<*mut Instr>;
        // ) -> &cxx::CxxVector<*mut super::bindings::Instr>;
        // ) -> Vec<TestCUfunction>;

        // this returns an owned std::vector
        // ) -> cxx::CxxVector<u32>;
        // ) -> &cxx::CxxVector<SharedCUfunction>;
        // ) -> &cxx::CxxVector<cxx::UniquePtr<CUfunction>>;
        // ) -> Vec<CUfunction>;
        // either cxx::UniquePtr<cxx::CxxVector<T>> or &cxx::CxxVector<T>
        // CxxVector<T> does not support T being an opaque Rust type.
        // You should use a Vec<T> (C++ rust::Vec<T>) instead for collections
        // of opaque Rust types on the language boundary.

        //
        // ) -> cxx::UniquePtr<cxx::CxxVector<CUfunction>>;
        // ) -> cxx::UniquePtr<cxx::CxxVector<CUfunction>>;
        // ) -> &mut cxx::CxxVector<*mut CUfunc_st>;
        // ) -> usize; // &cxx::CxxVector<usize>;
        // ) -> &cxx::CxxVector<CUfunction>;
        // ) -> Vec<CUfunction>;
        // ) -> cxx::Vec<i32>;
        // ) -> cxx::CxxVector<*mut CUfunc_st>;
        // ) -> &cxx::CxxVector<i32>;
        // ) -> *mut cxx::CxxVector<*mut CUfunc_st>;
        // ) -> &cxx::CxxVector<CUfunction>;
        // &cxx::CxxVector<i32>;

        // &cxx::CxxVector<cxx::CxxString>;
        // fn new_blobstore_client() -> UniquePtr<BlobstoreClient>;
        // fn put(&self, parts: &mut MultiBuf) -> u64;
        // fn tag(&self, blobid: u64, tag: &str);
        // fn metadata(&self, blobid: u64) -> BlobMetadata;
    }
}

pub use ffi::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
