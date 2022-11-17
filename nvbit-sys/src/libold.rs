#![allow(warnings, dead_code)]
use std::fmt;
use std::pin::Pin;

#[allow(warnings, dead_code)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings/nvbit.rs"));

    pub mod channel {
        include!(concat!(env!("OUT_DIR"), "/bindings/channel.rs"));
    }

    // unsafe impl cxx::ExternType for CUfunction {
    //     type Id = cxx::type_id!("CUfunction");
    //     // type Kind = cxx::kind::Trivial;
    //     type Kind = cxx::kind::Opaque;
    // }

    // unsafe impl cxx::ExternType for CUcontext {
    //     type Id = cxx::type_id!("CUcontext");
    //     // type Kind = cxx::kind::Trivial;
    //     type Kind = cxx::kind::Opaque;
    // }
}

// unsafe impl cxx::ExternType for bindings::CUfunction {
//     type Id = cxx::type_id!("CUfunction");
//     type Kind = cxx::kind::Opaque;
// }

// pub use bindings::*;

// namespace = "org::blobstore"

unsafe impl cxx::ExternType for bindings::CUfunc_st {
    type Id = cxx::type_id!("CUfunc_st");
    type Kind = cxx::kind::Trivial;
    // type Kind = cxx::kind::Opaque;
}

unsafe impl cxx::ExternType for bindings::CUctx_st {
    type Id = cxx::type_id!("CUctx_st");
    type Kind = cxx::kind::Trivial;
    // type Kind = cxx::kind::Opaque;
}

unsafe impl cxx::ExternType for bindings::Instr {
    type Id = cxx::type_id!("Instr");
    // type Kind = cxx::kind::Trivial;
    type Kind = cxx::kind::Opaque;
}

// unsafe impl Send for bindings::CUcontext {}
// unsafe impl Send for bindings::CUfunction {}

// unsafe impl Send for *const bindings::CUctx_st {}
// unsafe impl Send for *const bindings::CUfunc_st {}

// unsafe impl Send for *mut bindings::CUctx_st {}
// unsafe impl Send for *mut bindings::CUfunc_st {}

// unsafe impl Send for ffi::TestCUfunction {}
// unsafe impl Sync for ffi::TestCUfunction {}

#[cxx::bridge]
mod ffi {
    // struct BlobMetadata {
    //     size: usize,
    //     tags: Vec<String>,
    // }
    // type CUfunc_st; //  = super::bindings::CUfunction;
    // type CUfunction = super::bindings::CUfunc_st;

    #[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd)]
    // #[derive(Clone, Copy)]
    // #[rust_name = "CUfunctionShim"]
    struct CUfunctionShim {
        ptr: *mut CUfunc_st,
    }
    // {
    //     // test: usize,
    //     // inner: cxx::UniquePtr<CUfunc_st>,
    //     ptr: *mut CUfunc_st,
    //     // inner: *mut super::bindings::CUfunc_st,
    //     // inner: *mut CUfunc_st,
    //     // inner: *mut super::bindings::CUfunc_st,
    //     // vec: Vec<usize>,
    // }

    #[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd)]
    // #[derive(Copy)]
    // #[rust_name = "CUcontextShim"]
    struct CUcontextShim {
        ptr: *mut CUctx_st,
    }

    // #[derive(Clone, Hash, Eq, PartialEq, PartialOrd)]
    // #[derive(Copy)]
    // #[rust_name = "CUcontextShim"]
    // struct InstrShim<'a> {
    struct InstrShim {
        ptr: *mut Instr,
        // inner: Pin<&mut Instr>, // ptr: &mut Instr,
        // inner: Pin<&'a mut Instr>, // ptr: &mut Instr,
    }

    // #[derive(Clone, Copy, Hash, Eq, PartialEq, PartialOrd)]
    // struct TestCUfunction {
    //     test: usize,
    //     // inner: cxx::UniquePtr<CUfunc_st>,
    //     // inner: *mut CUfunc_st,
    //     // inner: *mut super::bindings::CUfunc_st,
    //     // inner: *mut CUfunc_st,
    //     // inner: *mut super::bindings::CUfunc_st,
    //     // vec: Vec<usize>,
    // }

    // is CUfunction sync? is it safe to pass it around?
    // its opaque so we cannot write to it anyways?
    // doing so would only mess up real bad

    // #[derive(PartialEq, PartialOrd)]
    // struct TestCUcontext {
    //     // test: usize,
    //     // inner: cxx::UniquePtr<CUfunc_st>,
    //     // inner: *mut CUctx_st,
    //     // inner: *mut super::bindings::CUctx_st,
    //     // vec: Vec<usize>,
    // }

    // Rust types and signatures exposed to C++.
    // we do not want to expose anything to C++
    extern "Rust" {
        // type CUfunction;
        // type MultiBuf;

        // fn next_chunk(buf: &mut MultiBuf) -> &[u8];
    }

    unsafe extern "C++" {
        // include!("nvbit.h");
        // include!("nvbit/nvbit.cc");
        // include!("nvbit-sys/nvbit/nvbit.h");
        include!("nvbit-sys/nvbit/nvbit.h");
        include!("nvbit-sys/nvbit/channel.h");
        // include!("nvbit-sys/nvbit_release/core/utils/channel.hpp");
        // include!("nvbit-sys/nvbit_release/core/utils/channel.hpp");
        // include!("nvbit-sys/nvbit/nvbit.h");
        // include!("nvbit_sys/nvbit/nvbit.cc");
        // include!("nvbit_release/core/nvbit.h");

        // type CUfunction;
        // type CUcontext;
        // type CUfunction = super::bindings::CUfunction;
        // type CUcontext = super::bindings::CUcontext;

        // type CUfunc_st;
        // type CUctx_st;
        type CUfunc_st = super::bindings::CUfunc_st;
        type CUctx_st = super::bindings::CUctx_st;

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
        type Instr = super::bindings::Instr;

        type ChannelDev; //  = super::bindings::Instr;
        type ChannelHost; //  = super::bindings::Instr;

        fn new_dev_channel() -> UniquePtr<ChannelDev>;
        fn new_host_channel() -> UniquePtr<ChannelHost>;

        // unsafe fn init(
        //     self: Pin<&mut ChannelDev>,
        //     id: i32,
        //     h_doorbell: *mut i32,
        //     buff_size: i32,
        // );

        // unsafe fn init(
        //     self: Pin<&mut ChannelHost>,
        //     id: i32,
        //     buff_size: i32,
        //     channel_dev: *mut ChannelDev,
        //     thread_fun: fn() -> libc::c_void,
        //     args: *mut libc::c_void,
        // );

        // const char* getSass();

        // /// returns the "string" containing the SASS, i.e. IMAD.WIDE R8, R8, R9
        // fn getSass(self: &Instr) -> *const c_char;

        // /// returns offset in bytes of this instruction within the function
        // fn getOffset(self: Pin<&mut Instr>) -> u32;
        // fn getOffset(self: &Instr) -> u32;

        // /// returns the id of the instruction within the function
        // fn getIdx(self: &Instr) -> u32;

        // /// returns true if instruction used predicate
        // fn hasPred(self: &Instr) -> bool;

        // /// returns predicate number, only valid if hasPred() == true
        // fn getPredNum(self: &Instr) -> i32; // should be c_int

        // /// returns true if predicate is negated (i.e. @!P0).
        // /// only valid if hasPred() == true
        // fn isPredNeg(self: &Instr) -> bool;

        // /// returns true if predicate is uniform predicate (e.g., @UP0).
        // /// only valid if hasPred() == true
        // fn isPredUniform(self: &Instr) -> bool;

        // /// returns full opcode of the instruction (i.e. IMAD.WIDE )
        // fn getOpcode(self: Pin<&mut Instr>) -> *const c_char;
        // fn getOpcode(self: &Instr) -> *const c_char;

        // /// returns short opcode of the instruction (i.e. IMAD.WIDE returns IMAD)
        // fn getOpcodeShort(self: &Instr) -> *const c_char;

        // /// returns MemorySpace type (was getMemOpType)
        // // InstrType::MemorySpace getMemorySpace();

        // /// returns true if this is a load instruction
        // fn isLoad(self: &Instr) -> bool;

        // /// returns true if this is a store instruction
        // fn isStore(self: &Instr) -> bool;

        // /// returns true if this is an extended instruction
        // fn isExtended(self: &Instr) -> bool;

        // /// returns the size of the instruction
        // fn getSize(self: &Instr) -> i32;

        // /// get number of operands
        // fn getNumOperands(self: &Instr) -> i32;

        // /// get specific operand
        // // const InstrType::operand_t* getOperand(int num_operand);

        // /// print fully decoded instruction
        // fn printDecoded(self: Pin<&mut Instr>);
        // fn printDecoded(self: &Instr);

        // /// prints one line instruction with idx, offset, sass
        // // fn print(self: &Instr, prefix: *const c_char);
        // fn print(self: &Instr, prefix: &c_char);

        // InstrType::MemorySpace

        // fn rust_nvbit_get_related_functions() -> usize;

        // fn nvbit_enable_instrumented(
        //     ctx: CUcontext,
        //     func: CUfunction,
        //     flag: bool,
        //     apply_to_related: bool,
        // );

        // long unsigned int (*)(CUctx_st*, CUfunc_st*)’} to ‘std::size_t (*)(const CUctx_st&, const CUfunc_st&)’ {aka ‘long unsigned int (*)(const CUctx_st&, const CUfunc_st&)
        // unsafe fn rust_new_nvbit_get_related_functions(
        // fn rust_new_nvbit_get_related_functions(
        unsafe fn rust_nvbit_get_related_functions(
            ctx: *mut CUctx_st,
            // ctx: &CUctx_st,
            // ctx: *mut CUctx_st,
            // ctx: CUcontextShim,
            // ctx: CUcontext,
            // ctx: TestCUcontext,
            // func: &CUfunc_st,
            // func: &mut CUfunc_st,
            func: *mut CUfunc_st,
            // func: CUfunctionShim,
            // func: CUfunctionShim,
            // func: CUfunction,
            // func: TestCUfunction,
            // ) -> cxx::UniquePtr<cxx::CxxVector<CUfunction>>;
            // ) -> &CxxVector<CUfunc_st>;
            // ) -> UniquePtr<CxxVector<CUfunc_st>>;
        ) -> UniquePtr<CxxVector<CUfunctionShim>>;
        // ) -> UniquePtr<CxxVector<CUfunc_st>>;
        // ) -> UniquePtr<CxxVector<UniquePtr<CUfunc_st>>>;
        // ) -> UniquePtr<CxxVector<CUfunction>>;
        // ) -> UniquePtr<CxxVector<CUfunctionShim>>;
        // ) -> Vec<CUfunctionShim>;
        // ) -> Vec<TestCUfunction>;

        // returns non mutable vec of instructions
        // const rust::Vec<uint8_t> &c_return_ref_rust_vec(const C &c);
        unsafe fn rust_nvbit_get_instrs(
            // unsafe fn nvbit_get_instrs(
            // ctx: &mut CUctx_st,
            // ctx: CUcontextShim,
            // ctx: CUcontext,
            ctx: *mut CUctx_st,
            // ctx: TestCUcontext,
            // func: &mut CUfunc_st,
            // func: CUfunctionShim,
            // func: CUfunction,
            func: *mut CUfunc_st,
            // func: TestCUfunction,
            // ) -> &CxxVector<u8>;
            // ) -> UniquePtr<CxxVector<SharedPointer<Instr>>>;
        ) -> UniquePtr<CxxVector<InstrShim>>;
        // ) -> &CxxVector<&Instr>;
        // ) -> CxxVector<Instr>;
        // ) -> &CxxVector<*mut Instr>;
        // ) -> cxx::UniquePtr<cxx::CxxVector<CUfunction>>;
        // ) -> cxx::UniquePtr<cxx::CxxVector<usize>>;
        // ) -> &cxx::String; // &cxx::CxxVector<u8>;
        // ) -> cxx::UniquePtr<cxx::CxxVector<u8>>;
        // ) -> &cxx::CxxVector<u8>;
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
// pub methods are fine here
// we still cannot construct a fake ptr ourselves
// because bindgen made it opaque right?

impl ffi::InstrShim {
    pub unsafe fn wrap(ptr: *mut ffi::Instr) -> Self {
        Self { ptr }
    }

    pub unsafe fn as_ref(&self) -> &ffi::Instr {
        &*self.ptr
    }

    pub unsafe fn as_mut_ref(&self) -> &mut ffi::Instr {
        &mut *self.ptr
    }

    pub unsafe fn as_ptr(&self) -> *const ffi::Instr {
        self.ptr as *const _
    }

    pub unsafe fn as_mut_ptr(&self) -> *mut ffi::Instr {
        self.ptr
    }
}

// impl<'a> fmt::Debug for ffi::InstrShim<'a> {
impl fmt::Debug for ffi::InstrShim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // let ptr = unsafe { &mut *self.as_mut_ptr() };
        // let instr = unsafe { Pin::new_unchecked(ptr) };
        // let opcode = instr.getOpcode();
        let opcode = unsafe { self.as_mut_ref().getOpcode() };
        let opcode = unsafe { std::ffi::CStr::from_ptr(opcode).to_string_lossy() };
        // let instr = Box::pin(ptr);
        // let instr = &self.inner;
        f.debug_struct("Instr").field("opcode", &opcode).finish()
    }
}

impl ffi::CUfunctionShim {
    pub unsafe fn wrap(ptr: *mut bindings::CUfunc_st) -> Self {
        Self { ptr }
    }

    pub unsafe fn as_ptr(&self) -> *const bindings::CUfunc_st {
        self.ptr as *const _
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut bindings::CUfunc_st {
        self.ptr
    }
}

impl ffi::CUcontextShim {
    pub unsafe fn wrap(ptr: *mut bindings::CUctx_st) -> Self {
        Self { ptr }
    }

    pub unsafe fn as_ptr(&self) -> *const bindings::CUctx_st {
        self.ptr as *const _
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut bindings::CUctx_st {
        self.ptr
    }
}

pub use ffi::*;
// pub use ffi::CUcontextShim as CUcontext;
// pub use ffi::CUfunctionShim as CUfunction;

extern "C" {
    // fn link_me();
    // fn init_nvbit();
    // fn instrument_inst();
}

// pub fn init() {
//     // unsafe { link_me() }
//     // unsafe { init_nvbit() }
// }

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn has_nvbit() {
    //     unsafe { init_nvbit() }
    // }
}
