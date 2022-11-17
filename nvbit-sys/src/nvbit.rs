use super::bindings;

unsafe impl cxx::ExternType for bindings::CUfunc_st {
    type Id = cxx::type_id!("CUfunc_st");
    type Kind = cxx::kind::Trivial;
}

unsafe impl cxx::ExternType for bindings::CUctx_st {
    type Id = cxx::type_id!("CUctx_st");
    type Kind = cxx::kind::Trivial;
}

unsafe impl cxx::ExternType for bindings::Instr {
    type Id = cxx::type_id!("Instr");
    type Kind = cxx::kind::Opaque;
}

#[cxx::bridge]
mod ffi {
    struct CUfunctionShim {
        ptr: *mut CUfunc_st,
    }

    struct CUcontextShim {
        ptr: *mut CUctx_st,
    }

    struct InstrShim {
        ptr: *mut Instr,
    }

    extern "Rust" {}

    unsafe extern "C++" {
        include!("nvbit-sys/nvbit/nvbit.h");

        type CUfunc_st = super::bindings::CUfunc_st;
        type CUctx_st = super::bindings::CUctx_st;

        type Instr = super::bindings::Instr;

        unsafe fn rust_nvbit_get_related_functions(
            ctx: *mut CUctx_st,
            func: *mut CUfunc_st,
        ) -> UniquePtr<CxxVector<CUfunctionShim>>;

        unsafe fn rust_nvbit_get_instrs(
            ctx: *mut CUctx_st,
            func: *mut CUfunc_st,
        ) -> UniquePtr<CxxVector<InstrShim>>;
    }
}

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

impl ffi::CUfunctionShim {
    pub unsafe fn wrap(ptr: *mut bindings::CUfunc_st) -> Self {
        Self { ptr }
    }

    pub unsafe fn as_ptr(&self) -> *const bindings::CUfunc_st {
        self.ptr as *const _
    }

    pub unsafe fn as_mut_ptr(&self) -> *mut bindings::CUfunc_st {
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

    pub unsafe fn as_mut_ptr(&self) -> *mut bindings::CUctx_st {
        self.ptr
    }
}

pub use ffi::*;
