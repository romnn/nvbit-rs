use super::bindings;

unsafe impl cxx::ExternType for bindings::CUfunc_st {
    type Id = cxx::type_id!("CUfunc_st");
    type Kind = cxx::kind::Trivial;
}

unsafe impl cxx::ExternType for bindings::CUctx_st {
    type Id = cxx::type_id!("CUctx_st");
    type Kind = cxx::kind::Trivial;
}

unsafe impl cxx::ExternType for bindings::InstrType_operand_t {
    type Id = cxx::type_id!("InstrType::operand_t");
    type Kind = cxx::kind::Trivial;
}

unsafe impl cxx::ExternType for bindings::InstrType_MemorySpace {
    type Id = cxx::type_id!("InstrType::MemorySpace");
    type Kind = cxx::kind::Trivial;
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
        include!("nvbit-sys/nvbit/nvbit_bridge.h");

        type CUfunc_st = super::bindings::CUfunc_st;
        type CUctx_st = super::bindings::CUctx_st;

        unsafe fn rust_nvbit_get_related_functions(
            ctx: *mut CUctx_st,
            func: *mut CUfunc_st,
        ) -> UniquePtr<CxxVector<CUfunctionShim>>;

        type Instr;

        unsafe fn rust_nvbit_get_instrs(
            ctx: *mut CUctx_st,
            func: *mut CUfunc_st,
        ) -> UniquePtr<CxxVector<InstrShim>>;

        #[namespace = "InstrType"]
        type MemorySpace = super::bindings::InstrType_MemorySpace;
        #[namespace = "InstrType"]
        type operand_t = super::bindings::InstrType_operand_t;

        /// returns the "string" containing the SASS, i.e. IMAD.WIDE R8, R8, R9
        fn getSass(self: Pin<&mut Instr>) -> *const c_char;

        /// returns offset in bytes of this instruction within the function
        fn getOffset(self: Pin<&mut Instr>) -> u32;

        /// returns the id of the instruction within the function
        fn getIdx(self: Pin<&mut Instr>) -> u32;

        /// returns true if instruction used predicate
        fn hasPred(self: Pin<&mut Instr>) -> bool;

        /// returns predicate number, only valid if hasPred() == true
        fn getPredNum(self: Pin<&mut Instr>) -> i32; // should be c_int

        /// returns true if predicate is negated (i.e. @!P0).
        /// only valid if hasPred() == true
        fn isPredNeg(self: Pin<&mut Instr>) -> bool;

        /// returns true if predicate is uniform predicate (e.g., @UP0).
        /// only valid if hasPred() == true
        fn isPredUniform(self: Pin<&mut Instr>) -> bool;

        /// returns full opcode of the instruction (i.e. IMAD.WIDE )
        fn getOpcode(self: Pin<&mut Instr>) -> *const c_char;

        /// returns short opcode of the instruction (i.e. IMAD.WIDE returns IMAD)
        fn getOpcodeShort(self: Pin<&mut Instr>) -> *const c_char;

        /// returns MemorySpace type (was getMemOpType)
        fn getMemorySpace(self: Pin<&mut Instr>) -> MemorySpace;

        /// returns true if this is a load instruction
        fn isLoad(self: Pin<&mut Instr>) -> bool;

        /// returns true if this is a store instruction
        fn isStore(self: Pin<&mut Instr>) -> bool;

        /// returns true if this is an extended instruction
        fn isExtended(self: Pin<&mut Instr>) -> bool;

        /// returns the size of the instruction
        fn getSize(self: Pin<&mut Instr>) -> i32;

        /// get number of operands
        fn getNumOperands(self: Pin<&mut Instr>) -> i32;

        /// get specific operand
        fn getOperand(self: Pin<&mut Instr>, num_operand: i32) -> *const operand_t;

        /// print fully decoded instruction
        fn printDecoded(self: Pin<&mut Instr>);

        /// prints one line instruction with idx, offset, sass
        unsafe fn print(self: Pin<&mut Instr>, prefix: *const c_char);
    }
}

impl ffi::InstrShim {
    pub fn wrap(ptr: *mut ffi::Instr) -> Self {
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *const ffi::Instr {
        self.ptr as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut ffi::Instr {
        self.ptr
    }
}

impl ffi::CUfunctionShim {
    pub fn wrap(ptr: *mut bindings::CUfunc_st) -> Self {
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *const bindings::CUfunc_st {
        self.ptr as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut bindings::CUfunc_st {
        self.ptr
    }
}

impl ffi::CUcontextShim {
    pub fn wrap(ptr: *mut bindings::CUctx_st) -> Self {
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *const bindings::CUctx_st {
        self.ptr as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut bindings::CUctx_st {
        self.ptr
    }
}

pub use ffi::*;
