use nvbit_sys::{bindings, nvbit};
use std::marker::PhantomData;
use std::{ffi, fmt, pin::Pin};

/// An instruction operand.
#[repr(transparent)]
#[derive(Debug)]
pub struct Operand<'a> {
    ptr: *const nvbit::operand_t,
    instr: PhantomData<Instruction<'a>>,
}

impl<'a> Operand<'a> {
    #[inline]
    #[must_use]
    pub fn into_owned(&self) -> nvbit::operand_t {
        unsafe { *self.ptr }
    }

    #[inline]
    #[must_use]
    pub fn kind(&self) -> OperandKind {
        let kind = unsafe { *self.ptr }.type_;
        kind.into()
    }
}

/// Operand kind
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub enum OperandKind {
    ImmutableUint64 = 0,
    ImmutableDouble = 1,
    Register = 2,
    Predicate = 3,
    // todo: what does that mean?
    URegister = 4,
    // todo: what does that mean?
    UPredicate = 5,
    CBank = 6,
    MemRef = 7,
    Generic = 8,
}

impl From<bindings::InstrType_OperandType> for OperandKind {
    #[inline]
    #[must_use]
    fn from(kind: bindings::InstrType_OperandType) -> Self {
        use bindings::InstrType_OperandType as OPT;
        match kind {
            OPT::IMM_UINT64 => OperandKind::ImmutableUint64,
            OPT::IMM_DOUBLE => OperandKind::ImmutableDouble,
            OPT::REG => OperandKind::Register,
            OPT::PRED => OperandKind::Predicate,
            OPT::UREG => OperandKind::URegister,
            OPT::UPRED => OperandKind::UPredicate,
            OPT::CBANK => OperandKind::CBank,
            OPT::MREF => OperandKind::MemRef,
            OPT::GENERIC => OperandKind::Generic,
        }
    }
}

/// Identifier of GPU memory space.
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub enum MemorySpace {
    None,
    Local,
    Generic,
    Global,
    Shared,
    Constant,
    GlobalToShared,
    Surface,
    Texture,
}

impl From<bindings::InstrType_MemorySpace> for MemorySpace {
    #[inline]
    #[must_use]
    fn from(mem_space: bindings::InstrType_MemorySpace) -> Self {
        use bindings::InstrType_MemorySpace as MS;
        match mem_space {
            MS::NONE => MemorySpace::None,
            MS::LOCAL => MemorySpace::Local,
            MS::GENERIC => MemorySpace::Generic,
            MS::GLOBAL => MemorySpace::Global,
            MS::SHARED => MemorySpace::Shared,
            MS::CONSTANT => MemorySpace::Constant,
            MS::GLOBAL_TO_SHARED => MemorySpace::GlobalToShared,
            MS::SURFACE => MemorySpace::Surface,
            MS::TEXTURE => MemorySpace::Texture,
        }
    }
}

/// An instruction operand predicate.
#[derive(Debug, Clone)]
pub struct Predicate {
    /// predicate number
    pub num: i32,
    /// whether predicate is negated (i.e. @!P0).
    pub is_neg: bool,
    /// whether predicate is uniform predicate (e.g., @UP0).
    pub is_uniform: bool,
}

/// An instruction.
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash)]
pub struct Instruction<'a> {
    inner: *mut nvbit::Instr,
    func: PhantomData<super::Function<'a>>,
}

impl<'a> fmt::Debug for Instruction<'a> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // cannot access opcode etc. because they formally require a mutable reference
        // (which i doubt but okay)
        f.debug_struct("Instruction")
            // .field("opcode", &self.opcode())
            .finish()
    }
}

impl<'a> AsMut<nvbit::Instr> for Instruction<'a> {
    #[inline]
    #[must_use]
    fn as_mut(&mut self) -> &mut nvbit::Instr {
        unsafe { &mut *self.inner as &mut nvbit::Instr }
    }
}

impl<'a> AsRef<nvbit::Instr> for Instruction<'a> {
    #[inline]
    #[must_use]
    fn as_ref(&self) -> &nvbit::Instr {
        unsafe { &*self.inner as &nvbit::Instr }
    }
}

impl<'a> Instruction<'a> {
    #[inline]
    #[must_use]
    pub fn new(instr: *mut nvbit::Instr) -> Self {
        Self {
            inner: instr,
            func: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const nvbit::Instr {
        self.inner as *const _
    }

    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut nvbit::Instr {
        self.inner
    }

    #[inline]
    #[must_use]
    pub fn pin_mut(&mut self) -> Pin<&mut nvbit::Instr> {
        unsafe { Pin::new_unchecked(self.as_mut()) }
    }

    /// Returns string containing the SASS, i.e. `IMAD.WIDE R8, R8, R9`.
    #[inline]
    #[must_use]
    pub fn sass(&mut self) -> Option<&'a str> {
        let sass = self.pin_mut().getSass();
        if sass.is_null() {
            None
        } else {
            unsafe { ffi::CStr::from_ptr(sass) }.to_str().ok()
        }
    }

    /// Returns offset in bytes of this instruction within the function.
    #[inline]
    #[must_use]
    pub fn offset(&mut self) -> u32 {
        self.pin_mut().getOffset()
    }

    /// Returns index of the instruction within the function.
    #[inline]
    #[must_use]
    pub fn idx(&mut self) -> u32 {
        self.pin_mut().getIdx()
    }

    /// Returns the instruction predicate.
    #[inline]
    #[must_use]
    pub fn has_pred(&mut self) -> Option<Predicate> {
        if self.pin_mut().hasPred() {
            Some(Predicate {
                num: self.pin_mut().getPredNum(),
                is_neg: self.pin_mut().isPredNeg(),
                is_uniform: self.pin_mut().isPredUniform(),
            })
        } else {
            None
        }
    }

    /// full opcode of the instruction (i.e. IMAD.WIDE )
    #[inline]
    #[must_use]
    pub fn opcode(&mut self) -> Option<&'a str> {
        let opcode = self.pin_mut().getOpcode();
        if opcode.is_null() {
            None
        } else {
            unsafe { ffi::CStr::from_ptr(opcode) }.to_str().ok()
        }
    }

    /// short opcode of the instruction (i.e. IMAD.WIDE returns IMAD)
    #[inline]
    #[must_use]
    pub fn opcode_short(&mut self) -> Option<&'a str> {
        let opcode = self.pin_mut().getOpcodeShort();
        if opcode.is_null() {
            None
        } else {
            unsafe { ffi::CStr::from_ptr(opcode) }.to_str().ok()
        }
    }

    /// memory space type
    #[inline]
    #[must_use]
    pub fn memory_space(&mut self) -> MemorySpace {
        self.pin_mut().getMemorySpace().into()
    }

    /// true if this is a load instruction
    #[inline]
    #[must_use]
    pub fn is_load(&mut self) -> bool {
        self.pin_mut().isLoad()
    }

    /// true if this is a store instruction
    #[inline]
    #[must_use]
    pub fn is_store(&mut self) -> bool {
        self.pin_mut().isStore()
    }

    /// true if this is an extended instruction
    #[inline]
    #[must_use]
    pub fn is_extended(&mut self) -> bool {
        self.pin_mut().isExtended()
    }

    /// size of the instruction
    #[inline]
    #[must_use]
    pub fn size(&mut self) -> usize {
        self.pin_mut().getSize().unsigned_abs() as usize
    }

    /// number of operands
    #[inline]
    pub fn operands(&mut self) -> impl Iterator<Item = Operand<'a>> + '_ {
        let num_operands = self.num_operands();
        (0..num_operands).filter_map(|i| self.operand(i))
    }

    /// number of operands
    #[inline]
    #[must_use]
    pub fn num_operands(&mut self) -> usize {
        self.pin_mut().getNumOperands().unsigned_abs() as usize
    }

    /// operand at index
    #[inline]
    #[must_use]
    pub fn operand(&mut self, idx: usize) -> Option<Operand<'a>> {
        if idx >= self.num_operands() {
            // out of bounds
            return None;
        }
        let idx: i32 = if let Ok(idx) = idx.try_into() {
            idx
        } else {
            return None;
        };
        let ptr = self.pin_mut().getOperand(idx);
        if ptr.is_null() {
            None
        } else {
            Some(Operand {
                ptr,
                instr: PhantomData,
            })
        }
    }

    /// print fully decoded instruction
    #[inline]
    pub fn print_decoded(&mut self) {
        self.pin_mut().printDecoded();
    }

    /// print fully decoded instruction
    #[inline]
    pub fn print_with_prefix(&mut self, prefix: &str) {
        let prefix = ffi::CString::new(prefix).unwrap_or_default().as_ptr();
        unsafe { self.pin_mut().print(prefix) };
    }
}

/// Insertion point where the instrumentation for an instruction should be inserted
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub enum InsertionPoint {
    Before,
    After,
}

impl From<InsertionPoint> for bindings::ipoint_t {
    #[inline]
    #[must_use]
    fn from(point: InsertionPoint) -> bindings::ipoint_t {
        use bindings::ipoint_t as IP;
        match point {
            InsertionPoint::Before => IP::IPOINT_BEFORE,
            InsertionPoint::After => IP::IPOINT_AFTER,
        }
    }
}

impl From<bindings::ipoint_t> for InsertionPoint {
    #[inline]
    #[must_use]
    fn from(point: bindings::ipoint_t) -> Self {
        use bindings::ipoint_t as IP;
        match point {
            IP::IPOINT_BEFORE => Self::Before,
            IP::IPOINT_AFTER => Self::After,
        }
    }
}

impl<'a> Instruction<'a> {
    /// This function inserts a device function call named `"dev_func_name"`,
    /// before or after this instruction.
    ///
    /// It is important to remember that calls to device functions are
    /// identified by name (as opposed to function pointers) and that
    /// we declare the function as:
    ///
    /// ```cpp
    /// extern "C" __device__ __noinline__ kernel() {}
    /// ```
    ///
    /// to prevent the compiler from optimizing out this device function
    /// during compilation.
    ///
    /// Multiple device functions can be inserted before or after and the
    /// order in which they get executed is defined by the order in which
    /// they have been inserted.
    ///
    /// # Panics
    ///
    /// Panics if the device function name is not a valid C string,
    /// e.g. contains an internal 0 byte.
    #[inline]
    pub fn insert_call(&mut self, dev_func_name: impl AsRef<str>, point: InsertionPoint) {
        let func_name = ffi::CString::new(dev_func_name.as_ref()).unwrap();
        unsafe {
            bindings::nvbit_insert_call(self.as_ptr().cast(), func_name.as_ptr(), point.into());
        }
    }

    /// Add `int32_t` argument to last injected call,
    /// value of the (uniform) predicate for this instruction.
    #[inline]
    pub fn add_call_arg_guard_pred_val(&mut self) {
        unsafe { bindings::nvbit_add_call_arg_guard_pred_val(self.as_ptr().cast(), false) };
    }

    /// Add `int32_t` argument to last injected call,
    /// value of the designated predicate for this instruction.
    #[inline]
    pub fn add_call_arg_pred_val_at(&mut self, pred_num: i32) {
        unsafe { bindings::nvbit_add_call_arg_pred_val_at(self.as_ptr().cast(), pred_num, false) };
    }

    /// Add `int32_t` argument to last injected call,
    /// value of the designated uniform predicate for this instruction.
    #[inline]
    pub fn add_call_arg_upred_val_at(&mut self, upred_num: i32) {
        unsafe {
            bindings::nvbit_add_call_arg_upred_val_at(self.as_ptr().cast(), upred_num, false);
        };
    }

    /// Add `int32_t` argument to last injected call,
    /// value of the entire predicate register for this thread.
    #[inline]
    pub fn add_call_arg_pred_reg(&mut self) {
        unsafe {
            bindings::nvbit_add_call_arg_pred_reg(self.as_ptr().cast(), false);
        };
    }

    /// Add `int32_t` argument to last injected call,
    /// value of the entire uniform predicate register for this thread.
    #[inline]
    pub fn add_call_arg_upred_reg(&mut self) {
        unsafe {
            bindings::nvbit_add_call_arg_upred_reg(self.as_ptr().cast(), false);
        };
    }

    /// Add `uint32_t` argument to last injected call, constant 32-bit value.
    #[inline]
    pub fn add_call_arg_const_val32(&mut self, val: u32) {
        unsafe { bindings::nvbit_add_call_arg_const_val32(self.as_ptr().cast(), val, false) };
    }

    /// Add `uint64_t` argument to last injected call,
    /// constant 64-bit value.
    #[inline]
    pub fn add_call_arg_const_val64(&mut self, val: u64) {
        unsafe { bindings::nvbit_add_call_arg_const_val64(self.as_ptr().cast(), val, false) };
    }

    /// Add `uint32_t` argument to last injected call,
    /// content of the register `reg_num`.
    #[inline]
    pub fn add_call_arg_reg_val(&mut self, reg_num: i32) {
        unsafe { bindings::nvbit_add_call_arg_reg_val(self.as_ptr().cast(), reg_num, false) };
    }

    /// Add `uint32_t` argument to last injected call,
    /// content of theuniform register `reg_num`.
    #[inline]
    pub fn add_call_arg_ureg_val(&mut self, reg_num: i32) {
        unsafe { bindings::nvbit_add_call_arg_ureg_val(self.as_ptr().cast(), reg_num, false) };
    }

    /// Add `uint32_t` argument to last injected call,
    /// 32-bit at launch value at `offset`,
    /// set at launch time with `nvbit_set_at_launch`.
    #[inline]
    pub fn add_call_arg_launch_val32(&mut self, offset: i32) {
        unsafe { bindings::nvbit_add_call_arg_launch_val32(self.as_ptr().cast(), offset, false) };
    }

    /// Add `uint64_t` argument to last injected call,
    /// 64-bit at launch value at offset "offset",
    /// set at launch time with `nvbit_set_at_launch`.
    #[inline]
    pub fn add_call_arg_launch_val64(&mut self, offset: i32) {
        unsafe { bindings::nvbit_add_call_arg_launch_val64(self.as_ptr().cast(), offset, false) };
    }

    /// Add `uint32_t` argument to last injected call,
    /// constant bank value at `c[bankid][bankoffset]`.
    #[inline]
    pub fn add_call_arg_cbank_val(&mut self, bank_id: i32, bank_offset: i32) {
        unsafe {
            bindings::nvbit_add_call_arg_cbank_val(
                self.as_ptr().cast(),
                bank_id,
                bank_offset,
                false,
            );
        };
    }

    /// The 64-bit memory reference address accessed by this instruction.
    ///
    /// Typically memory instructions have only 1 MREF so in general id = 0
    #[inline]
    pub fn add_call_arg_mref_addr64(&mut self, id: i32) {
        unsafe { bindings::nvbit_add_call_arg_mref_addr64(self.as_ptr().cast(), id, false) };
    }

    /// Remove the original instruction
    #[inline]
    pub fn remove_orig(&mut self) {
        unsafe { bindings::nvbit_remove_orig(self.as_ptr().cast()) };
    }
}
