use nvbit_sys::{bindings, nvbit};
use std::marker::PhantomData;
use std::{ffi, fmt, pin::Pin};

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum RegisterModifier {
    None = 0,
    X1 = 1,
    X4 = 2,
    X8 = 3,
    X16 = 4,
    U32 = 5,
    U64 = 6,
}

impl From<bindings::InstrType_RegModifierType> for RegisterModifier {
    #[inline]
    #[must_use]
    fn from(mem_space: bindings::InstrType_RegModifierType) -> Self {
        use bindings::InstrType_RegModifierType as RMT;
        match mem_space {
            RMT::NO_MOD => RegisterModifier::None,
            RMT::X1 => RegisterModifier::X1,
            RMT::X4 => RegisterModifier::X4,
            RMT::X8 => RegisterModifier::X8,
            RMT::X16 => RegisterModifier::X16,
            RMT::U32 => RegisterModifier::U32,
            RMT::U64 => RegisterModifier::U64,
        }
    }
}

/// An instruction operand.
#[derive(PartialEq, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum OperandKind {
    ImmutableUint64 {
        value: u64,
    },
    ImmutableDouble {
        value: f64,
    },
    Register {
        num: i32,
        prop: String,
    },
    Predicate {
        num: i32,
    },
    // URegister {},
    // UPredicate {},
    CBank {
        id: i32,
        has_imm_offset: bool,
        imm_offset: i32,
        has_reg_offset: bool,
        reg_offset: i32,
    },
    MemRef {
        has_ra: bool,
        ra_num: i32,
        ra_mod: RegisterModifier,
        has_ur: bool,
        ur_num: i32,
        ur_mod: RegisterModifier,
        has_imm: bool,
        imm: i32,
    },
    Generic {
        array: String,
    },
}

/// An instruction operand.
#[repr(transparent)]
#[derive(Debug, serde::Serialize)]
pub struct Operand<'a> {
    #[serde(serialize_with = "crate::to_raw_ptr")]
    inner: *const nvbit::operand_t,
    instr: PhantomData<Instruction<'a>>,
}

impl<'a> Operand<'a> {
    #[inline]
    #[must_use]
    pub fn kind(&self) -> OperandKind {
        use bindings::InstrType_OperandType as OPT;
        let operand = unsafe { *self.inner };
        match operand.type_ {
            OPT::IMM_UINT64 => OperandKind::ImmutableUint64 {
                value: unsafe { operand.u.imm_uint64.value },
            },
            OPT::IMM_DOUBLE => OperandKind::ImmutableDouble {
                value: unsafe { operand.u.imm_double.value },
            },
            OPT::UREG | OPT::REG => {
                let reg = unsafe { operand.u.reg };
                let prop: [u8; 256] = unsafe { std::mem::transmute(reg.prop) };
                let prop = String::from_utf8_lossy(&prop).to_string();
                OperandKind::Register { num: reg.num, prop }
            }
            OPT::UPRED | OPT::PRED => OperandKind::Predicate {
                num: unsafe { operand.u.pred.num },
            },
            OPT::CBANK => {
                let cbank = unsafe { operand.u.cbank };
                OperandKind::CBank {
                    id: cbank.id,
                    has_imm_offset: cbank.has_imm_offset,
                    imm_offset: cbank.imm_offset,
                    has_reg_offset: cbank.has_reg_offset,
                    reg_offset: cbank.reg_offset,
                }
            }
            OPT::MREF => {
                let mref = unsafe { operand.u.mref };
                OperandKind::MemRef {
                    has_ra: mref.has_ra,
                    ra_num: mref.ra_num,
                    ra_mod: mref.ra_mod.into(),
                    has_ur: mref.has_ur,
                    ur_num: mref.ur_num,
                    ur_mod: mref.ur_mod.into(),
                    has_imm: mref.has_imm,
                    imm: mref.imm,
                }
            }
            OPT::GENERIC => {
                let array = unsafe { operand.u.generic.array };
                let array: [u8; 256] = unsafe { std::mem::transmute(array) };
                let array = String::from_utf8_lossy(&array).to_string();
                OperandKind::Generic { array }
            }
        }
    }

    #[inline]
    #[must_use]
    pub fn name(&self) -> String {
        let name = unsafe { *self.inner }.str_;
        let name: [u8; 256] = unsafe { std::mem::transmute(name) };
        String::from_utf8_lossy(&name).to_string()
    }

    #[inline]
    #[must_use]
    pub fn is_neg(&self) -> bool {
        unsafe { *self.inner }.is_neg
    }

    #[inline]
    #[must_use]
    pub fn is_not(&self) -> bool {
        unsafe { *self.inner }.is_not
    }

    #[inline]
    #[must_use]
    pub fn is_abs(&self) -> bool {
        unsafe { *self.inner }.is_abs
    }

    #[inline]
    #[must_use]
    pub fn nbytes(&self) -> u32 {
        unsafe { *self.inner }.nbytes.unsigned_abs()
    }
}

/// Identifier of GPU memory space.
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum MemorySpace {
    None = 0,
    Local = 1,
    Generic = 2,
    Global = 3,
    Shared = 4,
    Constant = 5,
    GlobalToShared = 6,
    Surface = 7,
    Texture = 8,
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
#[derive(Default, PartialEq, Eq, Hash, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Predicate {
    /// predicate number
    pub num: std::ffi::c_int,
    /// whether predicate is negated (i.e. @!P0).
    pub is_neg: bool,
    /// whether predicate is uniform predicate (e.g., @UP0).
    pub is_uniform: bool,
}

/// An instruction.
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, serde::Serialize)]
pub struct Instruction<'a> {
    #[serde(serialize_with = "crate::to_raw_ptr")]
    inner: *mut nvbit::Instr,
    func: PhantomData<super::Function<'a>>,
}

impl<'a> fmt::Display for Instruction<'a> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // accesses formally require a mutable reference, but
        // this should be okay..
        let opcode = unsafe { nvbit_sys::Instr_getOpcode(self.inner.cast()) };
        let opcode = unsafe { ffi::CStr::from_ptr(opcode) }
            .to_str()
            .unwrap_or_default();
        f.debug_tuple("Instruction").field(&opcode).finish()
    }
}

impl<'a> fmt::Debug for Instruction<'a> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // accesses formally require a mutable reference, but
        // this should be okay..
        let opcode = unsafe { nvbit_sys::Instr_getOpcode(self.inner.cast()) };
        let opcode = unsafe { ffi::CStr::from_ptr(opcode) }
            .to_str()
            .unwrap_or_default();

        f.debug_struct("Instruction")
            .field("opcode", &opcode)
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
    pub fn predicate(&mut self) -> Option<Predicate> {
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
        let operand = self.pin_mut().getOperand(idx);
        if operand.is_null() {
            None
        } else {
            Some(Operand {
                inner: operand,
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
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
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
