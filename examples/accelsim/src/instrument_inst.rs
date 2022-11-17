use nvbit_sys::bindings;
use nvbit_sys::nvbit::Instr;

#[derive(Debug, Default, Clone)]
pub struct InstrumentInstArgs {
    pub opcode_id: libc::c_int,
    pub vpc: u32,
    pub is_mem: bool,
    pub addr: u64,
    pub width: i32,
    pub desReg: i32,
    pub srcReg1: i32,
    pub srcReg2: i32,
    pub srcReg3: i32,
    pub srcReg4: i32,
    pub srcReg5: i32,
    pub srcNum: i32,
    pub pchannel_dev: u64,
    pub ptotal_dynamic_instr_counter: u64,
    pub preported_dynamic_instr_counter: u64,
    pub pstop_report: u64,
}

impl InstrumentInstArgs {
    pub unsafe fn instrument(&self, instr: *const Instr) {
        bindings::nvbit_add_call_arg_guard_pred_val(instr, false);
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.opcode_id.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(instr, self.vpc, false);
        bindings::nvbit_add_call_arg_const_val32(instr, self.is_mem as u32, false);
        bindings::nvbit_add_call_arg_mref_addr64(
            instr,
            self.addr.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.width.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.desReg.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcReg1.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcReg2.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcReg3.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcReg4.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcReg5.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcNum.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val64(instr, self.pchannel_dev, false);
        bindings::nvbit_add_call_arg_const_val64(instr, self.ptotal_dynamic_instr_counter, false);
        bindings::nvbit_add_call_arg_const_val64(
            instr,
            self.preported_dynamic_instr_counter,
            false,
        );
        bindings::nvbit_add_call_arg_const_val64(instr, self.pstop_report, false);
    }
}
