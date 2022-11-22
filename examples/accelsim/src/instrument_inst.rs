#[allow(non_snake_case)]
#[derive(Debug, Default, Clone)]
pub struct Args {
    pub opcode_id: std::ffi::c_int,
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

impl Args {
    pub fn instrument<'a>(&self, instr: &mut nvbit_rs::Instruction<'a>) {
        instr.add_call_arg_guard_pred_val();
        instr.add_call_arg_const_val32(self.opcode_id.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.vpc);
        instr.add_call_arg_const_val32(self.is_mem.try_into().unwrap_or_default());
        instr.add_call_arg_mref_addr64(self.addr.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.width.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.desReg.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.srcReg1.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.srcReg2.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.srcReg3.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.srcReg4.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.srcReg5.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.srcNum.try_into().unwrap_or_default());
        instr.add_call_arg_const_val64(self.pchannel_dev);
        instr.add_call_arg_const_val64(self.ptotal_dynamic_instr_counter);
        instr.add_call_arg_const_val64(self.preported_dynamic_instr_counter);
        instr.add_call_arg_const_val64(self.pstop_report);
    }
}
