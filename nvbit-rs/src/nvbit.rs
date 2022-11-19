use super::{Context, Function, InsertionPoint, Instruction, CFG};
use nvbit_sys::{bindings, nvbit};
use std::ffi;

/// Return nvbit version
pub fn version() -> &'static str {
    std::str::from_utf8(unsafe { bindings::NVBIT_VERSION }).unwrap()
}

/// Get related functions for given CUfunction
pub fn get_related_functions<'c, 'f>(
    ctx: &mut Context<'c>,
    func: &mut Function<'f>,
) -> Vec<Function<'f>> {
    let c = ctx.as_mut_ptr();
    let f = func.as_mut_ptr();
    let mut related_functions: cxx::UniquePtr<cxx::Vector<nvbit::CUfunctionShim>> =
        unsafe { nvbit::rust_nvbit_get_related_functions(c, f) };

    related_functions
        .pin_mut()
        .iter_mut()
        .map(|mut f| Function::new(f.as_mut_ptr()))
        .collect()
}

/// Get instructions composing the CUfunction
pub fn get_instrs<'c, 'f>(ctx: &mut Context<'c>, func: &mut Function<'f>) -> Vec<Instruction<'f>> {
    let c = ctx.as_mut_ptr();
    let f = func.as_mut_ptr();
    let mut instructions: cxx::UniquePtr<cxx::Vector<nvbit::InstrShim>> =
        unsafe { nvbit_sys::nvbit::rust_nvbit_get_instrs(c, f) };

    instructions
        .pin_mut()
        .iter_mut()
        .map(|mut i| Instruction::new(i.as_mut_ptr()))
        .collect()
}

/// Get control flow graph (CFG)
pub fn get_CFG<'c, 'f>(ctx: &mut Context<'c>, func: &mut Function<'f>) -> CFG<'f> {
    // const CFG_t& nvbit_get_CFG(CUcontext ctx, CUfunction func);
    // TODO
    let cfg = unsafe { bindings::nvbit_get_CFG(ctx.as_mut_ptr(), func.as_mut_ptr()) };
    CFG::default()
}

/// Get the function name from a CUfunction
pub fn get_func_name<'c, 'f>(ctx: &mut Context<'c>, func: &mut Function<'f>) -> &'f str {
    let func_name: *const ffi::c_char =
        unsafe { bindings::nvbit_get_func_name(ctx.as_mut_ptr(), func.as_mut_ptr(), false) };
    unsafe { ffi::CStr::from_ptr(func_name).to_str().unwrap_or_default() }
}

#[derive(Clone, Debug, Default)]
pub struct LineInfo {
    pub line: u32,
    pub file_name: String,
    pub dir_name: String,
}

/// Get line information for a particular instruction offset if available.
///
/// Note: binary must be compiled with --generate-line-info (-lineinfo)
pub fn nvbit_get_line_info<'c, 'f>(
    ctx: &mut Context<'c>,
    func: &mut Function<'f>,
    offset: u32,
) -> LineInfo {
    use ffi::CStr;
    use std::mem::MaybeUninit;

    // bool nvbit_get_line_info(CUcontext cuctx, CUfunction cufunc, uint32_t offset,
    //      char** file_name, char** dir_name, uint32_t* line);
    let mut line = MaybeUninit::<u32>::uninit();
    let mut file_name = MaybeUninit::<*mut ffi::c_char>::uninit();
    let mut dir_name = MaybeUninit::<*mut ffi::c_char>::uninit();
    let success = unsafe {
        bindings::nvbit_get_line_info(
            ctx.as_mut_ptr(),
            func.as_mut_ptr(),
            offset,
            file_name.as_mut_ptr(),
            dir_name.as_mut_ptr(),
            line.as_mut_ptr(),
        )
    };
    if success {
        let file_name = unsafe { CStr::from_ptr(file_name.assume_init()) }
            .to_string_lossy()
            .to_string();
        let dir_name = unsafe { CStr::from_ptr(dir_name.assume_init()) }
            .to_string_lossy()
            .to_string();
        LineInfo {
            line: unsafe { line.assume_init() },
            file_name,
            dir_name,
        }
    } else {
        LineInfo::default()
    }
}

/// Get the SM family
pub fn get_sm_family<'c>(ctx: &mut Context<'c>) -> u32 {
    // uint32_t nvbit_get_sm_family(CUcontext cuctx);
    unsafe { bindings::nvbit_get_sm_family(ctx.as_mut_ptr()) }
}

/// Get PC address of the function
pub fn get_func_addr<'f>(func: &mut Function<'f>) -> u64 {
    // uint64_t nvbit_get_func_addr(CUfunction func);
    unsafe { bindings::nvbit_get_func_addr(func.as_mut_ptr()) }
}

/// Returns true if function is a kernel (i.e. __global__ )
pub fn is_func_kernel<'c, 'f>(ctx: &mut Context<'c>, func: &mut Function<'f>) -> bool {
    // bool nvbit_is_func_kernel(CUcontext ctx, CUfunction func);
    unsafe { bindings::nvbit_is_func_kernel(ctx.as_mut_ptr(), func.as_mut_ptr()) }
}

/// Returns a vector with the sizes (in bytes) of each function argument.
// TODO
// std::vector<int> nvbit_get_kernel_argument_sizes(CUfunction func);

/// Returns shmem base address from CUcontext.
///
/// Shmem range is [shmem_base_addr, shmem_base_addr+16MB) and
/// the base address is 16MB aligned.
pub fn get_shmem_base_addr<'c>(ctx: &mut Context<'c>) -> u64 {
    // uint64_t nvbit_get_shmem_base_addr(CUcontext cuctx);
    unsafe { bindings::nvbit_get_shmem_base_addr(ctx.as_mut_ptr()) }
}

/// Returns local memory base address from CUcontext.
///
/// Local mem range is [shmem_base_addr, shmem_base_addr+16MB) and
/// the base address is 16MB aligned.
pub fn get_local_mem_base_addr<'c>(ctx: &mut Context<'c>) -> u64 {
    // uint64_t nvbit_get_local_mem_base_addr(CUcontext cuctx);
    unsafe { bindings::nvbit_get_local_mem_base_addr(ctx.as_mut_ptr()) }
}

/// Run instrumented on original function
///
/// (and its related functions based on flag value)
pub fn enable_instrumented<'c, 'f>(
    ctx: &mut Context<'c>,
    func: &mut Function<'f>,
    enable: bool,
    related: bool,
) {
    // void nvbit_enable_instrumented(CUcontext ctx, CUfunction func, bool flag,
    //                                bool apply_to_related = true);
    unsafe {
        bindings::nvbit_enable_instrumented(ctx.as_mut_ptr(), func.as_mut_ptr(), enable, related)
    };
}

/// Set arguments at launch time,
/// that will be loaded on input argument of the instrumentation function
pub fn nvbit_set_at_launch<'c, 'f, T>(
    ctx: &mut Context<'c>,
    func: &mut Function<'f>,
    args: T,
    nbytes: u32,
) {
    // todo
    // void nvbit_set_at_launch(CUcontext ctx, CUfunction func, void* buf,
    //                          uint32_t nbytes);
}

/// Notify nvbit of a pthread used by the tool.
///
/// This pthread will not
///  * trigger any call backs even if executing CUDA events of kernel launches.
///  * Multiple pthreads can be registered one after the other.
#[cfg(unix)]
pub fn set_tool_thread<T>(handle: std::thread::JoinHandle<T>) {
    use std::os::unix::thread::JoinHandleExt;
    // void nvbit_set_tool_pthread(pthread_t tool_pthread);
    unsafe { bindings::nvbit_set_tool_pthread(handle.as_pthread_t()) };
}

/// Notify nvbit of a pthread no longer used by the tool.
#[cfg(unix)]
pub fn unset_tool_thread<T>(handle: std::thread::JoinHandle<T>) {
    use std::os::unix::thread::JoinHandleExt;
    // void nvbit_unset_tool_pthread(pthread_t tool_pthread);
    unsafe { bindings::nvbit_unset_tool_pthread(handle.as_pthread_t()) };
}

/// Set nvdisasm
pub fn nvbit_set_nvdisasm(asm: impl AsRef<str>) {
    let asm = ffi::CString::new(asm.as_ref()).unwrap();
    // void nvbit_set_nvdisasm(const char* nvdisasm);
    unsafe { bindings::nvbit_set_nvdisasm(asm.as_ptr()) };
}

///// This function inserts a device function call named "dev_func_name",
///// before or after Instr (ipoint_t { IPOINT_BEFORE, IPOINT_AFTER}).
/////
///// It is important to remember that calls to device functions are
///// identified by name (as opposed to function pointers) and that
///// we declare the function as:
/////
/////       extern "C" __device__ __noinline__
/////
///// to prevent the compiler from optimizing out this device function
///// during compilation.
/////
///// Multiple device functions can be inserted before or after and the
///// order in which they get executed is defined by the order in which
///// they have been inserted.
//pub fn nvbit_insert_call<'a>(
//    instr: &Instruction<'a>,
//    dev_func_name: impl AsRef<str>,
//    point: InsertionPoint,
//) {
//    // void nvbit_insert_call(const Instr* instr, const char* dev_func_name,
//    //                        ipoint_t point);
//    let func_name = ffi::CString::new(dev_func_name.as_ref()).unwrap();
//    unsafe {
//        bindings::nvbit_insert_call(instr.as_ptr() as *const _, func_name.as_ptr(), point.into())
//    }
//}

///// Add int32_t argument to last injected call,
///// value of the (uniform) predicate for this instruction
//pub fn nvbit_add_call_arg_guard_pred_val<'a>(instr: &Instruction<'a>) {
//    // void nvbit_add_call_arg_guard_pred_val(const Instr* instr,
//    //                                        bool is_variadic_arg = false);
//    unsafe { bindings::nvbit_add_call_arg_guard_pred_val(instr.as_ptr() as *const _, false) };
//}

///// Add int32_t argument to last injected call,
///// value of the designated predicate for this instruction
//pub fn nvbit_add_call_arg_pred_val_at<'a>(instr: &Instruction<'a>, pred_num: i32) {
//    // void nvbit_add_call_arg_pred_val_at(const Instr* instr, int pred_num,
//    //                                     bool is_variadic_arg = false);
//    unsafe {
//        bindings::nvbit_add_call_arg_pred_val_at(instr.as_ptr() as *const _, pred_num, false)
//    };
//}

///// Add int32_t argument to last injected call,
///// value of the designated uniform predicate for this instruction
//pub fn nvbit_add_call_arg_upred_val_at<'a>(instr: &Instruction<'a>, upred_num: i32) {
//    // void nvbit_add_call_arg_upred_val_at(const Instr* instr, int upred_num,
//    //                                      bool is_variadic_arg = false);
//    unsafe {
//        bindings::nvbit_add_call_arg_upred_val_at(instr.as_ptr() as *const _, upred_num, false)
//    };
//}

///// Add int32_t argument to last injected call,
///// value of the entire predicate register for this thread
//pub fn nvbit_add_call_arg_pred_reg<'a>(instr: &Instruction<'a>) {
//    // void nvbit_add_call_arg_pred_reg(const Instr* instr,
//    //                                  bool is_variadic_arg = false);
//    unsafe { bindings::nvbit_add_call_arg_pred_reg(instr.as_ptr() as *const _, false) };
//}

///// Add int32_t argument to last injected call,
///// value of the entire uniform predicate register for this thread
//pub fn nvbit_add_call_arg_upred_reg<'a>(instr: &Instruction<'a>) {
//    // void nvbit_add_call_arg_upred_reg(const Instr* instr,
//    //                                   bool is_variadic_arg = false);
//    unsafe { bindings::nvbit_add_call_arg_upred_reg(instr.as_ptr() as *const _, false) };
//}

///// Add uint32_t argument to last injected call, constant 32-bit value
//pub fn nvbit_add_call_arg_const_val32<'a>(instr: &Instruction<'a>, val: u32) {
//    // void nvbit_add_call_arg_const_val32(const Instr* instr, uint32_t val,
//    //                                     bool is_variadic_arg = false);
//    unsafe { bindings::nvbit_add_call_arg_const_val32(instr.as_ptr() as *const _, val, false) };
//}

///// Add uint64_t argument to last injected call,
///// constant 64-bit value
//pub fn nvbit_add_call_arg_const_val64<'a>(instr: &Instruction<'a>, val: u64) {
//    // void nvbit_add_call_arg_const_val64(const Instr* instr, uint64_t val,
//    //                                     bool is_variadic_arg = false);
//    unsafe { bindings::nvbit_add_call_arg_const_val64(instr.as_ptr() as *const _, val, false) };
//}

///// Add uint32_t argument to last injected call,
///// content of the register reg_num
//pub fn nvbit_add_call_arg_reg_val<'a>(instr: &Instruction<'a>, reg_num: i32) {
//    // void nvbit_add_call_arg_reg_val(const Instr* instr, int reg_num,
//    //                                 bool is_variadic_arg = false);
//    unsafe { bindings::nvbit_add_call_arg_reg_val(instr.as_ptr() as *const _, reg_num, false) };
//}

///// Add uint32_t argument to last injected call,
///// content of theuniform register reg_num
//pub fn nvbit_add_call_arg_ureg_val<'a>(instr: &Instruction<'a>, reg_num: i32) {
//    // void nvbit_add_call_arg_ureg_val(const Instr* instr, int reg_num,
//    //                                  bool is_variadic_arg = false);
//    unsafe { bindings::nvbit_add_call_arg_ureg_val(instr.as_ptr() as *const _, reg_num, false) };
//}

///// Add uint32_t argument to last injected call,
///// 32-bit at launch value at offset "offset",
///// set at launch time with nvbit_set_at_launch
//pub fn nvbit_add_call_arg_launch_val32<'a>(instr: &Instruction<'a>, offset: i32) {
//    // void nvbit_add_call_arg_launch_val32(const Instr* instr, int offset,
//    //                                      bool is_variadic_arg = false);
//    unsafe { bindings::nvbit_add_call_arg_launch_val32(instr.as_ptr() as *const _, offset, false) };
//}

///// Add uint64_t argument to last injected call,
///// 64-bit at launch value at offset "offset",
///// set at launch time with nvbit_set_at_launch
//pub fn nvbit_add_call_arg_launch_val64<'a>(instr: &Instruction<'a>, offset: i32) {
//    // void nvbit_add_call_arg_launch_val64(const Instr* instr, int offset,
//    //                                      bool is_variadic_arg = false);
//    unsafe { bindings::nvbit_add_call_arg_launch_val64(instr.as_ptr() as *const _, offset, false) };
//}

///// Add uint32_t argument to last injected call,
///// constant bank value at c[bankid][bankoffset]
//pub fn nvbit_add_call_arg_cbank_val<'a>(instr: &Instruction<'a>, bank_id: i32, bank_offset: i32) {
//    // void nvbit_add_call_arg_cbank_val(const Instr* instr, int bankid,
//    //                                   int bankoffset, bool is_variadic_arg = false);
//    unsafe {
//        bindings::nvbit_add_call_arg_cbank_val(
//            instr.as_ptr() as *const _,
//            bank_id,
//            bank_offset,
//            false,
//        )
//    };
//}

///// The 64-bit memory reference address accessed by this instruction.
/////
///// Typically memory instructions have only 1 MREF so in general id = 0
//pub fn nvbit_add_call_arg_mref_addr64<'a>(instr: &Instruction<'a>, id: i32) {
//    // void nvbit_add_call_arg_mref_addr64(const Instr* instr, int id = 0,
//    //                                     bool is_variadic_arg = false);
//    unsafe { bindings::nvbit_add_call_arg_mref_addr64(instr.as_ptr() as *const _, id, false) };
//}

///// Remove the original instruction
//pub fn nvbit_remove_orig<'a>(instr: &Instruction<'a>) {
//    // void nvbit_remove_orig(const Instr* instr);
//    unsafe { bindings::nvbit_remove_orig(instr.as_ptr() as *const _) };
//}
