use super::{Context, Function, Instruction, CFG};
use nvbit_sys::{bindings, nvbit};
use std::ffi;

/// Return nvbit version
pub fn version() -> &'static str {
    std::str::from_utf8(bindings::NVBIT_VERSION).unwrap()
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
pub fn get_cfg<'c, 'f>(ctx: &mut Context<'c>, func: &mut Function<'f>) -> CFG<'f> {
    // const CFG_t& nvbit_get_CFG(CUcontext ctx, CUfunction func);
    // TODO
    let _cfg = unsafe { bindings::nvbit_get_CFG(ctx.as_mut_ptr(), func.as_mut_ptr()) };
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
    unsafe { bindings::nvbit_get_sm_family(ctx.as_mut_ptr()) }
}

/// Get PC address of the function
pub fn get_func_addr<'f>(func: &mut Function<'f>) -> u64 {
    unsafe { bindings::nvbit_get_func_addr(func.as_mut_ptr()) }
}

/// Returns true if function is a kernel (i.e. __global__ )
pub fn is_func_kernel<'c, 'f>(ctx: &mut Context<'c>, func: &mut Function<'f>) -> bool {
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
    unsafe { bindings::nvbit_get_shmem_base_addr(ctx.as_mut_ptr()) }
}

/// Returns local memory base address from CUcontext.
///
/// Local mem range is [shmem_base_addr, shmem_base_addr+16MB) and
/// the base address is 16MB aligned.
pub fn get_local_mem_base_addr<'c>(ctx: &mut Context<'c>) -> u64 {
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
    unsafe {
        bindings::nvbit_enable_instrumented(ctx.as_mut_ptr(), func.as_mut_ptr(), enable, related)
    };
}

/// Set arguments at launch time,
/// that will be loaded on input argument of the instrumentation function
pub fn nvbit_set_at_launch<'c, 'f, T>(
    _ctx: &mut Context<'c>,
    _func: &mut Function<'f>,
    _args: T,
    _nbytes: u32,
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
    unsafe { bindings::nvbit_set_tool_pthread(handle.as_pthread_t()) };
}

/// Notify nvbit of a pthread no longer used by the tool.
#[cfg(unix)]
pub fn unset_tool_thread<T>(handle: std::thread::JoinHandle<T>) {
    use std::os::unix::thread::JoinHandleExt;
    unsafe { bindings::nvbit_unset_tool_pthread(handle.as_pthread_t()) };
}

/// Set nvdisasm
pub fn nvbit_set_nvdisasm(asm: impl AsRef<str>) {
    let asm = ffi::CString::new(asm.as_ref()).unwrap();
    unsafe { bindings::nvbit_set_nvdisasm(asm.as_ptr()) };
}
