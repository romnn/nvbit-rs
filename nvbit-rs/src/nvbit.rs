#![allow(clippy::missing_panics_doc)]

use super::{Context, Function, Instruction, CFG};
use nvbit_sys::{bindings, nvbit};
use std::ffi;

const MB: u64 = 1024 * 1024;

/// Shim that wraps a CUDA event name.
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct CudaEventName {
    inner: *const ffi::c_char,
}

impl std::fmt::Display for CudaEventName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

impl AsRef<str> for CudaEventName {
    fn as_ref(&self) -> &str {
        unsafe { ffi::CStr::from_ptr(self.inner).to_str().unwrap() }
    }
}

impl std::ops::Deref for CudaEventName {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

/// Return nvbit version
///
/// # Panics
/// Panics if the `nvbit` version string is not a valid UTF8 string.
#[must_use]
pub fn version() -> &'static str {
    let version = unsafe { ffi::CStr::from_bytes_with_nul_unchecked(bindings::NVBIT_VERSION) };
    version.to_str().unwrap()
}

/// Get related functions for given `CUfunction`
#[must_use]
pub fn get_related_functions<'f>(
    ctx: &mut Context<'_>,
    func: &mut Function<'f>,
) -> Vec<Function<'f>> {
    let c = ctx.as_mut_ptr();
    let f = func.as_mut_ptr();
    let mut related_functions: cxx::UniquePtr<cxx::Vector<nvbit::CUfunctionShim>> =
        unsafe { nvbit::rust_nvbit_get_related_functions(c, f) };

    related_functions
        .pin_mut()
        .iter_mut()
        .map(|mut f| Function::wrap(f.as_mut_ptr()))
        .collect()
}

/// Get instructions composing the `CUfunction`
#[must_use]
pub fn get_instrs<'f>(ctx: &mut Context<'_>, func: &mut Function<'f>) -> Vec<Instruction<'f>> {
    let c = ctx.as_mut_ptr();
    let f = func.as_mut_ptr();
    let mut instructions: cxx::UniquePtr<cxx::Vector<nvbit::InstrShim>> =
        unsafe { nvbit_sys::nvbit::rust_nvbit_get_instrs(c, f) };

    instructions
        .pin_mut()
        .iter_mut()
        .map(|mut i| Instruction::new(func.clone(), i.as_mut_ptr()))
        .collect()
}

/// Get control flow graph (CFG)
#[must_use]
pub fn get_cfg<'f>(ctx: &mut Context<'_>, func: &mut Function<'f>) -> CFG<'f> {
    // const CFG_t& nvbit_get_CFG(CUcontext ctx, CUfunction func);
    // TODO
    let _cfg = unsafe { bindings::nvbit_get_CFG(ctx.as_mut_ptr(), func.as_mut_ptr()) };
    CFG::default()
}

/// Get the mangled function name from a `CUfunction`.
#[must_use]
pub fn get_func_name_mangled<'f>(ctx: &mut Context<'_>, func: &mut Function<'f>) -> &'f str {
    let mangled = true;
    let func_name: *const ffi::c_char =
        unsafe { bindings::nvbit_get_func_name(ctx.as_mut_ptr(), func.as_mut_ptr(), mangled) };
    unsafe { ffi::CStr::from_ptr(func_name).to_str().unwrap_or_default() }
}

/// Get the unmangled function name from a `CUfunction`.
#[must_use]
pub fn get_func_name_unmangled<'f>(ctx: &mut Context<'_>, func: &mut Function<'f>) -> &'f str {
    let mangled = false;
    let func_name: *const ffi::c_char =
        unsafe { bindings::nvbit_get_func_name(ctx.as_mut_ptr(), func.as_mut_ptr(), mangled) };
    unsafe { ffi::CStr::from_ptr(func_name).to_str().unwrap_or_default() }
}

/// Line information of an instruction.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LineInfo {
    pub line: u32,
    pub file_name: String,
    pub dir_name: String,
}

/// Get line information for a particular instruction offset if available.
///
/// Note: binary must be compiled with --generate-line-info (-lineinfo).
#[must_use]
pub fn get_line_info(
    ctx: &mut Context<'_>,
    func: &mut Function<'_>,
    offset: u32,
) -> Option<LineInfo> {
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
        Some(LineInfo {
            line: unsafe { line.assume_init() },
            file_name,
            dir_name,
        })
    } else {
        None
        // LineInfo::default()
    }
}

/// Get the SM family
#[must_use]
pub fn get_sm_family(ctx: &mut Context<'_>) -> u32 {
    unsafe { bindings::nvbit_get_sm_family(ctx.as_mut_ptr()) }
}

/// Get PC address of the function
#[must_use]
pub fn get_func_addr(func: &mut Function<'_>) -> u64 {
    unsafe { bindings::nvbit_get_func_addr(func.as_mut_ptr()) }
}

/// Returns true if function is a kernel (i.e. `__global__`)
#[must_use]
pub fn is_func_kernel(ctx: &mut Context<'_>, func: &mut Function<'_>) -> bool {
    unsafe { bindings::nvbit_is_func_kernel(ctx.as_mut_ptr(), func.as_mut_ptr()) }
}

/// Returns a vector with the sizes (in bytes) of each function argument.
// TODO
// std::vector<int> nvbit_get_kernel_argument_sizes(CUfunction func);

/// Shared memory base address from `CUcontext` (inclusive).
///
/// Shared memory address range is [`shmem_base_addr`, `shmem_base_addr`+16MB).
/// The base address is 16MB aligned.
#[must_use]
pub fn shmem_base_addr(ctx: &mut Context<'_>) -> u64 {
    unsafe { bindings::nvbit_get_shmem_base_addr(ctx.as_mut_ptr()) }
}

/// Shared memory address limit from `CUcontext` (exclusive).
///
/// This is equal to `shmem_base_addr`+16MB.
/// The address is 16MB aligned.
#[must_use]
pub fn shmem_addr_limit(ctx: &mut Context<'_>) -> u64 {
    shmem_base_addr(ctx) + (16 * MB)
}

/// Local memory base address from `CUcontext` (inclusive).
///
/// Local memory address range is [`local_base_addr`, `local_base_addr`+16MB).
/// The base address is 16MB aligned.
#[must_use]
pub fn local_mem_base_addr(ctx: &mut Context<'_>) -> u64 {
    unsafe { bindings::nvbit_get_local_mem_base_addr(ctx.as_mut_ptr()) }
}

/// Local memory address limit from `CUcontext` (exclusive).
///
/// This is equal to `local_base_addr`+16MB.
/// The address is 16MB aligned.
#[must_use]
pub fn local_mem_addr_limit(ctx: &mut Context<'_>) -> u64 {
    local_mem_base_addr(ctx) + (16 * MB)
}

/// Run instrumentation on original function.
///
/// Also enables instrumentation for its related functions
/// if the `related` flag is set.
pub fn enable_instrumented(
    ctx: &mut Context<'_>,
    func: &mut Function<'_>,
    enable: bool,
    related: bool,
) {
    unsafe {
        bindings::nvbit_enable_instrumented(ctx.as_mut_ptr(), func.as_mut_ptr(), enable, related);
    };
}

/// Set arguments at launch time,
///
/// Will be loaded on input argument of the instrumentation function.
///
/// # Panics
/// If the size of T in bytes cannot be determined.
pub fn set_at_launch<T>(ctx: &mut Context<'_>, func: &mut Function<'_>, value: &mut T) {
    let nbytes = std::mem::size_of::<T>();
    unsafe {
        bindings::nvbit_set_at_launch(
            ctx.as_mut_ptr(),
            func.as_mut_ptr(),
            (value as *mut T).cast(),
            nbytes.try_into().unwrap(),
        );
    };
}

/// Notify nvbit of a pthread used by the tool.
///
/// This pthread will not
///  * trigger any call backs even if executing CUDA events of kernel launches.
///  * Multiple pthreads can be registered one after the other.
#[cfg(unix)]
pub fn set_tool_thread<T>(handle: &std::thread::JoinHandle<T>) {
    use std::os::unix::thread::JoinHandleExt;
    unsafe { bindings::nvbit_set_tool_pthread(handle.as_pthread_t()) };
}

/// Notify nvbit of a pthread no longer used by the tool.
#[cfg(unix)]
pub fn unset_tool_thread<T>(handle: &std::thread::JoinHandle<T>) {
    use std::os::unix::thread::JoinHandleExt;
    unsafe { bindings::nvbit_unset_tool_pthread(handle.as_pthread_t()) };
}

/// Set nvdisasm
///
/// # Panics
/// Panics if the supplied assembly string is not a valid C string,
/// e.g. it contains an internal 0 byte.
pub fn set_nvdisasm(asm: impl AsRef<str>) {
    let asm = ffi::CString::new(asm.as_ref()).unwrap();
    unsafe { bindings::nvbit_set_nvdisasm(asm.as_ptr()) };
}
