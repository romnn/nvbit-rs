use nvbit_sys::{model, CudaResult, IntoCudaResult};
use std::ffi;
use std::marker::PhantomData;

/// Opaque handle to a CUDA `CUdeviceptr_v1` device.
///
/// Wraps either `nvbit_sys::CUdeviceptr_v1` (`u32`)
/// or `nvbit_sys::CUdeviceptr_v2` (`u64`) - both of which are
/// upcasted to `u64`.
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Device {
    inner: u64,
    // inner: nvbit_sys::CUdeviceptr_v2,
}

impl From<&Device> for model::Device {
    #[inline]
    fn from(val: &Device) -> Self {
        model::Device(val.as_ptr())
    }
}

impl Device {
    /// Creates a new `Device` wrapping a `CUdeviceptr_v1`.
    #[inline]
    #[must_use]
    pub fn wrap(ptr: impl Into<u64>) -> Self {
        Self { inner: ptr.into() }
    }

    /// Gets a `nvbit_sys::CUdeviceptr_v2` pointer to the device
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> nvbit_sys::CUdeviceptr_v2 {
        self.as_ptr_v2()
    }

    /// Gets a `nvbit_sys::CUdeviceptr_v1` pointer to the device
    ///
    /// # Panics
    /// If the device pointer cannot be represented using a 32bit pointer.
    #[inline]
    #[must_use]
    pub fn as_ptr_v1(&self) -> nvbit_sys::CUdeviceptr_v1 {
        self.inner.try_into().unwrap()
    }

    /// Gets a `nvbit_sys::CUdeviceptr_v2` pointer to the device
    #[inline]
    #[must_use]
    pub fn as_ptr_v2(&self) -> nvbit_sys::CUdeviceptr_v2 {
        self.inner
    }
}

/// Opaque handle to a CUDA `CUcontext` context.
///
/// The handle can be used to uniquely identify a context,
/// but not for interacting with the context.
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct ContextHandle<'a> {
    inner: nvbit_sys::CUcontext,
    module: PhantomData<&'a ()>,
}

unsafe impl<'a> Send for ContextHandle<'a> {}
unsafe impl<'a> Sync for ContextHandle<'a> {}

impl<'a> ContextHandle<'a> {
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const nvbit_sys::CUctx_st {
        self.inner
    }
}

impl From<&ContextHandle<'_>> for model::Context {
    #[inline]
    fn from(val: &ContextHandle<'_>) -> Self {
        model::Context(val.as_ptr() as u64)
    }
}

/// Opaque handle to a CUDA `CUcontext` context.
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Context<'a> {
    inner: nvbit_sys::CUcontext,
    module: PhantomData<&'a nvbit_sys::CUcontext>,
}

unsafe impl<'a> Send for Context<'a> {}
unsafe impl<'a> Sync for Context<'a> {}

impl<'a> Context<'a> {
    /// Creates a new `Context` wrapping a `CUcontext`.
    #[inline]
    #[must_use]
    pub fn wrap(inner: nvbit_sys::CUcontext) -> Self {
        Self {
            inner,
            module: PhantomData,
        }
    }

    /// Returns a handle to this context.
    #[inline]
    #[must_use]
    pub fn handle(&self) -> ContextHandle<'a> {
        ContextHandle {
            inner: self.inner,
            module: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const nvbit_sys::CUctx_st {
        self.inner.cast()
    }

    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut nvbit_sys::CUctx_st {
        self.inner
    }
}

impl From<&Context<'_>> for model::Context {
    #[inline]
    fn from(val: &Context<'_>) -> Self {
        model::Context(val.as_ptr() as u64)
    }
}

/// A handle to a CUDA `CUfunction` function.
///
/// The handle can be used to uniquely identify a context,
/// but not for interacting with the context.
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct FunctionHandle<'a> {
    inner: nvbit_sys::CUfunction,
    module: PhantomData<&'a ()>,
}

unsafe impl<'a> Send for FunctionHandle<'a> {}
unsafe impl<'a> Sync for FunctionHandle<'a> {}

impl<'a> FunctionHandle<'a> {
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> nvbit_sys::CUfunction {
        self.inner.cast()
    }
}

impl From<&FunctionHandle<'_>> for model::Function {
    #[inline]
    fn from(val: &FunctionHandle<'_>) -> Self {
        model::Function(val.as_ptr() as u64)
    }
}

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct Stream<'a> {
    inner: nvbit_sys::CUstream,
    ctx: PhantomData<Context<'a>>,
}

unsafe impl<'a> Send for Stream<'a> {}
unsafe impl<'a> Sync for Stream<'a> {}

impl<'a> Stream<'a> {
    /// Creates a new `Stream` wrapping a `CUstream`.
    #[inline]
    #[must_use]
    pub fn wrap(inner: nvbit_sys::CUstream) -> Self {
        Self {
            inner,
            ctx: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const nvbit_sys::CUstream_st {
        self.inner.cast()
    }

    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut nvbit_sys::CUstream_st {
        self.inner.cast()
    }
}

impl From<&Stream<'_>> for model::Stream {
    #[inline]
    fn from(val: &Stream<'_>) -> Self {
        model::Stream(val.as_ptr() as u64)
    }
}

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct Function<'a> {
    inner: nvbit_sys::CUfunction,
    ctx: PhantomData<Context<'a>>,
}

unsafe impl<'a> Send for Function<'a> {}
unsafe impl<'a> Sync for Function<'a> {}

impl<'a> Function<'a> {
    /// Creates a new `Function` wrapping a `CUfunction`.
    #[inline]
    #[must_use]
    pub fn wrap(inner: nvbit_sys::CUfunction) -> Self {
        Self {
            inner,
            ctx: PhantomData,
        }
    }

    /// Returns a handle to this function.
    #[inline]
    #[must_use]
    pub fn handle(&self) -> FunctionHandle<'a> {
        FunctionHandle {
            inner: self.inner,
            module: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const nvbit_sys::CUfunc_st {
        self.inner.cast()
    }

    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut nvbit_sys::CUfunc_st {
        self.inner
    }

    #[inline]
    #[must_use]
    pub fn name<'c>(&mut self, ctx: &mut Context<'c>) -> &'a str {
        self.unmangled_name(ctx)
    }

    #[inline]
    #[must_use]
    pub fn mangled_name<'c>(&mut self, ctx: &mut Context<'c>) -> &'a str {
        super::get_func_name_mangled(ctx, self)
    }

    #[inline]
    #[must_use]
    pub fn unmangled_name<'c>(&mut self, ctx: &mut Context<'c>) -> &'a str {
        super::get_func_name_unmangled(ctx, self)
    }

    #[inline]
    #[must_use]
    pub fn addr(&mut self) -> u64 {
        super::get_func_addr(self)
    }

    #[inline]
    pub fn enable_instrumented(
        &mut self,
        ctx: &mut Context<'_>,
        flag: bool,
        apply_to_related: bool,
    ) {
        super::enable_instrumented(ctx, self, flag, apply_to_related);
    }

    #[inline]
    #[must_use]
    pub fn related_functions<'c>(&mut self, ctx: &mut Context<'c>) -> Vec<Function<'a>> {
        super::get_related_functions(ctx, self)
    }

    #[inline]
    #[must_use]
    pub fn instructions<'c>(&mut self, ctx: &mut Context<'c>) -> Vec<super::Instruction<'a>> {
        super::get_instrs(ctx, self)
    }
}

impl From<&Function<'_>> for model::Function {
    #[inline]
    fn from(val: &Function<'_>) -> Self {
        model::Function(val.as_ptr() as u64)
    }
}

impl<'a> Function<'a> {
    /// Returns the number of registers for this function.
    ///
    /// # Errors
    /// Returns an error if the CUDA attribute can not be read.
    pub fn num_registers(&mut self) -> CudaResult<i32> {
        let mut value = 0;
        self.get_attribute(&mut value, model::FunctionAttribute::NumRegs)?;
        Ok(value)
    }

    /// Returns the number of shared memory bytes for this function.
    ///
    /// # Errors
    /// Returns an error if the CUDA attribute can not be read.
    pub fn shared_memory_bytes(&mut self) -> CudaResult<usize> {
        let mut value = 0;
        self.get_attribute(&mut value, model::FunctionAttribute::SharedSizeBytes)?;
        Ok(value)
    }

    /// Returns the binary version of this function.
    ///
    /// # Errors
    /// Returns an error if the CUDA attribute can not be read.
    pub fn binary_version(&mut self) -> CudaResult<i32> {
        let mut value = 0;
        self.get_attribute(&mut value, model::FunctionAttribute::BinaryVersion)?;
        Ok(value)
    }

    /// Gets an attribute for this function.
    ///
    /// See: [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaFuncAttributes.html#structcudaFuncAttributes)
    ///
    /// # Errors
    /// Returns an error if the CUDA attribute can not be read.
    pub fn get_attribute<T>(
        &mut self,
        dest: &mut T,
        attr: model::FunctionAttribute,
    ) -> CudaResult<()> {
        let result = unsafe {
            nvbit_sys::cuFuncGetAttribute((dest as *mut T).cast(), attr.into(), self.as_mut_ptr())
        };
        result.into_result()?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum EventParams<'a> {
    Launch {
        func: Function<'a>,
    },
    KernelLaunch {
        func: Function<'a>,
        grid: model::Dim,
        block: model::Dim,
        shared_mem_bytes: u32,
        h_stream: Stream<'a>,
        kernel_params: *mut *mut ffi::c_void,
        extra: *mut *mut ffi::c_void,
    },
    MemFree {
        device_ptr: u64,
    },
    MemAlloc {
        device_ptr: u64,
        num_bytes: u64,
    },
    MemCopyHostToDevice {
        dest_device: Device,
        src_host: *const ffi::c_void,
        num_bytes: u64,
    },
    MemCopyDeviceToHost {
        src_device: Device,
        dest_host: *const ffi::c_void,
        num_bytes: u64,
    },
    ProfilerStart,
    ProfilerStop,
}

impl<'a> EventParams<'a> {
    pub fn new(cbid: nvbit_sys::nvbit_api_cuda_t, params: *mut ffi::c_void) -> Option<Self> {
        use nvbit_sys::nvbit_api_cuda_t as cuda_t;
        match cbid {
            cuda_t::API_CUDA_cuMemFree => {
                let p: nvbit_sys::cuMemFree_params = unsafe { *params.cast() };
                Some(Self::MemFree {
                    device_ptr: u64::from(p.dptr),
                })
            }
            cuda_t::API_CUDA_cuMemFree_v2 => {
                let p: nvbit_sys::cuMemFree_v2_params = unsafe { *params.cast() };
                Some(Self::MemFree { device_ptr: p.dptr })
            }
            cuda_t::API_CUDA_cuMemAlloc => {
                let p: nvbit_sys::cuMemAlloc_params = unsafe { *params.cast() };
                Some(Self::MemAlloc {
                    device_ptr: u64::from(unsafe { *p.dptr }),
                    num_bytes: p.bytesize.into(),
                })
            }
            cuda_t::API_CUDA_cuMemAlloc_v2 => {
                let p: nvbit_sys::cuMemAlloc_v2_params = unsafe { *params.cast() };
                Some(Self::MemAlloc {
                    device_ptr: unsafe { *p.dptr },
                    num_bytes: p.bytesize,
                })
            }
            cuda_t::API_CUDA_cuMemcpyDtoH => {
                let p: nvbit_sys::cuMemcpyDtoH_params_st = unsafe { *params.cast() };
                Some(Self::MemCopyDeviceToHost {
                    src_device: Device::wrap(p.srcDevice),
                    dest_host: p.dstHost,
                    num_bytes: p.ByteCount.into(),
                })
            }
            cuda_t::API_CUDA_cuMemcpyDtoH_v2 => {
                let p: nvbit_sys::cuMemcpyDtoH_v2_params_st = unsafe { *params.cast() };
                Some(Self::MemCopyDeviceToHost {
                    src_device: Device::wrap(p.srcDevice),
                    dest_host: p.dstHost,
                    num_bytes: p.ByteCount,
                })
            }
            cuda_t::API_CUDA_cuMemcpyHtoD => {
                let p: nvbit_sys::cuMemcpyHtoD_params = unsafe { *params.cast() };
                Some(Self::MemCopyHostToDevice {
                    dest_device: Device::wrap(p.dstDevice),
                    src_host: p.srcHost,
                    num_bytes: p.ByteCount.into(),
                })
            }
            cuda_t::API_CUDA_cuMemcpyHtoD_v2 => {
                let p: nvbit_sys::cuMemcpyHtoD_v2_params = unsafe { *params.cast() };
                Some(Self::MemCopyHostToDevice {
                    dest_device: Device::wrap(p.dstDevice),
                    src_host: p.srcHost,
                    num_bytes: p.ByteCount,
                })
            }
            cuda_t::API_CUDA_cuProfilerStart => Some(Self::ProfilerStart),
            cuda_t::API_CUDA_cuProfilerStop => Some(Self::ProfilerStop),
            cuda_t::API_CUDA_cuLaunch
            | cuda_t::API_CUDA_cuLaunchGrid
            | cuda_t::API_CUDA_cuLaunchGridAsync => {
                let p: nvbit_sys::cuLaunch_params = unsafe { *params.cast() };
                Some(Self::Launch {
                    func: Function::wrap(p.f),
                })
            }
            cuda_t::API_CUDA_cuLaunchKernel_ptsz | cuda_t::API_CUDA_cuLaunchKernel => {
                let p: nvbit_sys::cuLaunchKernel_params = unsafe { *params.cast() };
                Some(Self::KernelLaunch {
                    func: Function::wrap(p.f),
                    grid: model::Dim {
                        x: p.gridDimX,
                        y: p.gridDimY,
                        z: p.gridDimZ,
                    },
                    block: model::Dim {
                        x: p.blockDimX,
                        y: p.blockDimY,
                        z: p.blockDimZ,
                    },
                    shared_mem_bytes: p.sharedMemBytes,
                    h_stream: Stream::wrap(p.hStream),
                    kernel_params: p.kernelParams,
                    extra: p.extra,
                })
            }
            // 600+ more api calls to cover here :(
            // _ => todo!(),
            _ => None,
        }
    }
}
