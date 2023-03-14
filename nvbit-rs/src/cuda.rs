use super::{CudaResult, IntoCudaResult};
use std::ffi;
use std::marker::PhantomData;

/// Opaque handle to a CUDA `CUdeviceptr_v1` device.
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug, serde::Serialize)]
pub struct Device {
    inner: nvbit_sys::CUdeviceptr_v1,
}

impl Device {
    /// Creates a new `Device` wrapping a `CUdeviceptr_v1`.
    #[inline]
    #[must_use]
    pub fn wrap(inner: nvbit_sys::CUdeviceptr_v1) -> Self {
        Self { inner }
    }
}

/// Opaque handle to a CUDA `CUcontext` context.
///
/// The handle can be used to uniquely identify a context,
/// but not for interacting with the context.
#[derive(PartialEq, Eq, Hash, Debug, serde::Serialize)]
pub struct ContextHandle<'a> {
    #[serde(serialize_with = "crate::to_raw_ptr")]
    inner: nvbit_sys::CUcontext,
    module: PhantomData<&'a ()>,
}

unsafe impl<'a> Send for ContextHandle<'a> {}
unsafe impl<'a> Sync for ContextHandle<'a> {}

/// Opaque handle to a CUDA `CUcontext` context.
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug, serde::Serialize)]
pub struct Context<'a> {
    #[serde(serialize_with = "crate::to_raw_ptr")]
    inner: nvbit_sys::CUcontext,
    module: PhantomData<&'a ()>,
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
        self.inner.cast()
    }
}

/// A handle to a CUDA `CUfunction` function.
///
/// The handle can be used to uniquely identify a context,
/// but not for interacting with the context.
#[derive(PartialEq, Eq, Hash, Debug, serde::Serialize)]
pub struct FunctionHandle<'a> {
    #[serde(serialize_with = "crate::to_raw_ptr")]
    inner: nvbit_sys::CUfunction,
    module: PhantomData<&'a ()>,
}

unsafe impl<'a> Send for FunctionHandle<'a> {}
unsafe impl<'a> Sync for FunctionHandle<'a> {}

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug, Clone, serde::Serialize)]
pub struct Stream<'a> {
    #[serde(serialize_with = "crate::to_raw_ptr")]
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

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug, Clone, serde::Serialize)]
pub struct Function<'a> {
    #[serde(serialize_with = "crate::to_raw_ptr")]
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
        self.inner.cast()
    }

    #[inline]
    #[must_use]
    pub fn name<'c>(&mut self, ctx: &mut Context<'c>) -> &'a str {
        super::get_func_name(ctx, self)
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

/// CUDA function attribute.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum FunctionAttribute {
    MaxThreadsPerBlock,
    SharedSizeBytes,
    ConstSizeBytes,
    LocalSizeBytes,
    NumRegs,
    PTXVersion,
    BinaryVersion,
    CacheModeCA,
    MaxDynamicSharedSizeBytes,
    PreferredSharedMemoryCarveout,
    Max,
}

impl From<FunctionAttribute> for nvbit_sys::CUfunction_attribute_enum {
    fn from(point: FunctionAttribute) -> nvbit_sys::CUfunction_attribute_enum {
        use nvbit_sys::CUfunction_attribute_enum as ATTR;
        match point {
            FunctionAttribute::MaxThreadsPerBlock => ATTR::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            FunctionAttribute::SharedSizeBytes => ATTR::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            FunctionAttribute::ConstSizeBytes => ATTR::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
            FunctionAttribute::LocalSizeBytes => ATTR::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            FunctionAttribute::NumRegs => ATTR::CU_FUNC_ATTRIBUTE_NUM_REGS,
            FunctionAttribute::PTXVersion => ATTR::CU_FUNC_ATTRIBUTE_PTX_VERSION,
            FunctionAttribute::BinaryVersion => ATTR::CU_FUNC_ATTRIBUTE_BINARY_VERSION,
            FunctionAttribute::CacheModeCA => ATTR::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
            FunctionAttribute::MaxDynamicSharedSizeBytes => {
                ATTR::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            }
            FunctionAttribute::PreferredSharedMemoryCarveout => {
                ATTR::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            }
            FunctionAttribute::Max => ATTR::CU_FUNC_ATTRIBUTE_MAX,
        }
    }
}

impl<'a> Function<'a> {
    /// Returns the number of registers for this function.
    ///
    /// # Errors
    /// Returns an error if the CUDA attribute can not be read.
    pub fn num_registers(&mut self) -> CudaResult<i32> {
        let mut value = 0;
        self.get_attribute(&mut value, FunctionAttribute::NumRegs)?;
        Ok(value)
    }

    /// Returns the number of shared memory bytes for this function.
    ///
    /// # Errors
    /// Returns an error if the CUDA attribute can not be read.
    pub fn shared_memory_bytes(&mut self) -> CudaResult<usize> {
        let mut value = 0;
        self.get_attribute(&mut value, FunctionAttribute::SharedSizeBytes)?;
        Ok(value)
    }

    /// Returns the binary version of this function.
    ///
    /// # Errors
    /// Returns an error if the CUDA attribute can not be read.
    pub fn binary_version(&mut self) -> CudaResult<i32> {
        let mut value = 0;
        self.get_attribute(&mut value, FunctionAttribute::BinaryVersion)?;
        Ok(value)
    }

    /// Gets an attribute for this function.
    ///
    /// See: [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaFuncAttributes.html#structcudaFuncAttributes)
    ///
    /// # Errors
    /// Returns an error if the CUDA attribute can not be read.
    pub fn get_attribute<T>(&mut self, dest: &mut T, attr: FunctionAttribute) -> CudaResult<()> {
        let result = unsafe {
            nvbit_sys::cuFuncGetAttribute((dest as *mut T).cast(), attr.into(), self.as_mut_ptr())
        };
        result.into_result()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Dim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({},{},{})", self.x, self.y, self.z)
    }
}

#[derive(Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum EventParams<'a> {
    Launch {
        func: Function<'a>,
    },
    KernelLaunch {
        func: Function<'a>,
        grid: Dim,
        block: Dim,
        shared_mem_bytes: u32,
        h_stream: Stream<'a>,
        #[serde(serialize_with = "crate::to_raw_ptr")]
        kernel_params: *mut *mut ffi::c_void,
        #[serde(serialize_with = "crate::to_raw_ptr")]
        extra: *mut *mut ffi::c_void,
    },
    MemCopyHostToDevice {
        dest_device: Device,
        #[serde(serialize_with = "crate::to_raw_ptr")]
        src_host: *const ffi::c_void,
        bytes: u32,
    },
    ProfilerStart,
    ProfilerStop,
}

impl<'a> EventParams<'a> {
    pub fn new(cbid: nvbit_sys::nvbit_api_cuda_t, params: *mut ffi::c_void) -> Option<Self> {
        use nvbit_sys::nvbit_api_cuda_t as cuda_t;
        match cbid {
            cuda_t::API_CUDA_cuMemcpyHtoD_v2 => {
                let p = unsafe { &mut *params.cast::<nvbit_sys::cuMemcpyHtoD_params>() };
                Some(Self::MemCopyHostToDevice {
                    dest_device: Device::wrap(p.dstDevice),
                    src_host: p.srcHost,
                    bytes: p.ByteCount,
                })
            }
            cuda_t::API_CUDA_cuProfilerStart => Some(Self::ProfilerStart),
            cuda_t::API_CUDA_cuProfilerStop => Some(Self::ProfilerStop),
            cuda_t::API_CUDA_cuLaunch
            | cuda_t::API_CUDA_cuLaunchGrid
            | cuda_t::API_CUDA_cuLaunchGridAsync => {
                let p = unsafe { &mut *params.cast::<nvbit_sys::cuLaunch_params>() };
                Some(Self::Launch {
                    func: Function::wrap(p.f),
                })
            }
            cuda_t::API_CUDA_cuLaunchKernel_ptsz | cuda_t::API_CUDA_cuLaunchKernel => {
                let p = unsafe { &mut *params.cast::<nvbit_sys::cuLaunchKernel_params>() };
                Some(Self::KernelLaunch {
                    func: Function::wrap(p.f),
                    grid: Dim {
                        x: p.gridDimX,
                        y: p.gridDimY,
                        z: p.gridDimZ,
                    },
                    block: Dim {
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
