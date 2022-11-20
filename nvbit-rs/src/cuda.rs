use super::{CudaResult, IntoCudaResult};
use nvbit_sys::bindings;
use std::marker::PhantomData;

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct ContextHandle<'a> {
    inner: bindings::CUcontext,
    module: PhantomData<&'a ()>,
}

unsafe impl<'a> Send for ContextHandle<'a> {}
unsafe impl<'a> Sync for ContextHandle<'a> {}

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Context<'a> {
    inner: bindings::CUcontext,
    module: PhantomData<&'a ()>,
}

unsafe impl<'a> Send for Context<'a> {}
unsafe impl<'a> Sync for Context<'a> {}

impl<'a> Context<'a> {
    pub fn new(inner: bindings::CUcontext) -> Self {
        Context {
            inner,
            module: PhantomData,
        }
    }

    pub fn handle(&self) -> ContextHandle<'a> {
        ContextHandle {
            inner: self.inner,
            module: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *const bindings::CUctx_st {
        self.inner as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut bindings::CUctx_st {
        self.inner as *mut _
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct FunctionHandle<'a> {
    inner: bindings::CUfunction,
    module: PhantomData<&'a ()>,
}

unsafe impl<'a> Send for FunctionHandle<'a> {}
unsafe impl<'a> Sync for FunctionHandle<'a> {}

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct Function<'a> {
    inner: bindings::CUfunction,
    ctx: PhantomData<Context<'a>>,
}

unsafe impl<'a> Send for Function<'a> {}
unsafe impl<'a> Sync for Function<'a> {}

impl<'a> Function<'a> {
    pub fn new(inner: bindings::CUfunction) -> Self {
        Function {
            inner,
            ctx: PhantomData,
        }
    }

    pub fn handle(&self) -> FunctionHandle<'a> {
        FunctionHandle {
            inner: self.inner,
            module: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *const bindings::CUfunc_st {
        self.inner as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut bindings::CUfunc_st {
        self.inner as *mut _
    }
}

/// CUDA function attribute
#[derive(Debug, Clone, Copy)]
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

impl From<FunctionAttribute> for bindings::CUfunction_attribute_enum {
    fn from(point: FunctionAttribute) -> bindings::CUfunction_attribute_enum {
        use bindings::CUfunction_attribute_enum as ATTR;
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
    pub fn num_registers(&mut self) -> CudaResult<i32> {
        self.get_attribute(FunctionAttribute::NumRegs)
    }

    pub fn shared_memory_bytes(&mut self) -> CudaResult<i32> {
        self.get_attribute(FunctionAttribute::SharedSizeBytes)
    }

    pub fn binary_version(&mut self) -> CudaResult<i32> {
        self.get_attribute(FunctionAttribute::BinaryVersion)
    }

    pub fn get_attribute(&mut self, attr: FunctionAttribute) -> CudaResult<i32> {
        let mut val = 0i32;
        let result = unsafe {
            bindings::cuFuncGetAttribute(&mut val as *mut _, attr.into(), self.as_mut_ptr())
        };
        result.into_result()?;
        Ok(val)
    }
}
