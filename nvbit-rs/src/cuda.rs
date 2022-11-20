use nvbit_sys::{bindings, nvbit};
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
