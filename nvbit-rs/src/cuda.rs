use nvbit_sys::{bindings, nvbit};
use std::marker::PhantomData;

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

    // pub fn inner(&self) -> bindings::CUcontext {
    //     self.inner
    // }

    pub fn as_ptr(&self) -> *const bindings::CUctx_st {
        self.inner as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut bindings::CUctx_st {
        self.inner as *mut _
    }
}

// impl<'a> Deref for Context<'a> {
//     type Target = CUcontext;

//     fn deref(&self) -> &Self::Target {
//         &self.inner
//     }
// }

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

    // pub fn inner(&self) -> bindings::CUfunction {
    //     self.inner
    // }

    pub fn as_ptr(&self) -> *const bindings::CUfunc_st {
        self.inner as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut bindings::CUfunc_st {
        self.inner as *mut _
    }
}
