pub mod nvbit;
pub mod utils;

#[allow(
    warnings,
    clippy::all,
    clippy::pedantic,
    clippy::restriction,
    clippy::nursery
)]
pub mod bindings;

pub use bindings::*;
pub use nvbit_model as model;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CudaError(pub model::CudaErrorKind);

impl From<model::CudaErrorKind> for CudaError {
    fn from(kind: model::CudaErrorKind) -> Self {
        Self(kind)
    }
}

impl From<bindings::cudaError_enum> for CudaError {
    fn from(err: bindings::cudaError_enum) -> Self {
        Self(err.into())
    }
}

impl From<CudaError> for bindings::cudaError_enum {
    fn from(err: CudaError) -> Self {
        err.0.into()
    }
}

pub type CudaResult<T> = Result<T, CudaError>;

pub trait IntoCudaResult {
    #[allow(clippy::missing_errors_doc)]
    fn into_result(self) -> CudaResult<()>;
}

impl IntoCudaResult for bindings::cudaError_enum {
    fn into_result(self) -> CudaResult<()> {
        let kind = model::CudaErrorKind::from(self);
        kind.into_result()
    }
}

impl IntoCudaResult for model::CudaErrorKind {
    fn into_result(self) -> CudaResult<()> {
        use model::CudaErrorKind;
        match self {
            CudaErrorKind::Success => Ok(()),
            other => Err(CudaError(other)),
        }
    }
}

// impl IntoCudaResult for bindings::cudaError_enum {
//     fn into_result(self) -> CudaResult<()> {
//         use bindings::cudaError_enum as ERR;
//         match self {
//         }
//     }
// }

impl std::error::Error for CudaError {}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut msg_ptr: *const std::ffi::c_char = std::ptr::null();
        let msg = unsafe {
            cuGetErrorString(
                Into::<cudaError_enum>::into(*self),
                std::ptr::addr_of_mut!(msg_ptr),
            )
            .into_result()
            .map_err(|_| std::fmt::Error)?;
            std::ffi::CStr::from_ptr(msg_ptr).to_str().unwrap()
        };
        write!(f, "{msg}")
        // write!(f, "todo")
    }
}
