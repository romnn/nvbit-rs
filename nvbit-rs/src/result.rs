use nvbit_sys::bindings;
use std::{ffi, fmt};

#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaError {
    InvalidValue,
    OutOfMemory,
    NotInitialized,
    Deinitialized,
    ProfilerDisabled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    NoDevice,
    InvalidDevice,
    InvalidImage,
    InvalidContext,
    ContextAlreadyCurrent,
    MapFailed,
    UnmapFailed,
    ArrayIsMapped,
    AlreadyMapped,
    NoBinaryForGpu,
    AlreadyAcquired,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    EccUncorrectable,
    UnsupportedLimit,
    ContextAlreadyInUse,
    PeerAccessUnsupported,
    InvalidPtx,
    InvalidGraphicsContext,
    NvlinkUncorrectable,
    InvalidSouce,
    FileNotFound,
    SharedObjectSymbolNotFound,
    SharedObjectInitFailed,
    OperatingSystemError,
    InvalidHandle,
    NotFound,
    NotReady,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchIncompatibleTexturing,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    PrimaryContextActive,
    ContextIsDestroyed,
    AssertError,
    TooManyPeers,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    HardwareStackError,
    IllegalInstruction,
    MisalignedAddress,
    InvalidAddressSpace,
    InvalidProgramCounter,
    LaunchFailed,
    NotPermitted,
    NotSupported,
    Unknown,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr: *const ffi::c_char = std::ptr::null();
        let msg = unsafe {
            bindings::cuGetErrorString(
                Into::<bindings::cudaError_enum>::into(*self),
                &mut ptr as *mut *const ffi::c_char,
            )
            .into_result()
            .map_err(|_| fmt::Error)?;
            ffi::CStr::from_ptr(ptr).to_str().unwrap()
        };
        write!(f, "{}", msg)
    }
}

impl std::error::Error for CudaError {}

pub type CudaResult<T> = Result<T, CudaError>;

pub(crate) trait IntoCudaResult {
    fn into_result(self) -> CudaResult<()>;
}

impl Into<bindings::cudaError_enum> for CudaError {
    fn into(self: CudaError) -> bindings::cudaError_enum {
        use bindings::cudaError_enum as ERR;
        match self {
            CudaError::InvalidValue => ERR::CUDA_ERROR_INVALID_VALUE,
            CudaError::OutOfMemory => ERR::CUDA_ERROR_OUT_OF_MEMORY,
            CudaError::NotInitialized => ERR::CUDA_ERROR_NOT_INITIALIZED,
            CudaError::Deinitialized => ERR::CUDA_ERROR_DEINITIALIZED,
            CudaError::ProfilerDisabled => ERR::CUDA_ERROR_PROFILER_DISABLED,
            CudaError::ProfilerNotInitialized => ERR::CUDA_ERROR_PROFILER_NOT_INITIALIZED,
            CudaError::ProfilerAlreadyStarted => ERR::CUDA_ERROR_PROFILER_ALREADY_STARTED,
            CudaError::ProfilerAlreadyStopped => ERR::CUDA_ERROR_PROFILER_ALREADY_STOPPED,
            CudaError::NoDevice => ERR::CUDA_ERROR_NO_DEVICE,
            CudaError::InvalidDevice => ERR::CUDA_ERROR_INVALID_DEVICE,
            CudaError::InvalidImage => ERR::CUDA_ERROR_INVALID_IMAGE,
            CudaError::InvalidContext => ERR::CUDA_ERROR_INVALID_CONTEXT,
            CudaError::ContextAlreadyCurrent => ERR::CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
            CudaError::MapFailed => ERR::CUDA_ERROR_MAP_FAILED,
            CudaError::UnmapFailed => ERR::CUDA_ERROR_UNMAP_FAILED,
            CudaError::ArrayIsMapped => ERR::CUDA_ERROR_ARRAY_IS_MAPPED,
            CudaError::AlreadyMapped => ERR::CUDA_ERROR_ALREADY_MAPPED,
            CudaError::NoBinaryForGpu => ERR::CUDA_ERROR_NO_BINARY_FOR_GPU,
            CudaError::AlreadyAcquired => ERR::CUDA_ERROR_ALREADY_ACQUIRED,
            CudaError::NotMapped => ERR::CUDA_ERROR_NOT_MAPPED,
            CudaError::NotMappedAsArray => ERR::CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
            CudaError::NotMappedAsPointer => ERR::CUDA_ERROR_NOT_MAPPED_AS_POINTER,
            CudaError::EccUncorrectable => ERR::CUDA_ERROR_ECC_UNCORRECTABLE,
            CudaError::UnsupportedLimit => ERR::CUDA_ERROR_UNSUPPORTED_LIMIT,
            CudaError::ContextAlreadyInUse => ERR::CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
            CudaError::PeerAccessUnsupported => ERR::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
            CudaError::InvalidPtx => ERR::CUDA_ERROR_INVALID_PTX,
            CudaError::InvalidGraphicsContext => ERR::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
            CudaError::NvlinkUncorrectable => ERR::CUDA_ERROR_NVLINK_UNCORRECTABLE,
            CudaError::InvalidSouce => ERR::CUDA_ERROR_INVALID_SOURCE,
            CudaError::FileNotFound => ERR::CUDA_ERROR_FILE_NOT_FOUND,
            CudaError::SharedObjectSymbolNotFound => ERR::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
            CudaError::SharedObjectInitFailed => ERR::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
            CudaError::OperatingSystemError => ERR::CUDA_ERROR_OPERATING_SYSTEM,
            CudaError::InvalidHandle => ERR::CUDA_ERROR_INVALID_HANDLE,
            CudaError::NotFound => ERR::CUDA_ERROR_NOT_FOUND,
            CudaError::NotReady => ERR::CUDA_ERROR_NOT_READY,
            CudaError::IllegalAddress => ERR::CUDA_ERROR_ILLEGAL_ADDRESS,
            CudaError::LaunchOutOfResources => ERR::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
            CudaError::LaunchTimeout => ERR::CUDA_ERROR_LAUNCH_TIMEOUT,
            CudaError::LaunchIncompatibleTexturing => ERR::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
            CudaError::PeerAccessAlreadyEnabled => ERR::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
            CudaError::PeerAccessNotEnabled => ERR::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
            CudaError::PrimaryContextActive => ERR::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
            CudaError::ContextIsDestroyed => ERR::CUDA_ERROR_CONTEXT_IS_DESTROYED,
            CudaError::AssertError => ERR::CUDA_ERROR_ASSERT,
            CudaError::TooManyPeers => ERR::CUDA_ERROR_TOO_MANY_PEERS,
            CudaError::HostMemoryAlreadyRegistered => {
                ERR::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED
            }
            CudaError::HostMemoryNotRegistered => ERR::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
            CudaError::HardwareStackError => ERR::CUDA_ERROR_HARDWARE_STACK_ERROR,
            CudaError::IllegalInstruction => ERR::CUDA_ERROR_ILLEGAL_INSTRUCTION,
            CudaError::MisalignedAddress => ERR::CUDA_ERROR_MISALIGNED_ADDRESS,
            CudaError::InvalidAddressSpace => ERR::CUDA_ERROR_INVALID_ADDRESS_SPACE,
            CudaError::InvalidProgramCounter => ERR::CUDA_ERROR_INVALID_PC,
            CudaError::LaunchFailed => ERR::CUDA_ERROR_LAUNCH_FAILED,
            CudaError::NotPermitted => ERR::CUDA_ERROR_NOT_PERMITTED,
            CudaError::NotSupported => ERR::CUDA_ERROR_NOT_SUPPORTED,
            CudaError::Unknown => unreachable!("unknown cuda error"),
        }
    }
}

impl IntoCudaResult for bindings::cudaError_enum {
    fn into_result(self) -> CudaResult<()> {
        use bindings::cudaError_enum as ERR;
        match self {
            ERR::CUDA_SUCCESS => Ok(()),
            ERR::CUDA_ERROR_INVALID_VALUE => Err(CudaError::InvalidValue),
            ERR::CUDA_ERROR_OUT_OF_MEMORY => Err(CudaError::OutOfMemory),
            ERR::CUDA_ERROR_NOT_INITIALIZED => Err(CudaError::NotInitialized),
            ERR::CUDA_ERROR_DEINITIALIZED => Err(CudaError::Deinitialized),
            ERR::CUDA_ERROR_PROFILER_DISABLED => Err(CudaError::ProfilerDisabled),
            ERR::CUDA_ERROR_PROFILER_NOT_INITIALIZED => Err(CudaError::ProfilerNotInitialized),
            ERR::CUDA_ERROR_PROFILER_ALREADY_STARTED => Err(CudaError::ProfilerAlreadyStarted),
            ERR::CUDA_ERROR_PROFILER_ALREADY_STOPPED => Err(CudaError::ProfilerAlreadyStopped),
            ERR::CUDA_ERROR_NO_DEVICE => Err(CudaError::NoDevice),
            ERR::CUDA_ERROR_INVALID_DEVICE => Err(CudaError::InvalidDevice),
            ERR::CUDA_ERROR_INVALID_IMAGE => Err(CudaError::InvalidImage),
            ERR::CUDA_ERROR_INVALID_CONTEXT => Err(CudaError::InvalidContext),
            ERR::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => Err(CudaError::ContextAlreadyCurrent),
            ERR::CUDA_ERROR_MAP_FAILED => Err(CudaError::MapFailed),
            ERR::CUDA_ERROR_UNMAP_FAILED => Err(CudaError::UnmapFailed),
            ERR::CUDA_ERROR_ARRAY_IS_MAPPED => Err(CudaError::ArrayIsMapped),
            ERR::CUDA_ERROR_ALREADY_MAPPED => Err(CudaError::AlreadyMapped),
            ERR::CUDA_ERROR_NO_BINARY_FOR_GPU => Err(CudaError::NoBinaryForGpu),
            ERR::CUDA_ERROR_ALREADY_ACQUIRED => Err(CudaError::AlreadyAcquired),
            ERR::CUDA_ERROR_NOT_MAPPED => Err(CudaError::NotMapped),
            ERR::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Err(CudaError::NotMappedAsArray),
            ERR::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Err(CudaError::NotMappedAsPointer),
            ERR::CUDA_ERROR_ECC_UNCORRECTABLE => Err(CudaError::EccUncorrectable),
            ERR::CUDA_ERROR_UNSUPPORTED_LIMIT => Err(CudaError::UnsupportedLimit),
            ERR::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => Err(CudaError::ContextAlreadyInUse),
            ERR::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => Err(CudaError::PeerAccessUnsupported),
            ERR::CUDA_ERROR_INVALID_PTX => Err(CudaError::InvalidPtx),
            ERR::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => Err(CudaError::InvalidGraphicsContext),
            ERR::CUDA_ERROR_NVLINK_UNCORRECTABLE => Err(CudaError::NvlinkUncorrectable),
            ERR::CUDA_ERROR_INVALID_SOURCE => Err(CudaError::InvalidSouce),
            ERR::CUDA_ERROR_FILE_NOT_FOUND => Err(CudaError::FileNotFound),
            ERR::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
                Err(CudaError::SharedObjectSymbolNotFound)
            }
            ERR::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => Err(CudaError::SharedObjectInitFailed),
            ERR::CUDA_ERROR_OPERATING_SYSTEM => Err(CudaError::OperatingSystemError),
            ERR::CUDA_ERROR_INVALID_HANDLE => Err(CudaError::InvalidHandle),
            ERR::CUDA_ERROR_NOT_FOUND => Err(CudaError::NotFound),
            ERR::CUDA_ERROR_NOT_READY => Err(CudaError::NotReady),
            ERR::CUDA_ERROR_ILLEGAL_ADDRESS => Err(CudaError::IllegalAddress),
            ERR::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => Err(CudaError::LaunchOutOfResources),
            ERR::CUDA_ERROR_LAUNCH_TIMEOUT => Err(CudaError::LaunchTimeout),
            ERR::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
                Err(CudaError::LaunchIncompatibleTexturing)
            }
            ERR::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Err(CudaError::PeerAccessAlreadyEnabled),
            ERR::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => Err(CudaError::PeerAccessNotEnabled),
            ERR::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => Err(CudaError::PrimaryContextActive),
            ERR::CUDA_ERROR_CONTEXT_IS_DESTROYED => Err(CudaError::ContextIsDestroyed),
            ERR::CUDA_ERROR_ASSERT => Err(CudaError::AssertError),
            ERR::CUDA_ERROR_TOO_MANY_PEERS => Err(CudaError::TooManyPeers),
            ERR::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
                Err(CudaError::HostMemoryAlreadyRegistered)
            }
            ERR::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => Err(CudaError::HostMemoryNotRegistered),
            ERR::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(CudaError::HardwareStackError),
            ERR::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(CudaError::IllegalInstruction),
            ERR::CUDA_ERROR_MISALIGNED_ADDRESS => Err(CudaError::MisalignedAddress),
            ERR::CUDA_ERROR_INVALID_ADDRESS_SPACE => Err(CudaError::InvalidAddressSpace),
            ERR::CUDA_ERROR_INVALID_PC => Err(CudaError::InvalidProgramCounter),
            ERR::CUDA_ERROR_LAUNCH_FAILED => Err(CudaError::LaunchFailed),
            ERR::CUDA_ERROR_NOT_PERMITTED => Err(CudaError::NotPermitted),
            ERR::CUDA_ERROR_NOT_SUPPORTED => Err(CudaError::NotSupported),
            _ => Err(CudaError::Unknown),
        }
    }
}
