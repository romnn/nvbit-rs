use super::model;

include!(concat!(env!("OUT_DIR"), "/bindings/nvbit.rs"));

impl From<model::FunctionAttribute> for CUfunction_attribute_enum {
    fn from(point: model::FunctionAttribute) -> Self {
        use model::FunctionAttribute;

        match point {
            FunctionAttribute::MaxThreadsPerBlock => Self::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            FunctionAttribute::SharedSizeBytes => Self::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            FunctionAttribute::ConstSizeBytes => Self::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
            FunctionAttribute::LocalSizeBytes => Self::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            FunctionAttribute::NumRegs => Self::CU_FUNC_ATTRIBUTE_NUM_REGS,
            FunctionAttribute::PTXVersion => Self::CU_FUNC_ATTRIBUTE_PTX_VERSION,
            FunctionAttribute::BinaryVersion => Self::CU_FUNC_ATTRIBUTE_BINARY_VERSION,
            FunctionAttribute::CacheModeCA => Self::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
            FunctionAttribute::MaxDynamicSharedSizeBytes => {
                Self::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            }
            FunctionAttribute::PreferredSharedMemoryCarveout => {
                Self::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            }
            FunctionAttribute::Max => Self::CU_FUNC_ATTRIBUTE_MAX,
        }
    }
}

impl From<InstrType_RegModifierType> for model::RegisterModifier {
    fn from(modifier: InstrType_RegModifierType) -> Self {
        use InstrType_RegModifierType as RMT;
        match modifier {
            RMT::NO_MOD => Self::None,
            RMT::X1 => Self::X1,
            RMT::X4 => Self::X4,
            RMT::X8 => Self::X8,
            RMT::X16 => Self::X16,
            RMT::U32 => Self::U32,
            RMT::U64 => Self::U64,
        }
    }
}

impl From<InstrType_MemorySpace> for model::MemorySpace {
    #[inline]
    #[must_use]
    fn from(mem_space: InstrType_MemorySpace) -> Self {
        use InstrType_MemorySpace as MS;

        match mem_space {
            MS::NONE => Self::None,
            MS::LOCAL => Self::Local,
            MS::GENERIC => Self::Generic,
            MS::GLOBAL => Self::Global,
            MS::SHARED => Self::Shared,
            MS::CONSTANT => Self::Constant,
            MS::GLOBAL_TO_SHARED => Self::GlobalToShared,
            MS::SURFACE => Self::Surface,
            MS::TEXTURE => Self::Texture,
        }
    }
}

impl From<model::InsertionPoint> for ipoint_t {
    #[inline]
    #[must_use]
    fn from(point: model::InsertionPoint) -> Self {
        use model::InsertionPoint;
        match point {
            InsertionPoint::Before => Self::IPOINT_BEFORE,
            InsertionPoint::After => Self::IPOINT_AFTER,
        }
    }
}

impl From<ipoint_t> for model::InsertionPoint {
    #[inline]
    #[must_use]
    fn from(point: ipoint_t) -> Self {
        match point {
            ipoint_t::IPOINT_BEFORE => Self::Before,
            ipoint_t::IPOINT_AFTER => Self::After,
        }
    }
}

impl From<model::CudaErrorKind> for cudaError_enum {
    fn from(err: model::CudaErrorKind) -> Self {
        use model::CudaErrorKind;
        match err {
            CudaErrorKind::Success => Self::CUDA_SUCCESS,
            CudaErrorKind::InvalidValue => Self::CUDA_ERROR_INVALID_VALUE,
            CudaErrorKind::OutOfMemory => Self::CUDA_ERROR_OUT_OF_MEMORY,
            CudaErrorKind::NotInitialized => Self::CUDA_ERROR_NOT_INITIALIZED,
            CudaErrorKind::Deinitialized => Self::CUDA_ERROR_DEINITIALIZED,
            CudaErrorKind::ProfilerDisabled => Self::CUDA_ERROR_PROFILER_DISABLED,
            CudaErrorKind::ProfilerNotInitialized => Self::CUDA_ERROR_PROFILER_NOT_INITIALIZED,
            CudaErrorKind::ProfilerAlreadyStarted => Self::CUDA_ERROR_PROFILER_ALREADY_STARTED,
            CudaErrorKind::ProfilerAlreadyStopped => Self::CUDA_ERROR_PROFILER_ALREADY_STOPPED,
            CudaErrorKind::NoDevice => Self::CUDA_ERROR_NO_DEVICE,
            CudaErrorKind::InvalidDevice => Self::CUDA_ERROR_INVALID_DEVICE,
            CudaErrorKind::InvalidImage => Self::CUDA_ERROR_INVALID_IMAGE,
            CudaErrorKind::InvalidContext => Self::CUDA_ERROR_INVALID_CONTEXT,
            CudaErrorKind::ContextAlreadyCurrent => Self::CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
            CudaErrorKind::MapFailed => Self::CUDA_ERROR_MAP_FAILED,
            CudaErrorKind::UnmapFailed => Self::CUDA_ERROR_UNMAP_FAILED,
            CudaErrorKind::ArrayIsMapped => Self::CUDA_ERROR_ARRAY_IS_MAPPED,
            CudaErrorKind::AlreadyMapped => Self::CUDA_ERROR_ALREADY_MAPPED,
            CudaErrorKind::NoBinaryForGpu => Self::CUDA_ERROR_NO_BINARY_FOR_GPU,
            CudaErrorKind::AlreadyAcquired => Self::CUDA_ERROR_ALREADY_ACQUIRED,
            CudaErrorKind::NotMapped => Self::CUDA_ERROR_NOT_MAPPED,
            CudaErrorKind::NotMappedAsArray => Self::CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
            CudaErrorKind::NotMappedAsPointer => Self::CUDA_ERROR_NOT_MAPPED_AS_POINTER,
            CudaErrorKind::EccUncorrectable => Self::CUDA_ERROR_ECC_UNCORRECTABLE,
            CudaErrorKind::UnsupportedLimit => Self::CUDA_ERROR_UNSUPPORTED_LIMIT,
            CudaErrorKind::ContextAlreadyInUse => Self::CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
            CudaErrorKind::PeerAccessUnsupported => Self::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
            CudaErrorKind::InvalidPtx => Self::CUDA_ERROR_INVALID_PTX,
            CudaErrorKind::InvalidGraphicsContext => Self::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
            CudaErrorKind::NvlinkUncorrectable => Self::CUDA_ERROR_NVLINK_UNCORRECTABLE,
            CudaErrorKind::InvalidSouce => Self::CUDA_ERROR_INVALID_SOURCE,
            CudaErrorKind::FileNotFound => Self::CUDA_ERROR_FILE_NOT_FOUND,
            CudaErrorKind::SharedObjectSymbolNotFound => {
                Self::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND
            }
            CudaErrorKind::SharedObjectInitFailed => Self::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
            CudaErrorKind::OperatingSystemError => Self::CUDA_ERROR_OPERATING_SYSTEM,
            CudaErrorKind::InvalidHandle => Self::CUDA_ERROR_INVALID_HANDLE,
            CudaErrorKind::NotFound => Self::CUDA_ERROR_NOT_FOUND,
            CudaErrorKind::NotReady => Self::CUDA_ERROR_NOT_READY,
            CudaErrorKind::IllegalAddress => Self::CUDA_ERROR_ILLEGAL_ADDRESS,
            CudaErrorKind::LaunchOutOfResources => Self::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
            CudaErrorKind::LaunchTimeout => Self::CUDA_ERROR_LAUNCH_TIMEOUT,
            CudaErrorKind::LaunchIncompatibleTexturing => {
                Self::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING
            }
            CudaErrorKind::PeerAccessAlreadyEnabled => Self::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
            CudaErrorKind::PeerAccessNotEnabled => Self::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
            CudaErrorKind::PrimaryContextActive => Self::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
            CudaErrorKind::ContextIsDestroyed => Self::CUDA_ERROR_CONTEXT_IS_DESTROYED,
            CudaErrorKind::AssertError => Self::CUDA_ERROR_ASSERT,
            CudaErrorKind::TooManyPeers => Self::CUDA_ERROR_TOO_MANY_PEERS,
            CudaErrorKind::HostMemoryAlreadyRegistered => {
                Self::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED
            }
            CudaErrorKind::HostMemoryNotRegistered => Self::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
            CudaErrorKind::HardwareStackError => Self::CUDA_ERROR_HARDWARE_STACK_ERROR,
            CudaErrorKind::IllegalInstruction => Self::CUDA_ERROR_ILLEGAL_INSTRUCTION,
            CudaErrorKind::MisalignedAddress => Self::CUDA_ERROR_MISALIGNED_ADDRESS,
            CudaErrorKind::InvalidAddressSpace => Self::CUDA_ERROR_INVALID_ADDRESS_SPACE,
            CudaErrorKind::InvalidProgramCounter => Self::CUDA_ERROR_INVALID_PC,
            CudaErrorKind::LaunchFailed => Self::CUDA_ERROR_LAUNCH_FAILED,
            CudaErrorKind::NotPermitted => Self::CUDA_ERROR_NOT_PERMITTED,
            CudaErrorKind::NotSupported => Self::CUDA_ERROR_NOT_SUPPORTED,
            CudaErrorKind::Unknown => unreachable!("unknown cuda error"),
        }
    }
}

impl From<cudaError_enum> for model::CudaErrorKind {
    fn from(err: cudaError_enum) -> Self {
        use cudaError_enum as ERR;
        match err {
            ERR::CUDA_SUCCESS => Self::Success,
            ERR::CUDA_ERROR_INVALID_VALUE => Self::InvalidValue,
            ERR::CUDA_ERROR_OUT_OF_MEMORY => Self::OutOfMemory,
            ERR::CUDA_ERROR_NOT_INITIALIZED => Self::NotInitialized,
            ERR::CUDA_ERROR_DEINITIALIZED => Self::Deinitialized,
            ERR::CUDA_ERROR_PROFILER_DISABLED => Self::ProfilerDisabled,
            ERR::CUDA_ERROR_PROFILER_NOT_INITIALIZED => Self::ProfilerNotInitialized,
            ERR::CUDA_ERROR_PROFILER_ALREADY_STARTED => Self::ProfilerAlreadyStarted,
            ERR::CUDA_ERROR_PROFILER_ALREADY_STOPPED => Self::ProfilerAlreadyStopped,
            ERR::CUDA_ERROR_NO_DEVICE => Self::NoDevice,
            ERR::CUDA_ERROR_INVALID_DEVICE => Self::InvalidDevice,
            ERR::CUDA_ERROR_INVALID_IMAGE => Self::InvalidImage,
            ERR::CUDA_ERROR_INVALID_CONTEXT => Self::InvalidContext,
            ERR::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => Self::ContextAlreadyCurrent,
            ERR::CUDA_ERROR_MAP_FAILED => Self::MapFailed,
            ERR::CUDA_ERROR_UNMAP_FAILED => Self::UnmapFailed,
            ERR::CUDA_ERROR_ARRAY_IS_MAPPED => Self::ArrayIsMapped,
            ERR::CUDA_ERROR_ALREADY_MAPPED => Self::AlreadyMapped,
            ERR::CUDA_ERROR_NO_BINARY_FOR_GPU => Self::NoBinaryForGpu,
            ERR::CUDA_ERROR_ALREADY_ACQUIRED => Self::AlreadyAcquired,
            ERR::CUDA_ERROR_NOT_MAPPED => Self::NotMapped,
            ERR::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Self::NotMappedAsArray,
            ERR::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Self::NotMappedAsPointer,
            ERR::CUDA_ERROR_ECC_UNCORRECTABLE => Self::EccUncorrectable,
            ERR::CUDA_ERROR_UNSUPPORTED_LIMIT => Self::UnsupportedLimit,
            ERR::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => Self::ContextAlreadyInUse,
            ERR::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => Self::PeerAccessUnsupported,
            ERR::CUDA_ERROR_INVALID_PTX => Self::InvalidPtx,
            ERR::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => Self::InvalidGraphicsContext,
            ERR::CUDA_ERROR_NVLINK_UNCORRECTABLE => Self::NvlinkUncorrectable,
            ERR::CUDA_ERROR_INVALID_SOURCE => Self::InvalidSouce,
            ERR::CUDA_ERROR_FILE_NOT_FOUND => Self::FileNotFound,
            ERR::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => Self::SharedObjectSymbolNotFound,

            ERR::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => Self::SharedObjectInitFailed,
            ERR::CUDA_ERROR_OPERATING_SYSTEM => Self::OperatingSystemError,
            ERR::CUDA_ERROR_INVALID_HANDLE => Self::InvalidHandle,
            ERR::CUDA_ERROR_NOT_FOUND => Self::NotFound,
            ERR::CUDA_ERROR_NOT_READY => Self::NotReady,
            ERR::CUDA_ERROR_ILLEGAL_ADDRESS => Self::IllegalAddress,
            ERR::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => Self::LaunchOutOfResources,
            ERR::CUDA_ERROR_LAUNCH_TIMEOUT => Self::LaunchTimeout,
            ERR::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => Self::LaunchIncompatibleTexturing,
            ERR::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Self::PeerAccessAlreadyEnabled,
            ERR::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => Self::PeerAccessNotEnabled,
            ERR::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => Self::PrimaryContextActive,
            ERR::CUDA_ERROR_CONTEXT_IS_DESTROYED => Self::ContextIsDestroyed,
            ERR::CUDA_ERROR_ASSERT => Self::AssertError,
            ERR::CUDA_ERROR_TOO_MANY_PEERS => Self::TooManyPeers,
            ERR::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => Self::HostMemoryAlreadyRegistered,
            ERR::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => Self::HostMemoryNotRegistered,
            ERR::CUDA_ERROR_HARDWARE_STACK_ERROR => Self::HardwareStackError,
            ERR::CUDA_ERROR_ILLEGAL_INSTRUCTION => Self::IllegalInstruction,
            ERR::CUDA_ERROR_MISALIGNED_ADDRESS => Self::MisalignedAddress,
            ERR::CUDA_ERROR_INVALID_ADDRESS_SPACE => Self::InvalidAddressSpace,
            ERR::CUDA_ERROR_INVALID_PC => Self::InvalidProgramCounter,
            ERR::CUDA_ERROR_LAUNCH_FAILED => Self::LaunchFailed,
            ERR::CUDA_ERROR_NOT_PERMITTED => Self::NotPermitted,
            ERR::CUDA_ERROR_NOT_SUPPORTED => Self::NotSupported,
            _ => Self::Unknown,
        }
    }
}
