pub mod dim;
pub use dim::*;

use serde::{Deserialize, Serialize};

/// A CUDA device.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Device(pub u64);

/// A CUDA context.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Context(pub u64);

/// A CUDA function.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Function(pub u64);

/// A CUDA stream.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Stream(pub u64);

/// CUDA function attribute.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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

/// NVBIT Register modifiers.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegisterModifier {
    None = 0,
    X1 = 1,
    X4 = 2,
    X8 = 3,
    X16 = 4,
    U32 = 5,
    U64 = 6,
}

/// An instruction operand.
#[derive(Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum OperandKind {
    ImmutableUint64 {
        value: u64,
    },
    ImmutableDouble {
        value: f64,
    },
    Register {
        num: i32,
        prop: String,
    },
    Predicate {
        num: i32,
    },
    // URegister {},
    // UPredicate {},
    CBank {
        id: i32,
        has_imm_offset: bool,
        imm_offset: i32,
        has_reg_offset: bool,
        reg_offset: i32,
    },
    MemRef {
        has_ra: bool,
        ra_num: i32,
        ra_mod: RegisterModifier,
        has_ur: bool,
        ur_num: i32,
        ur_mod: RegisterModifier,
        has_imm: bool,
        imm: i32,
    },
    Generic {
        array: String,
    },
}

/// Identifier of GPU memory space.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemorySpace {
    None = 0,
    Local = 1,
    Generic = 2,
    Global = 3,
    Shared = 4,
    Constant = 5,
    GlobalToShared = 6,
    Surface = 7,
    Texture = 8,
}

/// An instruction operand predicate.
#[derive(
    Debug, Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Predicate {
    /// predicate number
    pub num: std::ffi::c_int,
    /// whether predicate is negated (i.e. @!P0).
    pub is_neg: bool,
    /// whether predicate is uniform predicate (e.g., @UP0).
    pub is_uniform: bool,
}

/// Insertion point where the instrumentation for an instruction should be inserted
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InsertionPoint {
    Before,
    After,
}

/// All possible CUDA error kinds
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CudaErrorKind {
    Success,
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
