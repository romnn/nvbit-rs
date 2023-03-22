use serde::{Deserialize, Serialize};

/// A CUDA device.
#[derive(PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Device(pub u64);

/// A CUDA context.
#[derive(PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Context(pub u64);

/// A CUDA function.
#[derive(PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Function(pub u64);

/// A CUDA stream.
#[derive(PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Stream(pub u64);

/// CUDA function attribute.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
