use super::model;

include!(concat!(env!("OUT_DIR"), "/bindings/nvbit.rs"));

impl From<model::FunctionAttribute> for CUfunction_attribute_enum {
    fn from(point: model::FunctionAttribute) -> CUfunction_attribute_enum {
        use model::FunctionAttribute;
        use CUfunction_attribute_enum as ATTR;

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
