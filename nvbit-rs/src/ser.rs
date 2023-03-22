/// Trait for all types that wrap raw pointers.
pub trait AsPtr<T> {
    fn as_ptr(&self) -> *const T;
}

impl<T> AsPtr<T> for *mut T {
    fn as_ptr(&self) -> *const T {
        *self as _
    }
}

impl<T> AsPtr<T> for *const T {
    fn as_ptr(&self) -> *const T {
        self.cast()
    }
}

/// Serializes a types wrapping a pointer.
///
/// # Errors
/// If the value cannot be serialized.
pub fn to_raw_ptr<S, P, T>(x: &P, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
    P: AsPtr<T>,
{
    s.serialize_u64(x.as_ptr() as u64)
}
