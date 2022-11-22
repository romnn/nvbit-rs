/// Errors that can occur when creating a new buffer.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Layout(#[from] std::alloc::LayoutError),
}

/// A contiguous, linear memory buffer on the heap.
#[repr(C)]
pub struct Buffer(Box<[u8]>);

impl Buffer {
    /// Allocates a new contiguous, linear memory buffer on the heap.
    ///
    /// # Errors
    /// If the memory layout is not supported, an error is returned.
    #[inline]
    pub fn new(size: usize) -> Result<Self, Error> {
        if size == 0 {
            return Ok(Self(Box::<[u8]>::default()));
        }
        let layout = std::alloc::Layout::array::<u8>(size)?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let slice_ptr = std::ptr::slice_from_raw_parts_mut(ptr, size);
        Ok(Self(unsafe { Box::from_raw(slice_ptr) }))
    }
}

impl std::ops::Deref for Buffer {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Buffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
