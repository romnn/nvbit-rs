#[repr(C)]
pub struct Buffer(Box<[u8]>);

impl Buffer {
    #[inline]
    pub fn new(size: usize) -> Self {
        if size == 0 {
            return Self(Box::<[u8]>::default());
        }
        assert!(size < i64::MAX as usize);
        let layout = std::alloc::Layout::array::<u8>(size).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let slice_ptr = std::ptr::slice_from_raw_parts_mut(ptr, size);
        Self(unsafe { Box::from_raw(slice_ptr) })
    }
}

impl std::ops::Deref for Buffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
