pub fn alloc_box_buffer(len: usize) -> Box<[u8]> {
    if len == 0 {
        return <Box<[u8]>>::default();
    }
    let layout = std::alloc::Layout::array::<u8>(len).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    let slice_ptr = std::ptr::slice_from_raw_parts_mut(ptr, len);
    unsafe { Box::from_raw(slice_ptr) }
}

#[repr(C)]
pub struct Buffer {
    size: i64,
    data: *mut u8, // libc::c_void,
}

impl Buffer {
    #[inline]
    pub fn with_size(size: usize) -> Self {
        assert!(size < i64::MAX as usize);
        let mut buf = vec![];
        buf.reserve_exact(size);
        buf.resize(size, 0);
        Self::from_vec(buf)
    }

    #[inline]
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        use std::convert::TryFrom;
        let mut buf = bytes.into_boxed_slice();
        let data = buf.as_mut_ptr();
        let size = i64::try_from(buf.len()).expect("buffer length cannot fit into a i64.");
        std::mem::forget(buf);
        Self { data, size }
    }

    #[inline]
    fn len(&self) -> usize {
        use std::convert::TryInto;
        self.size
            .try_into()
            .expect("buffer length negative or overflowed")
    }
}

impl std::ops::Deref for Buffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        if self.data.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.data, self.len()) }
        }
    }
}

impl std::ops::DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.data.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.data, self.len()) }
        }
    }
}

impl From<Vec<u8>> for Buffer {
    #[inline]
    fn from(bytes: Vec<u8>) -> Self {
        Self::from_vec(bytes)
    }
}
