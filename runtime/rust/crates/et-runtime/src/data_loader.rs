use et_core::error::{Error, Result};

pub struct FreeableBuffer {
    data: *const u8,
    size: usize,
    free_fn: Option<unsafe extern "C" fn(*mut u8, usize)>,
}

impl FreeableBuffer {
    pub fn new(
        data: *const u8,
        size: usize,
        free_fn: Option<unsafe extern "C" fn(*mut u8, usize)>,
    ) -> Self {
        FreeableBuffer {
            data,
            size,
            free_fn,
        }
    }

    pub fn empty() -> Self {
        FreeableBuffer {
            data: core::ptr::null(),
            size: 0,
            free_fn: None,
        }
    }

    pub fn data(&self) -> *const u8 {
        self.data
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.data.is_null() || self.size == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(self.data, self.size) }
    }
}

impl Drop for FreeableBuffer {
    fn drop(&mut self) {
        if let Some(free_fn) = self.free_fn {
            if !self.data.is_null() {
                unsafe {
                    free_fn(self.data as *mut u8, self.size);
                }
            }
        }
    }
}

pub trait DataLoader {
    fn load(&self, offset: usize, size: usize) -> Result<FreeableBuffer>;
    fn size(&self) -> Result<usize>;
}

pub struct BufferDataLoader<'a> {
    data: &'a [u8],
}

impl<'a> BufferDataLoader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BufferDataLoader { data }
    }
}

impl<'a> DataLoader for BufferDataLoader<'a> {
    fn load(&self, offset: usize, size: usize) -> Result<FreeableBuffer> {
        let end = offset.checked_add(size).ok_or(Error::InvalidArgument)?;
        if end > self.data.len() {
            return Err(Error::AccessFailed);
        }
        Ok(FreeableBuffer::new(
            unsafe { self.data.as_ptr().add(offset) },
            size,
            None,
        ))
    }

    fn size(&self) -> Result<usize> {
        Ok(self.data.len())
    }
}
