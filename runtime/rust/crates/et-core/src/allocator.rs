use crate::error::{Error, Result};
use core::ptr;

pub struct BumpAllocator {
    begin: *mut u8,
    end: *mut u8,
    cur: *mut u8,
    size: u32,
}

impl BumpAllocator {
    pub fn new(buf: &mut [u8]) -> Self {
        let size = buf.len() as u32;
        let begin = buf.as_mut_ptr();
        let end = unsafe { begin.add(buf.len()) };
        BumpAllocator {
            begin,
            end,
            cur: begin,
            size,
        }
    }

    pub fn allocate(&mut self, size: usize, align: usize) -> *mut u8 {
        if self.begin.is_null() || self.end.is_null() {
            return ptr::null_mut();
        }
        if !align.is_power_of_two() {
            return ptr::null_mut();
        }

        let cur_addr = self.cur as usize;
        let aligned_addr = (cur_addr + align - 1) & !(align - 1);
        let padding = aligned_addr - cur_addr;
        let available = self.end as usize - cur_addr;

        if padding > available || size > available - padding {
            return ptr::null_mut();
        }

        let start = aligned_addr as *mut u8;
        self.cur = unsafe { start.add(size) };
        start
    }

    pub fn allocate_instance<T>(&mut self) -> *mut T {
        self.allocate(core::mem::size_of::<T>(), core::mem::align_of::<T>()) as *mut T
    }

    pub fn allocate_slice<T>(&mut self, count: usize) -> *mut T {
        let byte_size = count.checked_mul(core::mem::size_of::<T>());
        match byte_size {
            Some(s) => self.allocate(s, core::mem::align_of::<T>()) as *mut T,
            None => ptr::null_mut(),
        }
    }

    pub fn reset(&mut self) {
        self.cur = self.begin;
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn used(&self) -> usize {
        self.cur as usize - self.begin as usize
    }
}

pub struct HierarchicalAllocator<'a> {
    buffers: &'a mut [&'a mut [u8]],
}

impl<'a> HierarchicalAllocator<'a> {
    pub fn new(buffers: &'a mut [&'a mut [u8]]) -> Self {
        HierarchicalAllocator { buffers }
    }

    pub fn get_offset_address(
        &mut self,
        memory_id: u32,
        offset: usize,
        size: usize,
    ) -> Result<*mut u8> {
        let total = offset.checked_add(size).ok_or(Error::InvalidArgument)?;
        let id = memory_id as usize;
        if id >= self.buffers.len() {
            return Err(Error::InvalidArgument);
        }
        if total > self.buffers[id].len() {
            return Err(Error::MemoryAllocationFailed);
        }
        Ok(unsafe { self.buffers[id].as_mut_ptr().add(offset) })
    }

    pub fn num_buffers(&self) -> usize {
        self.buffers.len()
    }
}

pub struct MemoryManager<'a> {
    pub method_allocator: &'a mut BumpAllocator,
    pub planned_memory: Option<&'a mut HierarchicalAllocator<'a>>,
    pub temp_allocator: Option<&'a mut BumpAllocator>,
}

impl<'a> MemoryManager<'a> {
    pub fn new(
        method_allocator: &'a mut BumpAllocator,
        planned_memory: Option<&'a mut HierarchicalAllocator<'a>>,
        temp_allocator: Option<&'a mut BumpAllocator>,
    ) -> Self {
        MemoryManager {
            method_allocator,
            planned_memory,
            temp_allocator,
        }
    }
}
