#![no_std]

extern "C" {
    pub fn et_backend_lookup(name: *const u8, name_len: usize) -> *mut u8;
    pub fn et_backend_is_available(backend: *mut u8) -> i32;

    pub fn et_backend_init(
        backend: *mut u8,
        allocator: *mut u8,
        data: *const u8,
        size: usize,
        out_handle: *mut *mut u8,
    ) -> u32;

    pub fn et_backend_execute(
        backend: *mut u8,
        handle: *mut u8,
        args: *mut *mut u8,
        n_args: usize,
        temp_alloc: *mut u8,
    ) -> u32;

    pub fn et_backend_destroy(backend: *mut u8, handle: *mut u8);
}
