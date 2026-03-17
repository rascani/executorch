// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#![no_std]

pub type OpFunction = Option<unsafe extern "C" fn(ctx: *mut u8, args: *mut *mut u8, n_args: usize)>;

extern "C" {
    pub fn et_kernel_lookup(name: *const u8, name_len: usize) -> OpFunction;

    pub fn et_kernel_call(
        func: unsafe extern "C" fn(ctx: *mut u8, args: *mut *mut u8, n_args: usize),
        ctx: *mut u8,
        args: *mut *mut u8,
        n_args: usize,
    );

    pub fn et_kernel_context_new(temp_buf: *mut u8, temp_size: u32) -> *mut u8;
    pub fn et_kernel_context_destroy(ctx: *mut u8);
    pub fn et_kernel_context_failure_state(ctx: *const u8) -> u32;
    pub fn et_kernel_context_reset_temp(ctx: *mut u8);

    pub fn et_sizeof_evalue() -> usize;
    pub fn et_alignof_evalue() -> usize;
}
