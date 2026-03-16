/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*EtOpFunction)(void* ctx, void** args, size_t n_args);

// Look up a kernel by its fully qualified name (e.g. "aten::add.out").
// name is NOT required to be null-terminated; name_len gives the length.
// Returns NULL if the kernel is not found.
EtOpFunction et_kernel_lookup(const char* name, size_t name_len);

// Call a previously-looked-up kernel function pointer.
void et_kernel_call(
    EtOpFunction func,
    void* ctx,
    void** args,
    size_t n_args);

// Create a KernelRuntimeContext. temp_allocator may be NULL.
// The returned pointer must be freed with et_kernel_context_destroy.
void* et_kernel_context_new(void* temp_allocator);

// Destroy a KernelRuntimeContext created by et_kernel_context_new.
void et_kernel_context_destroy(void* ctx);

// Return the failure state of a KernelRuntimeContext.
// 0 means success (Error::Ok).
uint32_t et_kernel_context_failure_state(const void* ctx);

// Reset the failure state of a KernelRuntimeContext back to Ok.
void et_kernel_context_reset_failure(void* ctx);

// Layout validation helpers — checked at build time in Rust build.rs.
size_t et_sizeof_evalue(void);
size_t et_alignof_evalue(void);

#ifdef __cplusplus
} // extern "C"
#endif
