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

// Look up a registered backend by name.
// Returns an opaque pointer to the BackendInterface, or NULL if not found.
void* et_backend_lookup(const char* name, size_t name_len);

// Check if a backend is available.
// Returns 1 if available, 0 if not.
int et_backend_is_available(void* backend);

// Initialize a backend delegate.
// processed_data/processed_size point to the preprocessed blob.
// runtime_allocator is a MemoryAllocator* (may be NULL).
// On success, writes an opaque DelegateHandle to *out_handle and returns 0.
// On failure, returns a non-zero error code.
uint32_t et_backend_init(
    void* backend,
    void* runtime_allocator,
    const void* processed_data,
    size_t processed_size,
    void** out_handle);

// Execute a backend delegate.
// args is an array of EValue* pointers; n_args is its length.
// temp_allocator is a MemoryAllocator* (may be NULL).
// Returns 0 on success, non-zero error code on failure.
uint32_t et_backend_execute(
    void* backend,
    void* handle,
    void** args,
    size_t n_args,
    void* temp_allocator);

// Destroy a backend delegate handle.
void et_backend_destroy(void* backend, void* handle);

#ifdef __cplusplus
} // extern "C"
#endif
