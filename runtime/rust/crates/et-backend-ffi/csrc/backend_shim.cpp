/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "backend_shim.h"

#include <executorch/runtime/backend/backend_execution_context.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/memory_allocator.h>

#include <cstring>

using namespace executorch::runtime;

namespace {

class TrackingAllocator : public MemoryAllocator {
 public:
  using MemoryAllocator::MemoryAllocator;

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    void* result = MemoryAllocator::allocate(size, alignment);
    if (result) {
      uint8_t* end = static_cast<uint8_t*>(result) + size;
      size_t used = end - base_address();
      if (used > max_used_) {
        max_used_ = used;
      }
    }
    return result;
  }

  size_t max_used() const {
    return max_used_;
  }

 private:
  size_t max_used_ = 0;
};

} // namespace

extern "C" {

void* et_backend_lookup(const char* name, size_t name_len) {
  char buf[256];
  size_t copy_len = name_len < sizeof(buf) - 1 ? name_len : sizeof(buf) - 1;
  memcpy(buf, name, copy_len);
  buf[copy_len] = '\0';

  BackendInterface* backend = get_backend_class(buf);
  return static_cast<void*>(backend);
}

int et_backend_is_available(void* backend) {
  if (!backend) {
    return 0;
  }
  auto* iface = static_cast<BackendInterface*>(backend);
  return iface->is_available() ? 1 : 0;
}

uint32_t et_backend_init(
    void* backend,
    uint8_t* alloc_buf,
    uint32_t alloc_size,
    const void* processed_data,
    size_t processed_size,
    void** out_handle,
    uint32_t* out_alloc_used) {
  if (!backend || !out_handle) {
    return static_cast<uint32_t>(Error::InvalidArgument);
  }

  auto* iface = static_cast<BackendInterface*>(backend);

  TrackingAllocator alloc(alloc_size, alloc_buf);
  MemoryAllocator* alloc_ptr = alloc_buf ? &alloc : nullptr;

  BackendInitContext init_ctx(alloc_ptr);

  FreeableBuffer processed(processed_data, processed_size, /*free_fn=*/nullptr);

  auto result = iface->init(init_ctx, &processed, {});
  if (!result.ok()) {
    return static_cast<uint32_t>(result.error());
  }
  *out_handle = *result;
  if (out_alloc_used) {
    *out_alloc_used = static_cast<uint32_t>(alloc.max_used());
  }
  return static_cast<uint32_t>(Error::Ok);
}

uint32_t et_backend_execute(
    void* backend,
    void* handle,
    void** args,
    size_t n_args,
    uint8_t* temp_buf,
    uint32_t temp_size) {
  if (!backend) {
    return static_cast<uint32_t>(Error::InvalidArgument);
  }

  auto* iface = static_cast<BackendInterface*>(backend);

  MemoryAllocator alloc(temp_size, temp_buf);
  MemoryAllocator* alloc_ptr = temp_buf ? &alloc : nullptr;

  BackendExecutionContext exec_ctx(/*event_tracer=*/nullptr, alloc_ptr);

  auto* evalue_ptrs = reinterpret_cast<EValue**>(args);
  Error err = iface->execute(
      exec_ctx, static_cast<DelegateHandle*>(handle), {evalue_ptrs, n_args});
  return static_cast<uint32_t>(err);
}

void et_backend_destroy(void* backend, void* handle) {
  if (!backend) {
    return;
  }
  auto* iface = static_cast<BackendInterface*>(backend);
  iface->destroy(static_cast<DelegateHandle*>(handle));
}

} // extern "C"
