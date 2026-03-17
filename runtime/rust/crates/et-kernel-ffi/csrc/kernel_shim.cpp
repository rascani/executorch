/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma clang diagnostic ignored "-Wcast-function-type-mismatch"

#include "kernel_shim.h"

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>

#include <cstring>

using namespace executorch::runtime;

namespace {

struct KernelContextData {
  MemoryAllocator alloc;
  KernelRuntimeContext context;

  KernelContextData(uint8_t* buf, uint32_t size)
      : alloc(size, buf),
        context(/*event_tracer=*/nullptr, buf ? &alloc : nullptr) {}
};

} // namespace

extern "C" {

EtOpFunction et_kernel_lookup(const char* name, size_t name_len) {
  char buf[256];
  size_t copy_len = name_len < sizeof(buf) - 1 ? name_len : sizeof(buf) - 1;
  memcpy(buf, name, copy_len);
  buf[copy_len] = '\0';

  auto result = get_op_function_from_registry(buf);
  if (!result.ok()) {
    return nullptr;
  }
  return reinterpret_cast<EtOpFunction>(*result);
}

void et_kernel_call(
    EtOpFunction func,
    void* ctx,
    void** args,
    size_t n_args) {
  auto* data = static_cast<KernelContextData*>(ctx);
  auto* evalue_ptrs = reinterpret_cast<EValue**>(args);

  auto op = reinterpret_cast<OpFunction>(func);
  op(data->context, {evalue_ptrs, n_args});
}

void* et_kernel_context_new(uint8_t* temp_buf, uint32_t temp_size) {
  return new KernelContextData(temp_buf, temp_size);
}

void et_kernel_context_destroy(void* ctx) {
  delete static_cast<KernelContextData*>(ctx);
}

uint32_t et_kernel_context_failure_state(const void* ctx) {
  auto* data = static_cast<const KernelContextData*>(ctx);
  return static_cast<uint32_t>(data->context.failure_state());
}

void et_kernel_context_reset_temp(void* ctx) {
  auto* data = static_cast<KernelContextData*>(ctx);
  data->alloc.reset();
}

size_t et_sizeof_evalue(void) {
  return sizeof(EValue);
}

size_t et_alignof_evalue(void) {
  return alignof(EValue);
}

} // extern "C"
