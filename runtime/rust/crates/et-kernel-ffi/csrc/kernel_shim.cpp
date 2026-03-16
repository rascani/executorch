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

extern "C" {

EtOpFunction et_kernel_lookup(const char* name, size_t name_len) {
  // Copy into a stack buffer and null-terminate.
  char buf[256];
  size_t copy_len = name_len < sizeof(buf) - 1 ? name_len : sizeof(buf) - 1;
  memcpy(buf, name, copy_len);
  buf[copy_len] = '\0';

  auto result = get_op_function_from_registry(buf);
  if (!result.ok()) {
    return nullptr;
  }
  // OpFunction and EtOpFunction have compatible calling conventions
  // because KernelRuntimeContext& is passed as a pointer, and
  // Span<EValue*> is a {pointer, size} pair passed as two args.
  // However, the Rust side calls through a (void* ctx, void** args, size_t
  // n_args) signature, which is what et_kernel_call adapts.
  return reinterpret_cast<EtOpFunction>(*result);
}

void et_kernel_call(
    EtOpFunction func,
    void* ctx,
    void** args,
    size_t n_args) {
  // Cast back to the real C++ types.
  auto* context = static_cast<KernelRuntimeContext*>(ctx);
  auto* evalue_ptrs = reinterpret_cast<EValue**>(args);

  // The C++ OpFunction signature is:
  //   void(KernelRuntimeContext&, Span<EValue*>)
  // Span<EValue*> is a {data, size} pair.
  auto op = reinterpret_cast<OpFunction>(func);
  op(*context, {evalue_ptrs, n_args});
}

void* et_kernel_context_new(void* temp_allocator) {
  auto* alloc = static_cast<MemoryAllocator*>(temp_allocator);
  // Allocate on the heap. In a no_std embedded target, the caller would
  // provide storage; this shim is for hosted builds that link against
  // libexecutorch.a.
  return new KernelRuntimeContext(/*event_tracer=*/nullptr, alloc);
}

void et_kernel_context_destroy(void* ctx) {
  delete static_cast<KernelRuntimeContext*>(ctx);
}

uint32_t et_kernel_context_failure_state(const void* ctx) {
  auto* context = static_cast<const KernelRuntimeContext*>(ctx);
  return static_cast<uint32_t>(context->failure_state());
}

void et_kernel_context_reset_failure(void* ctx) {
  // KernelRuntimeContext doesn't expose a reset, but we can reconstruct.
  // For now, the Rust runtime creates a fresh context per method init,
  // so this is a no-op placeholder.
  (void)ctx;
}

size_t et_sizeof_evalue(void) {
  return sizeof(EValue);
}

size_t et_alignof_evalue(void) {
  return alignof(EValue);
}

} // extern "C"
