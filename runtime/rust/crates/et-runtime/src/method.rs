// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use et_core::allocator::{BumpAllocator, HierarchicalAllocator, MemoryManager};
use et_core::error::{Error, Result};
use et_core::evalue::{BoxedEvalueListI64, CArrayRef, EValue};
use et_core::scalar_type::ScalarType;
use et_core::tensor::{TensorImpl, TensorShapeDynamism};

use et_flatbuffer::executorch_flatbuffer::{self as fb, ExecutionPlan, InstructionArguments};

use crate::data_loader::DataLoader;
use crate::program::Program;

type KernelFn = unsafe extern "C" fn(ctx: *mut u8, args: *mut *mut u8, n_args: usize);
type OptKernelFn = Option<KernelFn>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum InstructionType {
    KernelCall,
    DelegateCall,
    MoveCall,
    JumpFalseCall,
    FreeCall,
}

struct InstructionData {
    kind: InstructionType,
    operand_a: usize,
    operand_b: usize,
}

pub struct Method {
    n_value: usize,
    values: *mut EValue,
    n_chains: usize,
    chain_instructions: *const ChainData,
    kernel_context: *mut u8,
    temp_allocator_ptr: *mut BumpAllocator,
    input_indices: *const usize,
    n_inputs: usize,
    output_indices: *const usize,
    n_outputs: usize,
}

struct ChainData {
    n_instructions: usize,
    instructions: *const InstructionData,
    kernels: *const OptKernelFn,
    arg_lists: *const ArgList,
}

struct ArgList {
    args: *const *mut EValue,
    n_args: usize,
}

impl Method {
    pub unsafe fn init<L: DataLoader>(
        program: &Program<L>,
        plan: ExecutionPlan<'_>,
        memory: &mut MemoryManager<'_>,
    ) -> Result<Self> {
        let allocator = &mut *memory.method_allocator;

        let fb_values = plan.values().ok_or(Error::InvalidProgram)?;
        let n_value = fb_values.len();
        let values: *mut EValue = allocator.allocate_slice(n_value);
        if values.is_null() && n_value > 0 {
            return Err(Error::MemoryAllocationFailed);
        }

        for i in 0..n_value {
            let fb_val = fb_values.get(i);
            let ev = Self::deserialize_evalue(
                &fb_val,
                allocator,
                program,
                memory.planned_memory.as_deref_mut(),
                values,
                n_value,
            )?;
            core::ptr::write(values.add(i), ev);
        }

        let fb_chains = plan.chains().ok_or(Error::InvalidProgram)?;
        let n_chains = fb_chains.len();
        let chain_data: *mut ChainData = allocator.allocate_slice(n_chains);
        if chain_data.is_null() && n_chains > 0 {
            return Err(Error::MemoryAllocationFailed);
        }

        let fb_operators = plan.operators();

        for ci in 0..n_chains {
            let fb_chain = fb_chains.get(ci);
            let fb_instructions = fb_chain.instructions().ok_or(Error::InvalidProgram)?;
            let n_instr = fb_instructions.len();

            let instructions: *mut InstructionData = allocator.allocate_slice(n_instr);
            let kernels: *mut OptKernelFn = allocator.allocate_slice(n_instr);
            let arg_lists: *mut ArgList = allocator.allocate_slice(n_instr);

            if (instructions.is_null() || kernels.is_null() || arg_lists.is_null()) && n_instr > 0 {
                return Err(Error::MemoryAllocationFailed);
            }

            for ii in 0..n_instr {
                let instr = fb_instructions.get(ii);
                let instr_type = instr.instr_args_type();

                let (kind, op_a, op_b, kernel, args, n_args) = match instr_type {
                    InstructionArguments::KernelCall => {
                        let kc = instr
                            .instr_args_as_kernel_call()
                            .ok_or(Error::InvalidProgram)?;

                        let ops = fb_operators.as_ref().ok_or(Error::InvalidProgram)?;
                        let op_index = kc.op_index() as usize;
                        if op_index >= ops.len() {
                            return Err(Error::InvalidProgram);
                        }
                        let op = ops.get(op_index);
                        let op_name = op.name().ok_or(Error::InvalidProgram)?;
                        let overload = op.overload().unwrap_or("");

                        let mut name_buf = [0u8; 256];
                        let full_name_len = Self::format_op_name(
                            op_name.as_bytes(),
                            overload.as_bytes(),
                            &mut name_buf,
                        );

                        let kernel_fn = et_kernel_ffi::et_kernel_lookup(
                            name_buf.as_ptr(),
                            full_name_len,
                        );
                        if kernel_fn.is_none() {
                            return Err(Error::OperatorMissing);
                        }

                        let fb_args = kc.args().ok_or(Error::InvalidProgram)?;
                        let na = fb_args.len();
                        let arg_ptrs: *mut *mut EValue = allocator.allocate_slice(na);
                        if arg_ptrs.is_null() && na > 0 {
                            return Err(Error::MemoryAllocationFailed);
                        }
                        for ai in 0..na {
                            let val_idx = fb_args.get(ai) as usize;
                            if val_idx >= n_value {
                                return Err(Error::InvalidProgram);
                            }
                            core::ptr::write(arg_ptrs.add(ai), values.add(val_idx));
                        }

                        (InstructionType::KernelCall, 0, 0, kernel_fn, arg_ptrs as *const *mut EValue, na)
                    }
                    InstructionArguments::DelegateCall => {
                        (InstructionType::DelegateCall, 0, 0, None, core::ptr::null(), 0)
                    }
                    InstructionArguments::MoveCall => {
                        let mc = instr
                            .instr_args_as_move_call()
                            .ok_or(Error::InvalidProgram)?;
                        (InstructionType::MoveCall, mc.move_from() as usize, mc.move_to() as usize, None, core::ptr::null(), 0)
                    }
                    InstructionArguments::JumpFalseCall => {
                        let jf = instr
                            .instr_args_as_jump_false_call()
                            .ok_or(Error::InvalidProgram)?;
                        (InstructionType::JumpFalseCall, jf.cond_value_index() as usize, jf.destination_instruction() as usize, None, core::ptr::null(), 0)
                    }
                    InstructionArguments::FreeCall => {
                        let fc = instr
                            .instr_args_as_free_call()
                            .ok_or(Error::InvalidProgram)?;
                        (InstructionType::FreeCall, fc.value_index() as usize, 0, None, core::ptr::null(), 0)
                    }
                    _ => {
                        return Err(Error::InvalidProgram);
                    }
                };

                core::ptr::write(
                    instructions.add(ii),
                    InstructionData { kind, operand_a: op_a, operand_b: op_b },
                );
                core::ptr::write(kernels.add(ii), kernel);
                core::ptr::write(
                    arg_lists.add(ii),
                    ArgList { args, n_args },
                );
            }

            core::ptr::write(
                chain_data.add(ci),
                ChainData {
                    n_instructions: n_instr,
                    instructions,
                    kernels,
                    arg_lists,
                },
            );
        }

        let fb_inputs = plan.inputs().ok_or(Error::InvalidProgram)?;
        let n_inputs = fb_inputs.len();
        let input_indices: *mut usize = allocator.allocate_slice(n_inputs);
        if input_indices.is_null() && n_inputs > 0 {
            return Err(Error::MemoryAllocationFailed);
        }
        for i in 0..n_inputs {
            core::ptr::write(input_indices.add(i), fb_inputs.get(i) as usize);
        }

        let fb_outputs = plan.outputs().ok_or(Error::InvalidProgram)?;
        let n_outputs = fb_outputs.len();
        let output_indices: *mut usize = allocator.allocate_slice(n_outputs);
        if output_indices.is_null() && n_outputs > 0 {
            return Err(Error::MemoryAllocationFailed);
        }
        for i in 0..n_outputs {
            core::ptr::write(output_indices.add(i), fb_outputs.get(i) as usize);
        }

        let temp_alloc_ptr = match &mut memory.temp_allocator {
            Some(ta) => *ta as *mut BumpAllocator,
            None => core::ptr::null_mut(),
        };

        let kernel_context = et_kernel_ffi::et_kernel_context_new(temp_alloc_ptr as *mut u8);

        Ok(Method {
            n_value,
            values,
            n_chains,
            chain_instructions: chain_data,
            kernel_context,
            temp_allocator_ptr: temp_alloc_ptr as *mut u8 as *mut BumpAllocator,
            input_indices,
            n_inputs,
            output_indices,
            n_outputs,
        })
    }

    pub fn execute(&mut self) -> Result<()> {
        for ci in 0..self.n_chains {
            let chain = unsafe { &*self.chain_instructions.add(ci) };
            let mut pc: usize = 0;

            while pc < chain.n_instructions {
                let instr = unsafe { &*chain.instructions.add(pc) };

                match instr.kind {
                    InstructionType::KernelCall => {
                        let kernel = unsafe { *chain.kernels.add(pc) };
                        let kernel = kernel.ok_or(Error::OperatorMissing)?;
                        let arg_list = unsafe { &*chain.arg_lists.add(pc) };
                        unsafe {
                            et_kernel_ffi::et_kernel_call(
                                kernel,
                                self.kernel_context,
                                arg_list.args as *mut *mut u8,
                                arg_list.n_args,
                            );
                        }
                        let err =
                            unsafe { et_kernel_ffi::et_kernel_context_failure_state(self.kernel_context) };
                        if err != 0 {
                            return Err(Error::try_from(err).unwrap_or(Error::Internal));
                        }
                    }
                    InstructionType::DelegateCall => {
                        return Err(Error::NotSupported);
                    }
                    InstructionType::MoveCall => {
                        let from_idx = instr.operand_a;
                        let to_idx = instr.operand_b;
                        if from_idx >= self.n_value || to_idx >= self.n_value {
                            return Err(Error::InvalidProgram);
                        }
                        unsafe {
                            let from = &*self.values.add(from_idx);
                            let to = &mut *self.values.add(to_idx);
                            to.copy_from(from);
                        }
                    }
                    InstructionType::JumpFalseCall => {
                        let cond_idx = instr.operand_a;
                        let dest = instr.operand_b;
                        if cond_idx >= self.n_value {
                            return Err(Error::InvalidProgram);
                        }
                        let cond_val = unsafe { &*self.values.add(cond_idx) };
                        let cond = cond_val.to_bool().map_err(|_| Error::InvalidType)?;
                        if !cond {
                            pc = dest;
                            continue;
                        }
                    }
                    InstructionType::FreeCall => {
                        let val_idx = instr.operand_a;
                        if val_idx >= self.n_value {
                            return Err(Error::InvalidProgram);
                        }
                        let val = unsafe { &*self.values.add(val_idx) };
                        if val.is_tensor() {
                            if let Ok(tensor) = val.to_tensor() {
                                tensor.set_data(core::ptr::null_mut());
                            }
                        }
                    }
                }

                if !self.temp_allocator_ptr.is_null() {
                    unsafe {
                        (*self.temp_allocator_ptr).reset();
                    }
                }
                pc += 1;
            }
        }
        Ok(())
    }

    pub fn set_input(&mut self, idx: usize, value: &EValue) -> Result<()> {
        if idx >= self.n_inputs {
            return Err(Error::InvalidArgument);
        }
        let val_idx = unsafe { *self.input_indices.add(idx) };
        if val_idx >= self.n_value {
            return Err(Error::InvalidProgram);
        }
        unsafe {
            let target = &mut *self.values.add(val_idx);
            target.copy_from(value);
        }
        Ok(())
    }

    pub fn get_output(&self, idx: usize) -> Result<&EValue> {
        if idx >= self.n_outputs {
            return Err(Error::InvalidArgument);
        }
        let val_idx = unsafe { *self.output_indices.add(idx) };
        if val_idx >= self.n_value {
            return Err(Error::InvalidProgram);
        }
        Ok(unsafe { &*self.values.add(val_idx) })
    }

    pub fn num_inputs(&self) -> usize {
        self.n_inputs
    }

    pub fn num_outputs(&self) -> usize {
        self.n_outputs
    }

    fn format_op_name(name: &[u8], overload: &[u8], buf: &mut [u8]) -> usize {
        let mut pos = 0;
        for &b in name {
            if pos >= buf.len() - 1 {
                break;
            }
            buf[pos] = b;
            pos += 1;
        }
        if !overload.is_empty() {
            if pos < buf.len() - 1 {
                buf[pos] = b'.';
                pos += 1;
            }
            for &b in overload {
                if pos >= buf.len() - 1 {
                    break;
                }
                buf[pos] = b;
                pos += 1;
            }
        }
        buf[pos] = 0;
        pos
    }

    unsafe fn deserialize_evalue<L: DataLoader>(
        fb_val: &fb::EValue<'_>,
        allocator: &mut BumpAllocator,
        program: &Program<L>,
        planned_memory: Option<&mut HierarchicalAllocator<'_>>,
        values: *mut EValue,
        n_value: usize,
    ) -> Result<EValue> {
        let val_type = fb_val.val_type();

        match val_type {
            fb::KernelTypes::Null => Ok(EValue::none()),

            fb::KernelTypes::Int => {
                let int_val = fb_val.val_as_int().ok_or(Error::InvalidProgram)?;
                Ok(EValue::from_int(int_val.int_val()))
            }

            fb::KernelTypes::Double => {
                let double_val = fb_val.val_as_double().ok_or(Error::InvalidProgram)?;
                Ok(EValue::from_double(double_val.double_val()))
            }

            fb::KernelTypes::Bool => {
                let bool_val = fb_val.val_as_bool().ok_or(Error::InvalidProgram)?;
                Ok(EValue::from_bool(bool_val.bool_val()))
            }

            fb::KernelTypes::Tensor => {
                let fb_tensor = fb_val.val_as_tensor().ok_or(Error::InvalidProgram)?;
                Self::deserialize_tensor(fb_tensor, allocator, program, planned_memory)
            }

            fb::KernelTypes::IntList => {
                let int_list = fb_val.val_as_int_list().ok_or(Error::InvalidProgram)?;
                let items = int_list.items().ok_or(Error::InvalidProgram)?;
                let n = items.len();

                let evalp_list: *mut *mut EValue = allocator.allocate_slice(n);
                if evalp_list.is_null() && n > 0 {
                    return Err(Error::MemoryAllocationFailed);
                }

                let int_scratch: *mut i64 = allocator.allocate_slice(n);
                if int_scratch.is_null() && n > 0 {
                    return Err(Error::MemoryAllocationFailed);
                }

                for j in 0..n {
                    let value_index = items.get(j) as usize;
                    if value_index >= n_value {
                        return Err(Error::InvalidProgram);
                    }
                    core::ptr::write(evalp_list.add(j), values.add(value_index));
                }

                let boxed: *mut BoxedEvalueListI64 = allocator.allocate_instance();
                if boxed.is_null() {
                    return Err(Error::MemoryAllocationFailed);
                }
                core::ptr::write(
                    boxed,
                    BoxedEvalueListI64 {
                        wrapped_data: evalp_list as *const *mut EValue,
                        wrapped_len: n,
                        unwrapped: int_scratch,
                    },
                );

                Ok(EValue::from_int_list(boxed))
            }

            fb::KernelTypes::BoolList => {
                let bool_list = fb_val.val_as_bool_list().ok_or(Error::InvalidProgram)?;
                let items = bool_list.items().ok_or(Error::InvalidProgram)?;
                let n = items.len();

                let data: *mut bool = allocator.allocate_slice(n);
                if data.is_null() && n > 0 {
                    return Err(Error::MemoryAllocationFailed);
                }
                for j in 0..n {
                    core::ptr::write(data.add(j), items.get(j));
                }

                let array_ref: *mut CArrayRef<bool> = allocator.allocate_instance();
                if array_ref.is_null() {
                    return Err(Error::MemoryAllocationFailed);
                }
                core::ptr::write(
                    array_ref,
                    CArrayRef { data: data as *const bool, len: n },
                );

                Ok(EValue::from_bool_list(array_ref))
            }

            fb::KernelTypes::DoubleList => {
                let double_list = fb_val.val_as_double_list().ok_or(Error::InvalidProgram)?;
                let items = double_list.items().ok_or(Error::InvalidProgram)?;
                let n = items.len();

                let data: *mut f64 = allocator.allocate_slice(n);
                if data.is_null() && n > 0 {
                    return Err(Error::MemoryAllocationFailed);
                }
                for j in 0..n {
                    core::ptr::write(data.add(j), items.get(j));
                }

                let array_ref: *mut CArrayRef<f64> = allocator.allocate_instance();
                if array_ref.is_null() {
                    return Err(Error::MemoryAllocationFailed);
                }
                core::ptr::write(
                    array_ref,
                    CArrayRef { data: data as *const f64, len: n },
                );

                Ok(EValue::from_double_list(array_ref))
            }

            _ => Ok(EValue::none()),
        }
    }

    unsafe fn deserialize_tensor<L: DataLoader>(
        fb_tensor: fb::Tensor<'_>,
        allocator: &mut BumpAllocator,
        program: &Program<L>,
        planned_memory: Option<&mut HierarchicalAllocator<'_>>,
    ) -> Result<EValue> {
        let fb_sizes = fb_tensor.sizes().ok_or(Error::InvalidProgram)?;
        let dim = fb_sizes.len();

        let sizes: *mut i32 = allocator.allocate_slice(dim);
        if sizes.is_null() && dim > 0 {
            return Err(Error::MemoryAllocationFailed);
        }
        for i in 0..dim {
            core::ptr::write(sizes.add(i), fb_sizes.get(i));
        }

        let dim_order: *mut u8 = if let Some(fb_dim_order) = fb_tensor.dim_order() {
            let p: *mut u8 = allocator.allocate_slice(dim);
            if p.is_null() && dim > 0 {
                return Err(Error::MemoryAllocationFailed);
            }
            for i in 0..fb_dim_order.len() {
                core::ptr::write(p.add(i), fb_dim_order.get(i));
            }
            p
        } else {
            let p: *mut u8 = allocator.allocate_slice(dim);
            if p.is_null() && dim > 0 {
                return Err(Error::MemoryAllocationFailed);
            }
            for i in 0..dim {
                core::ptr::write(p.add(i), i as u8);
            }
            p
        };

        let strides: *mut i32 = allocator.allocate_slice(dim);
        if strides.is_null() && dim > 0 {
            return Err(Error::MemoryAllocationFailed);
        }
        Self::compute_strides(sizes, dim_order, strides, dim);

        let fb_scalar_type = fb_tensor.scalar_type();
        let scalar_type = ScalarType::try_from(fb_scalar_type.0).map_err(|_| Error::InvalidProgram)?;

        let fb_dynamism = fb_tensor.shape_dynamism();
        let dynamism = match fb_dynamism.0 {
            0 => TensorShapeDynamism::Static,
            1 => TensorShapeDynamism::DynamicBound,
            2 => TensorShapeDynamism::DynamicUnbound,
            _ => TensorShapeDynamism::Static,
        };

        let data_buffer_idx = fb_tensor.data_buffer_idx() as usize;
        let allocation_info = fb_tensor.allocation_info();

        let data_ptr: *mut u8 = if data_buffer_idx > 0 && allocation_info.is_none() {
            let mut numel: usize = 1;
            for i in 0..dim {
                numel *= *sizes.add(i) as usize;
            }
            let nbytes = numel * scalar_type.element_size();
            program
                .get_constant_buffer_data(data_buffer_idx, nbytes)
                .unwrap_or(core::ptr::null()) as *mut u8
        } else if let Some(alloc_info) = allocation_info {
            let memory_id = alloc_info.memory_id();
            let offset_low = alloc_info.memory_offset_low() as u64;
            let offset_high = alloc_info.memory_offset_high() as u64;
            let offset = ((offset_high << 32) | offset_low) as usize;

            let mut numel: usize = 1;
            for i in 0..dim {
                numel *= *sizes.add(i) as usize;
            }
            let nbytes = numel * scalar_type.element_size();

            if let Some(ref mut pm) = { planned_memory } {
                pm.get_offset_address(memory_id, offset, nbytes)
                    .unwrap_or(core::ptr::null_mut())
            } else {
                core::ptr::null_mut()
            }
        } else {
            core::ptr::null_mut()
        };

        let tensor_impl: *mut TensorImpl = allocator.allocate_instance();
        if tensor_impl.is_null() {
            return Err(Error::MemoryAllocationFailed);
        }
        core::ptr::write(
            tensor_impl,
            TensorImpl::new(scalar_type, dim as isize, sizes, data_ptr, dim_order, strides, dynamism),
        );

        Ok(EValue::from_tensor(tensor_impl))
    }

    unsafe fn compute_strides(
        sizes: *const i32,
        dim_order: *const u8,
        strides: *mut i32,
        dim: usize,
    ) {
        if dim == 0 {
            return;
        }
        for d in (0..dim).rev() {
            let cur_dim = *dim_order.add(d) as usize;
            if d == dim - 1 {
                core::ptr::write(strides.add(cur_dim), 1);
            } else {
                let next_dim = *dim_order.add(d + 1) as usize;
                let stride = *strides.add(next_dim) * *sizes.add(next_dim);
                core::ptr::write(strides.add(cur_dim), stride);
            }
        }
    }
}

impl Drop for Method {
    fn drop(&mut self) {
        if !self.kernel_context.is_null() {
            unsafe {
                et_kernel_ffi::et_kernel_context_destroy(self.kernel_context);
            }
        }
    }
}
