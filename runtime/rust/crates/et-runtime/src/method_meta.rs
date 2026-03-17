// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use et_core::error::{Error, Result};
use et_core::scalar_type::ScalarType;
use et_flatbuffer::executorch_flatbuffer::ExecutionPlan;

pub struct TensorInfo {
    pub scalar_type: ScalarType,
    pub is_memory_planned: bool,
    pub nbytes: usize,
    pub ndim: usize,
    sizes_buf: [i32; 16],
}

impl TensorInfo {
    pub fn sizes(&self) -> &[i32] {
        &self.sizes_buf[..self.ndim]
    }
}

pub struct MethodMeta<'a> {
    plan: ExecutionPlan<'a>,
}

impl<'a> MethodMeta<'a> {
    pub fn new(plan: ExecutionPlan<'a>) -> Self {
        MethodMeta { plan }
    }

    pub fn name(&self) -> Option<&'a str> {
        self.plan.name()
    }

    pub fn num_inputs(&self) -> usize {
        self.plan.inputs().map_or(0, |v| v.len())
    }

    pub fn num_outputs(&self) -> usize {
        self.plan.outputs().map_or(0, |v| v.len())
    }

    pub fn num_memory_planned_buffers(&self) -> usize {
        self.plan
            .non_const_buffer_sizes()
            .map_or(0, |v| v.len())
    }

    pub fn memory_planned_buffer_size(&self, idx: usize) -> Result<usize> {
        let sizes = self
            .plan
            .non_const_buffer_sizes()
            .ok_or(Error::NotFound)?;
        if idx >= sizes.len() {
            return Err(Error::InvalidArgument);
        }
        Ok(sizes.get(idx) as usize)
    }

    pub fn input_index(&self, idx: usize) -> Result<usize> {
        let inputs = self.plan.inputs().ok_or(Error::NotFound)?;
        if idx >= inputs.len() {
            return Err(Error::InvalidArgument);
        }
        Ok(inputs.get(idx) as usize)
    }

    pub fn input_tensor_meta(&self, idx: usize) -> Result<TensorInfo> {
        let inputs = self.plan.inputs().ok_or(Error::NotFound)?;
        if idx >= inputs.len() {
            return Err(Error::InvalidArgument);
        }
        let val_idx = inputs.get(idx) as usize;
        let values = self.plan.values().ok_or(Error::InvalidProgram)?;
        if val_idx >= values.len() {
            return Err(Error::InvalidProgram);
        }
        let val = values.get(val_idx);
        let tensor = val.val_as_tensor().ok_or(Error::InvalidArgument)?;
        let fb_sizes = tensor.sizes().ok_or(Error::InvalidProgram)?;
        let scalar_type =
            ScalarType::try_from(tensor.scalar_type().0).map_err(|_| Error::InvalidProgram)?;
        let numel: usize = (0..fb_sizes.len())
            .map(|i| fb_sizes.get(i) as usize)
            .product();
        let nbytes = numel * scalar_type.element_size();
        let is_memory_planned =
            tensor.allocation_info().is_some() || tensor.data_buffer_idx() != 0;

        let ndim = fb_sizes.len();
        if ndim > 16 {
            return Err(Error::InvalidProgram);
        }
        let mut sizes_buf = [0i32; 16];
        for i in 0..ndim {
            sizes_buf[i] = fb_sizes.get(i);
        }

        Ok(TensorInfo {
            scalar_type,
            is_memory_planned,
            nbytes,
            ndim,
            sizes_buf,
        })
    }

    pub fn output_index(&self, idx: usize) -> Result<usize> {
        let outputs = self.plan.outputs().ok_or(Error::NotFound)?;
        if idx >= outputs.len() {
            return Err(Error::InvalidArgument);
        }
        Ok(outputs.get(idx) as usize)
    }
}
