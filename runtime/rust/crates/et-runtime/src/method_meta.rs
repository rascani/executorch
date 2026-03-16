// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use et_core::error::{Error, Result};
use et_core::scalar_type::ScalarType;
use et_flatbuffer::executorch_flatbuffer::ExecutionPlan;

pub struct TensorInfo {
    pub sizes: &'static [i32],
    pub scalar_type: ScalarType,
    pub is_memory_planned: bool,
    pub nbytes: usize,
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

    pub fn output_index(&self, idx: usize) -> Result<usize> {
        let outputs = self.plan.outputs().ok_or(Error::NotFound)?;
        if idx >= outputs.len() {
            return Err(Error::InvalidArgument);
        }
        Ok(outputs.get(idx) as usize)
    }
}
