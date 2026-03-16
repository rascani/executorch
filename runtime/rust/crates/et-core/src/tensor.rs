// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::{Error, Result};
use crate::scalar_type::ScalarType;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorShapeDynamism {
    Static = 0,
    DynamicBound = 1,
    DynamicUnbound = 2,
}

#[repr(C)]
pub struct TensorImpl {
    pub sizes: *mut i32,
    pub dim_order: *mut u8,
    pub strides: *mut i32,
    pub data: *mut u8,
    pub dim: isize,
    pub numel: isize,
    pub numel_bound: usize,
    pub scalar_type: ScalarType,
    pub shape_dynamism: TensorShapeDynamism,
}

impl TensorImpl {
    pub fn new(
        scalar_type: ScalarType,
        dim: isize,
        sizes: *mut i32,
        data: *mut u8,
        dim_order: *mut u8,
        strides: *mut i32,
        dynamism: TensorShapeDynamism,
    ) -> Self {
        let numel = if dim == 0 {
            1
        } else {
            unsafe { Self::compute_numel(sizes, dim) }
        };
        TensorImpl {
            sizes,
            dim_order,
            strides,
            data,
            dim,
            numel,
            numel_bound: numel as usize,
            scalar_type,
            shape_dynamism: dynamism,
        }
    }

    unsafe fn compute_numel(sizes: *const i32, dim: isize) -> isize {
        let mut n: isize = 1;
        for i in 0..dim as usize {
            n *= *sizes.add(i) as isize;
        }
        n
    }

    pub fn nbytes(&self) -> usize {
        self.numel as usize * self.scalar_type.element_size()
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Tensor {
    pub impl_ptr: *mut TensorImpl,
}

impl Tensor {
    pub unsafe fn new(ptr: *mut TensorImpl) -> Self {
        Tensor { impl_ptr: ptr }
    }

    pub fn is_null(&self) -> bool {
        self.impl_ptr.is_null()
    }

    fn inner(&self) -> &TensorImpl {
        unsafe { &*self.impl_ptr }
    }

    pub fn dim(&self) -> isize {
        self.inner().dim
    }

    pub fn numel(&self) -> isize {
        self.inner().numel
    }

    pub fn scalar_type(&self) -> ScalarType {
        self.inner().scalar_type
    }

    pub fn nbytes(&self) -> usize {
        self.inner().nbytes()
    }

    pub fn sizes(&self) -> &[i32] {
        let inner = self.inner();
        if inner.dim == 0 || inner.sizes.is_null() {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(inner.sizes, inner.dim as usize) }
    }

    pub fn dim_order(&self) -> &[u8] {
        let inner = self.inner();
        if inner.dim == 0 || inner.dim_order.is_null() {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(inner.dim_order, inner.dim as usize) }
    }

    pub fn strides(&self) -> &[i32] {
        let inner = self.inner();
        if inner.dim == 0 || inner.strides.is_null() {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(inner.strides, inner.dim as usize) }
    }

    pub fn data_ptr<T>(&self) -> Result<*const T> {
        let inner = self.inner();
        if inner.data.is_null() {
            return Err(Error::InvalidState);
        }
        Ok(inner.data as *const T)
    }

    pub fn mutable_data_ptr<T>(&self) -> Result<*mut T> {
        let inner = self.inner();
        if inner.data.is_null() {
            return Err(Error::InvalidState);
        }
        Ok(inner.data as *mut T)
    }

    pub fn set_data(&self, ptr: *mut u8) {
        unsafe { (*self.impl_ptr).data = ptr }
    }

    pub fn shape_dynamism(&self) -> TensorShapeDynamism {
        self.inner().shape_dynamism
    }
}
