use crate::tag::Tag;
use crate::tensor::{Tensor, TensorImpl};
use crate::error::{Error, Result};

#[repr(C)]
#[derive(Clone, Copy)]
pub union EValuePayload {
    pub as_int: i64,
    pub as_double: f64,
    pub as_bool: bool,
    pub as_tensor: Tensor,
}

#[repr(C)]
pub struct EValue {
    pub payload: EValuePayload,
    pub tag: Tag,
}

impl EValue {
    pub fn none() -> Self {
        EValue {
            payload: EValuePayload { as_int: 0 },
            tag: Tag::None,
        }
    }

    pub fn from_int(val: i64) -> Self {
        EValue {
            payload: EValuePayload { as_int: val },
            tag: Tag::Int,
        }
    }

    pub fn from_double(val: f64) -> Self {
        EValue {
            payload: EValuePayload { as_double: val },
            tag: Tag::Double,
        }
    }

    pub fn from_bool(val: bool) -> Self {
        EValue {
            payload: EValuePayload { as_bool: val },
            tag: Tag::Bool,
        }
    }

    pub fn from_tensor(tensor_impl: *mut TensorImpl) -> Self {
        EValue {
            payload: EValuePayload {
                as_tensor: Tensor { impl_ptr: tensor_impl },
            },
            tag: Tag::Tensor,
        }
    }

    pub fn is_none(&self) -> bool {
        self.tag == Tag::None
    }

    pub fn is_int(&self) -> bool {
        self.tag == Tag::Int
    }

    pub fn is_double(&self) -> bool {
        self.tag == Tag::Double
    }

    pub fn is_bool(&self) -> bool {
        self.tag == Tag::Bool
    }

    pub fn is_tensor(&self) -> bool {
        self.tag == Tag::Tensor
    }

    pub fn to_int(&self) -> Result<i64> {
        if !self.is_int() {
            return Err(Error::InvalidType);
        }
        Ok(unsafe { self.payload.as_int })
    }

    pub fn to_double(&self) -> Result<f64> {
        if !self.is_double() {
            return Err(Error::InvalidType);
        }
        Ok(unsafe { self.payload.as_double })
    }

    pub fn to_bool(&self) -> Result<bool> {
        if !self.is_bool() {
            return Err(Error::InvalidType);
        }
        Ok(unsafe { self.payload.as_bool })
    }

    pub fn to_tensor(&self) -> Result<Tensor> {
        if !self.is_tensor() {
            return Err(Error::InvalidType);
        }
        Ok(unsafe { self.payload.as_tensor })
    }

    pub fn to_tensor_mut(&mut self) -> Result<Tensor> {
        if !self.is_tensor() {
            return Err(Error::InvalidType);
        }
        Ok(unsafe { self.payload.as_tensor })
    }

    pub fn copy_from(&mut self, other: &EValue) {
        self.tag = other.tag;
        self.payload = other.payload;
    }
}
