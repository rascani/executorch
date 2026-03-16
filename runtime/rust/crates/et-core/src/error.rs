// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Error {
    Internal = 0x01,
    InvalidState = 0x02,
    EndOfMethod = 0x03,
    AlreadyLoaded = 0x04,

    NotSupported = 0x10,
    NotImplemented = 0x11,
    InvalidArgument = 0x12,
    InvalidType = 0x13,
    OperatorMissing = 0x14,
    RegistrationExceedingMaxKernels = 0x15,
    RegistrationAlreadyRegistered = 0x16,

    NotFound = 0x20,
    MemoryAllocationFailed = 0x21,
    AccessFailed = 0x22,
    InvalidProgram = 0x23,
    InvalidExternalData = 0x24,
    OutOfResources = 0x25,

    DelegateInvalidCompatibility = 0x30,
    DelegateMemoryAllocationFailed = 0x31,
    DelegateInvalidHandle = 0x32,
}

impl Error {
    /// Convert a C++ error_code_t (u32) into a Rust Error.
    /// Returns `Ok(())` for the C++ `Error::Ok` (0x00) success value.
    pub fn from_code(val: u32) -> Result<()> {
        match val {
            0x00 => Ok(()),
            _ => Err(Self::try_from(val).unwrap_or(Error::Internal)),
        }
    }
}

impl TryFrom<u32> for Error {
    type Error = ();

    fn try_from(val: u32) -> core::result::Result<Self, ()> {
        match val {
            0x01 => Ok(Error::Internal),
            0x02 => Ok(Error::InvalidState),
            0x03 => Ok(Error::EndOfMethod),
            0x04 => Ok(Error::AlreadyLoaded),
            0x10 => Ok(Error::NotSupported),
            0x11 => Ok(Error::NotImplemented),
            0x12 => Ok(Error::InvalidArgument),
            0x13 => Ok(Error::InvalidType),
            0x14 => Ok(Error::OperatorMissing),
            0x15 => Ok(Error::RegistrationExceedingMaxKernels),
            0x16 => Ok(Error::RegistrationAlreadyRegistered),
            0x20 => Ok(Error::NotFound),
            0x21 => Ok(Error::MemoryAllocationFailed),
            0x22 => Ok(Error::AccessFailed),
            0x23 => Ok(Error::InvalidProgram),
            0x24 => Ok(Error::InvalidExternalData),
            0x25 => Ok(Error::OutOfResources),
            0x30 => Ok(Error::DelegateInvalidCompatibility),
            0x31 => Ok(Error::DelegateMemoryAllocationFailed),
            0x32 => Ok(Error::DelegateInvalidHandle),
            _ => Err(()),
        }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Debug derive gives us the variant name already
        write!(f, "{:?}", self)
    }
}

pub type Result<T> = core::result::Result<T, Error>;
