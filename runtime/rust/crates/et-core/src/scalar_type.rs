// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#[repr(i8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Byte = 0,
    Char = 1,
    Short = 2,
    Int = 3,
    Long = 4,
    Half = 5,
    Float = 6,
    Double = 7,
    ComplexHalf = 8,
    ComplexFloat = 9,
    ComplexDouble = 10,
    Bool = 11,
    QInt8 = 12,
    QUInt8 = 13,
    QInt32 = 14,
    BFloat16 = 15,
    QUInt4x2 = 16,
    QUInt2x4 = 17,
    Bits1x8 = 18,
    Bits2x4 = 19,
    Bits4x2 = 20,
    Bits8 = 21,
    Bits16 = 22,
    Float8e5m2 = 23,
    Float8e4m3fn = 24,
    Float8e5m2fnuz = 25,
    Float8e4m3fnuz = 26,
    UInt16 = 27,
    UInt32 = 28,
    UInt64 = 29,
    Undefined = 30,
}

impl ScalarType {
    pub fn element_size(self) -> usize {
        match self {
            Self::Byte | Self::Char | Self::Bool | Self::QInt8 | Self::QUInt8
            | Self::QUInt4x2 | Self::QUInt2x4 | Self::Bits1x8 | Self::Bits2x4
            | Self::Bits4x2 | Self::Bits8 | Self::Float8e5m2 | Self::Float8e4m3fn
            | Self::Float8e5m2fnuz | Self::Float8e4m3fnuz => 1,

            Self::Short | Self::Half | Self::BFloat16 | Self::Bits16 | Self::UInt16 => 2,

            Self::Int | Self::QInt32 | Self::ComplexHalf | Self::Float | Self::UInt32 => 4,

            Self::Long | Self::Double | Self::ComplexFloat | Self::UInt64 => 8,

            Self::ComplexDouble => 16,

            Self::Undefined => 0,
        }
    }
}

impl TryFrom<i8> for ScalarType {
    type Error = ();

    fn try_from(val: i8) -> core::result::Result<Self, ()> {
        match val {
            0 => Ok(Self::Byte),
            1 => Ok(Self::Char),
            2 => Ok(Self::Short),
            3 => Ok(Self::Int),
            4 => Ok(Self::Long),
            5 => Ok(Self::Half),
            6 => Ok(Self::Float),
            7 => Ok(Self::Double),
            8 => Ok(Self::ComplexHalf),
            9 => Ok(Self::ComplexFloat),
            10 => Ok(Self::ComplexDouble),
            11 => Ok(Self::Bool),
            12 => Ok(Self::QInt8),
            13 => Ok(Self::QUInt8),
            14 => Ok(Self::QInt32),
            15 => Ok(Self::BFloat16),
            16 => Ok(Self::QUInt4x2),
            17 => Ok(Self::QUInt2x4),
            18 => Ok(Self::Bits1x8),
            19 => Ok(Self::Bits2x4),
            20 => Ok(Self::Bits4x2),
            21 => Ok(Self::Bits8),
            22 => Ok(Self::Bits16),
            23 => Ok(Self::Float8e5m2),
            24 => Ok(Self::Float8e4m3fn),
            25 => Ok(Self::Float8e5m2fnuz),
            26 => Ok(Self::Float8e4m3fnuz),
            27 => Ok(Self::UInt16),
            28 => Ok(Self::UInt32),
            29 => Ok(Self::UInt64),
            _ => Err(()),
        }
    }
}
