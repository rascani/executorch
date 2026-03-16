// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use et_core::error::{Error, Result};

const FLATBUFFER_OFFSET_SIZE: usize = 4;
const FILE_IDENTIFIER_SIZE: usize = 4;
const EXPECTED_FILE_ID: &[u8; 4] = b"ET12";

const EXTENDED_HEADER_MAGIC: &[u8; 4] = b"eh00";
const EXTENDED_HEADER_OFFSET: usize = 8;
const EXTENDED_HEADER_STRUCT_SIZE: usize = 4 + 4 + 8 + 8;

pub const MIN_HEAD_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderStatus {
    CompatibleVersion,
    IncompatibleVersion,
    NotPresent,
    ShortData,
}

#[derive(Debug, Clone, Copy)]
pub struct ExtendedHeader {
    pub program_size: u64,
    pub segment_base_offset: u64,
}

pub fn check_header(data: &[u8]) -> HeaderStatus {
    if data.len() < MIN_HEAD_BYTES {
        return HeaderStatus::ShortData;
    }

    let file_id = &data[FLATBUFFER_OFFSET_SIZE..FLATBUFFER_OFFSET_SIZE + FILE_IDENTIFIER_SIZE];
    if file_id != EXPECTED_FILE_ID {
        return HeaderStatus::NotPresent;
    }

    HeaderStatus::CompatibleVersion
}

pub fn parse_extended_header(data: &[u8]) -> Result<Option<ExtendedHeader>> {
    if data.len() < EXTENDED_HEADER_OFFSET + EXTENDED_HEADER_STRUCT_SIZE {
        return Ok(None);
    }

    let magic_start = EXTENDED_HEADER_OFFSET;
    let magic = &data[magic_start..magic_start + 4];
    if magic != EXTENDED_HEADER_MAGIC {
        return Ok(None);
    }

    let header_data = &data[magic_start + 4..];
    if header_data.len() < 4 + 8 + 8 {
        return Err(Error::InvalidProgram);
    }

    let header_size = u32::from_le_bytes([
        header_data[0],
        header_data[1],
        header_data[2],
        header_data[3],
    ]);
    if (header_size as usize) < 8 + 8 {
        return Err(Error::InvalidProgram);
    }

    let program_size = u64::from_le_bytes([
        header_data[4],
        header_data[5],
        header_data[6],
        header_data[7],
        header_data[8],
        header_data[9],
        header_data[10],
        header_data[11],
    ]);

    let segment_base_offset = u64::from_le_bytes([
        header_data[12],
        header_data[13],
        header_data[14],
        header_data[15],
        header_data[16],
        header_data[17],
        header_data[18],
        header_data[19],
    ]);

    Ok(Some(ExtendedHeader {
        program_size,
        segment_base_offset,
    }))
}

pub fn get_flatbuffer_start(data: &[u8]) -> Result<usize> {
    match parse_extended_header(data)? {
        Some(_) => {
            Ok(EXTENDED_HEADER_OFFSET + EXTENDED_HEADER_STRUCT_SIZE)
        }
        None => Ok(0),
    }
}
