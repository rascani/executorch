// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use et_core::error::{Error, Result};
use et_flatbuffer::executorch_flatbuffer::Program as FBProgram;
use et_flatbuffer::header;

use crate::data_loader::{DataLoader, FreeableBuffer};
use crate::method_meta::MethodMeta;

extern crate flatbuffers;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verification {
    Minimal,
    InternalConsistency,
}

pub struct Program<L: DataLoader> {
    loader: L,
    program_data: FreeableBuffer,
    constant_segment_data: FreeableBuffer,
    segment_base_offset: u64,
}

impl<L: DataLoader> Program<L> {
    pub fn load(loader: L) -> Result<Self> {
        Self::load_with_verification(loader, Verification::Minimal)
    }

    pub fn load_with_verification(loader: L, verification: Verification) -> Result<Self> {
        let file_size = loader.size()?;
        if file_size < header::MIN_HEAD_BYTES {
            return Err(Error::InvalidProgram);
        }

        let head_data = loader.load(0, core::cmp::min(file_size, header::MIN_HEAD_BYTES))?;

        let status = header::check_header(head_data.as_slice());
        if status != header::HeaderStatus::CompatibleVersion {
            return Err(Error::InvalidProgram);
        }

        let extended = header::parse_extended_header(head_data.as_slice())?;
        drop(head_data);

        // The extended header is spliced into the flatbuffer at offset 8,
        // and the root offset at bytes [0..4] is adjusted to compensate.
        // So the entire range [0, program_size) IS the flatbuffer data.
        let (program_size, segment_base_offset) = match extended {
            Some(eh) => (eh.program_size as usize, eh.segment_base_offset),
            None => (file_size, 0),
        };

        let program_data = loader.load(0, program_size)?;

        // TODO: The Rust flatbuffers verifier rejects PTE files with unaligned
        // i64 fields (e.g. IntList items vectors at 4-byte-aligned offsets).
        // This is a spec violation in ExecuTorch's PTE serializer — the actual
        // read path handles unaligned access safely via copy_nonoverlapping.
        // Options: fix the serializer upstream, or patch/fork the flatbuffers
        // crate verifier to skip alignment checks.
        if verification == Verification::InternalConsistency {
            flatbuffers::root::<FBProgram>(program_data.as_slice())
                .map_err(|_| Error::InvalidProgram)?;
        }

        // Load the constant segment if present.
        let constant_segment_data = {
            let prog = unsafe {
                flatbuffers::root_unchecked::<FBProgram>(program_data.as_slice())
            };
            let cs = prog.constant_segment();
            let has_constants = cs
                .and_then(|s| s.offsets())
                .map_or(false, |o| o.len() > 1);
            if has_constants {
                let cs = cs.unwrap();
                let segments = prog.segments().ok_or(Error::InvalidProgram)?;
                let seg_idx = cs.segment_index() as usize;
                if seg_idx >= segments.len() {
                    return Err(Error::InvalidProgram);
                }
                let seg = segments.get(seg_idx);
                let offset = segment_base_offset + seg.offset();
                let size = seg.size_() as usize;
                loader.load(offset as usize, size)?
            } else {
                FreeableBuffer::empty()
            }
        };

        Ok(Program {
            loader,
            program_data,
            constant_segment_data,
            segment_base_offset,
        })
    }

    fn internal_program(&self) -> Result<FBProgram<'_>> {
        let data = self.program_data.as_slice();
        if data.is_empty() {
            return Err(Error::InvalidProgram);
        }
        Ok(unsafe { flatbuffers::root_unchecked::<FBProgram>(data) })
    }

    pub fn num_methods(&self) -> Result<usize> {
        let prog = self.internal_program()?;
        Ok(prog.execution_plan().map_or(0, |plans| plans.len()))
    }

    pub fn get_method_name(&self, index: usize) -> Result<&str> {
        let prog = self.internal_program()?;
        let plans = prog.execution_plan().ok_or(Error::NotFound)?;
        if index >= plans.len() {
            return Err(Error::InvalidArgument);
        }
        plans.get(index).name().ok_or(Error::InvalidProgram)
    }

    pub fn method_meta(&self, name: &str) -> Result<MethodMeta<'_>> {
        let prog = self.internal_program()?;
        let plans = prog.execution_plan().ok_or(Error::NotFound)?;
        for i in 0..plans.len() {
            let plan = plans.get(i);
            if plan.name() == Some(name) {
                return Ok(MethodMeta::new(plan));
            }
        }
        Err(Error::NotFound)
    }

    pub fn program_data(&self) -> &[u8] {
        self.program_data.as_slice()
    }

    pub fn constant_segment_data(&self) -> &[u8] {
        self.constant_segment_data.as_slice()
    }

    pub fn segment_base_offset(&self) -> u64 {
        self.segment_base_offset
    }

    pub fn get_constant_buffer_data(
        &self,
        buffer_index: usize,
        nbytes: usize,
    ) -> Result<*const u8> {
        let prog = self.internal_program()?;
        let cs_data = self.constant_segment_data.as_slice();
        if !cs_data.is_empty() {
            let cs = prog
                .constant_segment()
                .ok_or(Error::InvalidProgram)?;
            let offsets = cs.offsets().ok_or(Error::InvalidProgram)?;
            if buffer_index >= offsets.len() {
                return Err(Error::InvalidArgument);
            }
            let offset = offsets.get(buffer_index) as usize;
            if offset + nbytes > cs_data.len() {
                return Err(Error::InvalidArgument);
            }
            Ok(cs_data.as_ptr().wrapping_add(offset))
        } else {
            // Legacy: constant data in flatbuffer's constant_buffer field
            let cb = prog.constant_buffer().ok_or(Error::InvalidProgram)?;
            if buffer_index >= cb.len() {
                return Err(Error::InvalidArgument);
            }
            let storage = cb.get(buffer_index).storage().ok_or(Error::InvalidProgram)?;
            if nbytes > storage.len() {
                return Err(Error::InvalidArgument);
            }
            Ok(storage.bytes().as_ptr())
        }
    }

    pub fn get_backend_delegate_data(
        &self,
        index: usize,
    ) -> Result<(*const u8, usize)> {
        let prog = self.internal_program()?;
        let data_list = prog
            .backend_delegate_data()
            .ok_or(Error::InvalidProgram)?;
        if index >= data_list.len() {
            return Err(Error::NotFound);
        }
        let data = data_list
            .get(index)
            .data()
            .ok_or(Error::InvalidProgram)?;
        Ok((data.bytes().as_ptr(), data.len()))
    }

    pub fn load_segment(&self, segment_index: usize) -> Result<FreeableBuffer> {
        if self.segment_base_offset == 0 {
            return Err(Error::NotFound);
        }
        let prog = self.internal_program()?;
        let segments = prog.segments().ok_or(Error::NotFound)?;
        if segment_index >= segments.len() {
            return Err(Error::NotFound);
        }
        let seg = segments.get(segment_index);
        let offset = self.segment_base_offset + seg.offset();
        let size = seg.size_() as usize;
        self.loader.load(offset as usize, size)
    }

    pub fn loader(&self) -> &L {
        &self.loader
    }
}
