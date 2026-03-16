use et_core::error::{Error, Result};
use et_flatbuffer::executorch_flatbuffer::Program as FBProgram;
use et_flatbuffer::header;

use crate::data_loader::{DataLoader, FreeableBuffer};
use crate::method_meta::MethodMeta;

extern crate flatbuffers;

pub struct Program<L: DataLoader> {
    loader: L,
    program_data: FreeableBuffer,
    constant_segment_data: FreeableBuffer,
    segment_base_offset: u64,
}

impl<L: DataLoader> Program<L> {
    pub fn load(loader: L) -> Result<Self> {
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

        let (program_offset, program_size, segment_base_offset) = match extended {
            Some(eh) => {
                // The program data starts after the extended header.
                // Extended header: eh_magic(4) + eh_size(4) + program_size(8) + segment_base_offset(8) = 24
                // Starts at offset 8 (after flatbuffer offset + file identifier)
                let fb_offset = 8 + 24;
                (fb_offset, eh.program_size as usize, eh.segment_base_offset)
            }
            None => (0, file_size, 0),
        };

        let program_data = loader.load(program_offset, program_size)?;

        let constant_segment_data = FreeableBuffer::empty();

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
        flatbuffers::root::<FBProgram>(data).map_err(|_| Error::InvalidProgram)
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

    pub fn loader(&self) -> &L {
        &self.loader
    }
}
