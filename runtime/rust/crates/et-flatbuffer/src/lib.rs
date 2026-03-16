// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#![no_std]
#![allow(
    unused_imports,
    dead_code,
    clippy::all,
    non_camel_case_types,
    non_snake_case,
    deprecated
)]

#[path = "generated/scalar_type_generated.rs"]
pub mod scalar_type_generated;

#[path = "generated/program_generated.rs"]
pub mod program_generated;

pub mod header;

pub use program_generated::executorch_flatbuffer;
pub use scalar_type_generated::executorch_flatbuffer as scalar_type_fb;
