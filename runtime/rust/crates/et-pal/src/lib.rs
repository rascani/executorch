// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#![no_std]

pub trait Platform {
    fn current_ticks(&self) -> u64;
    fn ticks_to_ns(&self, ticks: u64) -> u64;
}
