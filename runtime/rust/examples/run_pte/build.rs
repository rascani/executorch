// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_dir = PathBuf::from(&manifest_dir);

    let repo_root = crate_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("cannot determine repo root");

    let build_dir = env::var("EXECUTORCH_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join("cmake-out"));

    let lib_dir = build_dir.join("lib");

    // portable_ops_lib uses static constructors to register kernels.
    // Force-load ensures the linker doesn't strip the unreferenced
    // registration object files.
    if cfg!(target_os = "macos") {
        println!(
            "cargo:rustc-link-arg=-Wl,-force_load,{}/libportable_ops_lib.a",
            lib_dir.display()
        );
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!(
            "cargo:rustc-link-arg={}/libportable_ops_lib.a",
            lib_dir.display()
        );
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
    }

    println!("cargo:rerun-if-env-changed=EXECUTORCH_BUILD_DIR");
}
