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

    // ExecuTorch repo root is four levels up: crates/et-kernel-ffi -> crates -> rust -> runtime -> repo
    let repo_root = crate_dir
        .parent() // crates
        .and_then(|p| p.parent()) // rust
        .and_then(|p| p.parent()) // runtime
        .and_then(|p| p.parent()) // repo root
        .expect("cannot determine repo root");

    // Default to cmake-out; override with EXECUTORCH_BUILD_DIR
    let build_dir = env::var("EXECUTORCH_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join("cmake-out"));

    let lib_dir = build_dir.join("lib");

    // Headers use #include <executorch/runtime/...>, resolved from repo parent.
    // <c10/util/irange.h> is shimmed from the portable_type/c10 directory.
    let include_root = repo_root.parent().expect("repo root has no parent");
    let c10_shim = repo_root.join("runtime/core/portable_type/c10");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .define("C10_USING_CUSTOM_GENERATED_MACROS", None)
        .file(crate_dir.join("csrc/kernel_shim.cpp"))
        .include(include_root)
        .include(&c10_shim)
        .compile("kernel_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=executorch");
    println!("cargo:rustc-link-lib=static=executorch_core");

    println!("cargo:rustc-link-lib=static=portable_ops_lib");
    println!("cargo:rustc-link-lib=static=portable_kernels");

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed=csrc/kernel_shim.h");
    println!("cargo:rerun-if-changed=csrc/kernel_shim.cpp");
    println!("cargo:rerun-if-env-changed=EXECUTORCH_BUILD_DIR");
}
