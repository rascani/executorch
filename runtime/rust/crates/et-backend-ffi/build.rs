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

    let include_root = repo_root.parent().expect("repo root has no parent");
    let c10_shim = repo_root.join("runtime/core/portable_type/c10");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .define("C10_USING_CUSTOM_GENERATED_MACROS", None)
        .file(crate_dir.join("csrc/backend_shim.cpp"))
        .include(include_root)
        .include(&c10_shim)
        .compile("backend_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=executorch");
    println!("cargo:rustc-link-lib=static=executorch_core");

    if std::env::var("CARGO_FEATURE_XNNPACK").is_ok() {
        let xnnpack_dir = build_dir.join("backends/xnnpack");
        println!(
            "cargo:rustc-link-search=native={}",
            xnnpack_dir.display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            xnnpack_dir.join("third-party/XNNPACK").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            xnnpack_dir.join("third-party/cpuinfo").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            xnnpack_dir.join("third-party/pthreadpool").display()
        );
        println!("cargo:rustc-link-lib=static=xnnpack_backend");
        println!("cargo:rustc-link-lib=static=XNNPACK");
        println!("cargo:rustc-link-lib=static=xnnpack-microkernels-prod");
        println!("cargo:rustc-link-lib=static=cpuinfo");
        println!("cargo:rustc-link-lib=static=pthreadpool");
        println!("cargo:rustc-link-lib=static=extension_threadpool");

        let kleidiai_dir = build_dir.join("kleidiai");
        if kleidiai_dir.join("libkleidiai.a").exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                kleidiai_dir.display()
            );
            println!("cargo:rustc-link-lib=static=kleidiai");
        }
    }

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed=csrc/backend_shim.h");
    println!("cargo:rerun-if-changed=csrc/backend_shim.cpp");
    println!("cargo:rerun-if-env-changed=EXECUTORCH_BUILD_DIR");
}
