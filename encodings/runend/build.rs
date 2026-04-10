// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

//! Build script for vortex-runend.
//!
//! Compiles the shared Mojo SIMD kernel (which includes run-end decode functions)
//! and links it as a static library. The `vortex_mojo` cfg flag is emitted so
//! Rust code can conditionally use the Mojo decode path.

use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // The Mojo kernel lives alongside this crate.
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let kernel_src = Path::new(&manifest_dir).join("kernels/decode.mojo");

    println!("cargo:rerun-if-changed={}", kernel_src.display());

    let mojo_bin = find_mojo();
    let mojo_bin = match mojo_bin {
        Some(p) => p,
        None => return,
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let obj_path = out_dir.join("vortex_mojo_runend.o");

    // Use MOJO_MCPU to control target CPU (defaults to "native").
    // CI sets this to "skylake" for vpgatherqd and SIMD broadcast.
    let mcpu = env::var("MOJO_MCPU").unwrap_or_else(|_| "native".to_owned());

    // On macOS, Mojo's host detection works correctly but it rejects the Cargo
    // triple format and "native" CPU also triggers the broken host triple detection.
    // Skip both flags on Apple targets; Mojo auto-detects correctly without them.
    let is_apple = env::var("TARGET")
        .map(|t| t.contains("apple"))
        .unwrap_or(false);
    let target_triple = env::var("TARGET")
        .ok()
        .filter(|_| !is_apple);

    let mut cmd = Command::new(&mojo_bin);
    cmd.arg("build").arg("--emit").arg("object");

    if !is_apple || mcpu != "native" {
        cmd.arg("--mcpu").arg(&mcpu).arg("--mtune").arg(&mcpu);
    }

    if let Some(triple) = &target_triple {
        cmd.arg("--target-triple").arg(triple);
    }

    let status = cmd.arg("-o").arg(&obj_path).arg(&kernel_src).status();

    let status = match status {
        Ok(s) => s,
        Err(e) => {
            println!("cargo:warning=Mojo compilation failed to launch: {e}");
            return;
        }
    };

    if !status.success() {
        println!(
            "cargo:warning=Mojo AOT compilation failed (exit {}), falling back to Rust decode",
            status
        );
        return;
    }

    let lib_path = out_dir.join("libvortex_mojo_runend.a");
    let ar_status = Command::new("ar")
        .args(["rcs"])
        .arg(&lib_path)
        .arg(&obj_path)
        .status();

    match ar_status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=ar failed (exit {s}), falling back to Rust decode");
            return;
        }
        Err(e) => {
            println!("cargo:warning=ar not found: {e}, falling back to Rust decode");
            return;
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=vortex_mojo_runend");
    println!("cargo:rustc-cfg=vortex_mojo");
}

fn find_mojo() -> Option<PathBuf> {
    if Command::new("mojo")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        return Some(PathBuf::from("mojo"));
    }

    if let Ok(home) = env::var("HOME") {
        let pip_mojo = PathBuf::from(home).join(".local/bin/mojo");
        if pip_mojo.exists()
            && Command::new(&pip_mojo)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        {
            return Some(pip_mojo);
        }
    }

    None
}
