// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

//! Build script for vortex-array.
//!
//! When the Mojo SDK is installed, this compiles the SIMD take kernels in `kernels/take.mojo`
//! ahead-of-time into a static library and links it into the crate. The `vortex_mojo` cfg flag
//! is emitted so that Rust code can conditionally enable the Mojo take path.
//!
//! When Mojo is **not** available the build script is a no-op and the existing Rust SIMD kernels
//! are used instead.

use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");

    let mojo_bin = find_mojo();
    let mojo_bin = match mojo_bin {
        Some(p) => p,
        None => return,
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let kernel_src = Path::new(&manifest_dir).join("kernels/take.mojo");

    let obj_path = out_dir.join("vortex_mojo_take.o");

    // AOT compile the Mojo kernel to a native object file.
    //
    // Use MOJO_MCPU to override the target CPU (defaults to "native"). In CI the runner
    // CPU may differ from the build host, so we allow pinning to a baseline like
    // "x86-64-v3" (AVX2) to avoid emitting unsupported instructions (e.g. AVX-512).
    let mcpu = env::var("MOJO_MCPU").unwrap_or_else(|_| "native".to_owned());

    // Cargo sets TARGET to e.g. "x86_64-unknown-linux-gnu". Pass it through so Mojo
    // doesn't fail with "unknown target triple" when the build env differs from the host.
    let target_triple = env::var("TARGET").ok();

    let mut cmd = Command::new(&mojo_bin);
    cmd.arg("build")
        .arg("--emit")
        .arg("object")
        .arg("--mcpu")
        .arg(&mcpu);

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
            "cargo:warning=Mojo AOT compilation failed (exit {}), falling back to Rust SIMD kernels",
            status
        );
        return;
    }

    // Archive the object file into a static library that Cargo can link.
    let lib_path = out_dir.join("libvortex_mojo_take.a");
    let ar_status = Command::new("ar")
        .args(["rcs"])
        .arg(&lib_path)
        .arg(&obj_path)
        .status();

    match ar_status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=ar failed (exit {s}), falling back to Rust SIMD kernels");
            return;
        }
        Err(e) => {
            println!("cargo:warning=ar not found: {e}, falling back to Rust SIMD kernels");
            return;
        }
    }

    // Tell Cargo to link the static library.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=vortex_mojo_take");

    // Enable the cfg flag so Rust code can use the Mojo kernels.
    println!("cargo:rustc-cfg=vortex_mojo");
}

/// Searches for the Mojo compiler binary. Checks `PATH` first, then the common
/// pip-installed location (`~/.local/bin/mojo`).
fn find_mojo() -> Option<PathBuf> {
    // Check PATH first.
    if Command::new("mojo")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        return Some(PathBuf::from("mojo"));
    }

    // Pip installs mojo to ~/.local/bin on Linux.
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
