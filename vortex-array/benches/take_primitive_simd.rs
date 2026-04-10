// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Raw buffer-level benchmarks for the primitive take (gather) kernel.
//!
//! Compares scalar, AVX2, and Mojo SIMD gather. All three are called through
//! the same `fn(&[T], &[u32]) -> Buffer<T>` Rust signature on raw slices.
//!
//! Run with: `cargo bench -p vortex-array --bench take_primitive_simd`

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::Uniform;
use rand::prelude::*;
use vortex_array::arrays::primitive::{bench_take_avx2, bench_take_mojo, bench_take_scalar};

fn main() {
    divan::main();
}

const NUM_INDICES: &[usize] = &[1_000, 10_000, 100_000];
const NUM_VALUES: usize = 65_536;

fn make_u32_indices(num_indices: usize) -> Vec<u32> {
    let rng = StdRng::seed_from_u64(42);
    let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
    rng.sample_iter(range).take(num_indices).collect()
}

// ---------------------------------------------------------------------------
// u32 values
// ---------------------------------------------------------------------------

#[divan::bench(args = NUM_INDICES, sample_count = 10_000)]
fn gather_u32_scalar(bencher: Bencher, n: usize) {
    let values: Vec<u32> = (0..NUM_VALUES as u32).collect();
    let indices = make_u32_indices(n);
    bencher.bench(|| divan::black_box(bench_take_scalar(&values, &indices)));
}

#[divan::bench(args = NUM_INDICES, sample_count = 10_000)]
fn gather_u32_avx2(bencher: Bencher, n: usize) {
    let values: Vec<u32> = (0..NUM_VALUES as u32).collect();
    let indices = make_u32_indices(n);
    bencher.bench(|| divan::black_box(bench_take_avx2(&values, &indices)));
}

#[divan::bench(args = NUM_INDICES, sample_count = 10_000)]
fn gather_u32_mojo(bencher: Bencher, n: usize) {
    let values: Vec<u32> = (0..NUM_VALUES as u32).collect();
    let indices = make_u32_indices(n);
    bencher.bench(|| divan::black_box(bench_take_mojo(&values, &indices)));
}

// ---------------------------------------------------------------------------
// u64 values
// ---------------------------------------------------------------------------

#[divan::bench(args = NUM_INDICES, sample_count = 10_000)]
fn gather_u64_scalar(bencher: Bencher, n: usize) {
    let values: Vec<u64> = (0..NUM_VALUES as u64).map(|i| i * 100).collect();
    let indices = make_u32_indices(n);
    bencher.bench(|| divan::black_box(bench_take_scalar(&values, &indices)));
}

#[divan::bench(args = NUM_INDICES, sample_count = 10_000)]
fn gather_u64_avx2(bencher: Bencher, n: usize) {
    let values: Vec<u64> = (0..NUM_VALUES as u64).map(|i| i * 100).collect();
    let indices = make_u32_indices(n);
    bencher.bench(|| divan::black_box(bench_take_avx2(&values, &indices)));
}

#[divan::bench(args = NUM_INDICES, sample_count = 10_000)]
fn gather_u64_mojo(bencher: Bencher, n: usize) {
    let values: Vec<u64> = (0..NUM_VALUES as u64).map(|i| i * 100).collect();
    let indices = make_u32_indices(n);
    bencher.bench(|| divan::black_box(bench_take_mojo(&values, &indices)));
}
