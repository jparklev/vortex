// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core slice-level filtering algorithms.
//!
//! Provides both immutable and mutable (in-place) filtering of typed slices by various mask
//! representations: indices and ranges (slices).

use std::mem::size_of;
use std::ptr;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::MaskIter;
use vortex_mask::MaskValues;

// This is modeled after the constant with the equivalent name in arrow-rs.
pub(super) const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

// ---------------------------------------------------------------------------
// Immutable slice filtering
// ---------------------------------------------------------------------------

/// Filter a slice by [`MaskValues`], dispatching to the indices or slices path based on a
/// selectivity threshold.
pub(super) fn filter_slice_by_mask_values<T: Copy>(slice: &[T], mask: &MaskValues) -> Buffer<T> {
    assert_eq!(
        mask.len(),
        slice.len(),
        "Selection mask length must equal the buffer length"
    );

    match mask.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
        MaskIter::Indices(indices) => filter_slice_by_indices(slice, indices),
        MaskIter::Slices(slices) => filter_slice_by_slices(slice, slices),
    }
}

/// Filter a slice by a set of strictly increasing indices.
fn filter_slice_by_indices<T: Copy>(slice: &[T], indices: &[usize]) -> Buffer<T> {
    #[cfg(vortex_mojo)]
    {
        if let Some(buf) = mojo::filter_by_indices_mojo(slice, indices) {
            return buf;
        }
    }

    Buffer::<T>::from_trusted_len_iter(indices.iter().map(|&idx| slice[idx]))
}

#[cfg(vortex_mojo)]
mod mojo {
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;

    use super::size_of;

    unsafe extern "C" {
        fn vortex_filter_1byte(src: usize, idx: usize, dst: usize, n: usize);
        fn vortex_filter_2byte(src: usize, idx: usize, dst: usize, n: usize);
        fn vortex_filter_4byte(src: usize, idx: usize, dst: usize, n: usize);
        fn vortex_filter_8byte(src: usize, idx: usize, dst: usize, n: usize);
    }

    /// SIMD gather for the filter-by-indices path. Returns `None` for unsupported
    /// element sizes so the caller falls back to scalar.
    pub(super) fn filter_by_indices_mojo<T: Copy>(
        slice: &[T],
        indices: &[usize],
    ) -> Option<Buffer<T>> {
        let kernel: unsafe extern "C" fn(usize, usize, usize, usize) = match size_of::<T>() {
            1 => vortex_filter_1byte,
            2 => vortex_filter_2byte,
            4 => vortex_filter_4byte,
            8 => vortex_filter_8byte,
            _ => return None,
        };

        let len = indices.len();
        let mut buffer = BufferMut::<T>::with_capacity(len);
        let dst = buffer.spare_capacity_mut().as_mut_ptr().cast::<T>();

        // SAFETY: The Mojo kernel reads `len` indices from `indices`, gathers from
        // `slice`, and writes `len` elements to `dst`. All pointers are valid.
        unsafe {
            kernel(
                slice.as_ptr() as usize,
                indices.as_ptr() as usize,
                dst as usize,
                len,
            );
            buffer.set_len(len);
        }

        Some(buffer.freeze())
    }
}

/// Filter a slice by a set of strictly increasing `(start, end)` ranges.
fn filter_slice_by_slices<T: Copy>(slice: &[T], slices: &[(usize, usize)]) -> Buffer<T> {
    let output_len: usize = slices.iter().map(|(start, end)| end - start).sum();

    let mut out = BufferMut::<T>::with_capacity(output_len);
    for (start, end) in slices {
        out.extend_from_slice(&slice[*start..*end]);
    }

    out.freeze()
}

// ---------------------------------------------------------------------------
// Mutable (in-place) slice filtering
// ---------------------------------------------------------------------------

/// Filter a mutable slice in-place by [`MaskValues`], returning the new valid length.
///
/// We always use the slices path here because iterating over indices will have strictly more
/// loop iterations than slices (more branches), and the overhead of batched `ptr::copy(len)` is
/// not that high.
pub(super) fn filter_slice_mut_by_mask_values<T: Copy>(
    slice: &mut [T],
    mask: &MaskValues,
) -> usize {
    assert_eq!(
        slice.len(),
        mask.len(),
        "Mask length must equal the slice length"
    );

    filter_slice_mut_by_slices(slice, mask.slices())
}

/// Filter a mutable slice in-place by a set of `(start, end)` ranges, returning the new length.
fn filter_slice_mut_by_slices<T: Copy>(slice: &mut [T], slices: &[(usize, usize)]) -> usize {
    let mut write_pos = 0;

    // For each range in the selection, copy all of the elements to the current write position.
    for &(start, end) in slices {
        let len = end - start;

        // SAFETY: Slices should be within bounds.
        unsafe {
            ptr::copy(
                slice.as_ptr().add(start),
                slice.as_mut_ptr().add(write_pos),
                len,
            )
        };

        write_pos += len;
    }

    write_pos
}
