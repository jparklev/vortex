// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST compression entry points.
//!
//! [`fsst_compress`] / [`fsst_train_compressor`] take `&VarBinViewArray` and are
//! the production path. [`fsst_compress_varbin`] / [`fsst_train_compressor_varbin`]
//! take `&VarBinArray` and are gated behind the `_test-harness` feature for use
//! by bench and test fixtures.

use fsst::Compressor;
#[cfg(any(test, feature = "_test-harness"))]
use num_traits::AsPrimitive;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
#[cfg(any(test, feature = "_test-harness"))]
use vortex_array::arrays::PrimitiveArray;
#[cfg(any(test, feature = "_test-harness"))]
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
#[cfg(any(test, feature = "_test-harness"))]
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
#[cfg(any(test, feature = "_test-harness"))]
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::FSST;
use crate::FSSTArray;

/// FSST worst case: every input byte expands to an escape + literal (2x) plus a small
/// per-string header.
const FSST_PER_BYTE_OVERHEAD: usize = 2;
const FSST_PER_ROW_OVERHEAD: usize = 7;

/// Starting capacity for the per-row `compress_into` scratch buffer; grown monotonically.
const DEFAULT_BUFFER_LEN: usize = 1024 * 1024;

/// Compress a [`VarBinViewArray`] using FSST.
pub fn fsst_compress(
    strings: &VarBinViewArray,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTArray> {
    let mask = strings.validity()?.execute_mask(strings.len(), ctx)?;
    let views = strings.views();
    let non_null = mask.true_count();

    let total_input_bytes = match mask.bit_buffer() {
        AllOr::All => views.iter().map(|v| v.len() as usize).sum(),
        AllOr::None => 0,
        AllOr::Some(bits) => views
            .iter()
            .zip(bits.iter())
            .filter(|&(_, b)| b)
            .map(|(v, _)| v.len() as usize)
            .sum(),
    };

    if fsst_output_fits_in_i32_offsets(total_input_bytes, non_null) {
        compress_views::<i32>(strings, &mask, compressor, ctx)
    } else {
        compress_views::<i64>(strings, &mask, compressor, ctx)
    }
}

/// Compress a [`VarBinArray`] using FSST.
#[cfg(any(test, feature = "_test-harness"))]
pub fn fsst_compress_varbin(
    strings: &VarBinArray,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTArray> {
    let mask = strings.validity()?.execute_mask(strings.len(), ctx)?;
    let offsets = strings.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let total_input_bytes = match_each_integer_ptype!(offsets.ptype(), |O| {
        let off = offsets.as_slice::<O>();
        let first: usize = off[0].as_();
        let last: usize = off[off.len() - 1].as_();
        last - first
    });
    let non_null = mask.true_count();

    if fsst_output_fits_in_i32_offsets(total_input_bytes, non_null) {
        compress_varbin::<i32>(strings, &offsets, &mask, compressor, ctx)
    } else {
        compress_varbin::<i64>(strings, &offsets, &mask, compressor, ctx)
    }
}

/// Train an FSST [`Compressor`] from a [`VarBinViewArray`]'s non-null rows.
pub fn fsst_train_compressor(
    strings: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Compressor> {
    let mask = strings.validity()?.execute_mask(strings.len(), ctx)?;
    let views = strings.views();
    let mut lines: Vec<&[u8]> = Vec::with_capacity(views.len());

    match mask.bit_buffer() {
        AllOr::All => {
            for view in views {
                lines.push(view_bytes(strings, view));
            }
        }
        AllOr::None => {}
        AllOr::Some(bits) => {
            for (view, valid) in views.iter().zip(bits.iter()) {
                if valid {
                    lines.push(view_bytes(strings, view));
                }
            }
        }
    }

    Ok(Compressor::train(&lines))
}

/// Train an FSST [`Compressor`] from a [`VarBinArray`]'s non-null rows.
#[cfg(any(test, feature = "_test-harness"))]
pub fn fsst_train_compressor_varbin(
    strings: &VarBinArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Compressor> {
    let mask = strings.validity()?.execute_mask(strings.len(), ctx)?;
    let offsets = strings.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let bytes = strings.bytes().as_slice();
    let mut lines: Vec<&[u8]> = Vec::with_capacity(strings.len());

    match_each_integer_ptype!(offsets.ptype(), |I| {
        let off = offsets.as_slice::<I>();
        for_each_varbin_row(off, bytes, &mask, |row| {
            if let Some(s) = row {
                lines.push(s);
            }
        });
    });

    Ok(Compressor::train(&lines))
}

#[inline]
fn fsst_output_fits_in_i32_offsets(total_input_bytes: usize, non_null: usize) -> bool {
    let worst = total_input_bytes
        .saturating_mul(FSST_PER_BYTE_OVERHEAD)
        .saturating_add(non_null.saturating_mul(FSST_PER_ROW_OVERHEAD));
    worst <= i32::MAX as usize
}

#[inline]
fn view_bytes<'a>(strings: &'a VarBinViewArray, view: &'a BinaryView) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let r = view.as_view();
        &strings.buffer(r.buffer_index as usize).as_slice()[r.as_range()]
    }
}

fn compress_views<O>(
    strings: &VarBinViewArray,
    mask: &Mask,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTArray>
where
    O: IntegerPType + 'static,
{
    let mut sink = FsstSink::<O>::with_capacity(strings.len(), compressor);
    let views = strings.views();
    match mask.bit_buffer() {
        AllOr::All => {
            for view in views {
                sink.emit(Some(view_bytes(strings, view)));
            }
        }
        AllOr::None => {
            for _ in 0..mask.len() {
                sink.emit(None);
            }
        }
        AllOr::Some(bits) => {
            for (view, valid) in views.iter().zip(bits.iter()) {
                sink.emit(valid.then(|| view_bytes(strings, view)));
            }
        }
    }
    sink.finish(strings.dtype().clone(), ctx)
}

#[cfg(any(test, feature = "_test-harness"))]
fn compress_varbin<O>(
    strings: &VarBinArray,
    offsets: &PrimitiveArray,
    mask: &Mask,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTArray>
where
    O: IntegerPType + 'static,
{
    let mut sink = FsstSink::<O>::with_capacity(strings.len(), compressor);
    let bytes = strings.bytes().as_slice();
    match_each_integer_ptype!(offsets.ptype(), |I| {
        let off = offsets.as_slice::<I>();
        for_each_varbin_row(off, bytes, mask, |row| sink.emit(row));
    });
    sink.finish(strings.dtype().clone(), ctx)
}

/// Call `f` once per row of a `VarBinArray` with the row bytes or `None`.
/// Validity dispatch is hoisted out of the per-row loop.
#[cfg(any(test, feature = "_test-harness"))]
#[inline]
fn for_each_varbin_row<'a, I, F>(off: &[I], bytes: &'a [u8], mask: &Mask, mut f: F)
where
    I: IntegerPType + 'static,
    F: FnMut(Option<&'a [u8]>),
{
    match mask.bit_buffer() {
        AllOr::All => {
            for w in off.windows(2) {
                f(Some(&bytes[w[0].as_()..w[1].as_()]));
            }
        }
        AllOr::None => {
            for _ in 0..mask.len() {
                f(None);
            }
        }
        AllOr::Some(bits) => {
            for (w, valid) in off.windows(2).zip(bits.iter()) {
                f(valid.then(|| &bytes[w[0].as_()..w[1].as_()]));
            }
        }
    }
}

/// Per-row output state for an FSST compression pass.
struct FsstSink<'c, O: IntegerPType + 'static> {
    buffer: Vec<u8>,
    builder: VarBinBuilder<O>,
    uncompressed_lengths: BufferMut<i32>,
    compressor: &'c Compressor,
}

impl<'c, O: IntegerPType + 'static> FsstSink<'c, O> {
    fn with_capacity(len: usize, compressor: &'c Compressor) -> Self {
        Self {
            buffer: Vec::with_capacity(DEFAULT_BUFFER_LEN),
            builder: VarBinBuilder::<O>::with_capacity(len),
            uncompressed_lengths: BufferMut::with_capacity(len),
            compressor,
        }
    }

    #[inline]
    fn emit(&mut self, row: Option<&[u8]>) {
        let Some(s) = row else {
            self.builder.append_null();
            self.uncompressed_lengths.push(0);
            return;
        };

        // A single row > i32::MAX (2 GiB) is not supported.
        self.uncompressed_lengths.push(
            i32::try_from(s.len()).vortex_expect("per-row uncompressed length must fit in i32"),
        );

        let target = FSST_PER_BYTE_OVERHEAD * s.len() + FSST_PER_ROW_OVERHEAD;
        if target > self.buffer.len() {
            self.buffer.reserve(target - self.buffer.len());
        }

        // SAFETY: `self.buffer` has capacity for the FSST worst-case output of `s`.
        unsafe { self.compressor.compress_into(s, &mut self.buffer) };

        self.builder.append_value(&self.buffer);
    }

    fn finish(self, dtype: DType, ctx: &mut ExecutionCtx) -> VortexResult<FSSTArray> {
        let codes = self.builder.finish(DType::Binary(dtype.nullability()));
        FSST::try_new(
            dtype,
            Buffer::copy_from(self.compressor.symbol_table()),
            Buffer::<u8>::copy_from(self.compressor.symbol_lengths()),
            codes,
            self.uncompressed_lengths.into_array(),
            ctx,
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::varbin::VarBinArrayExt;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;

    use super::fsst_compress;
    use super::fsst_output_fits_in_i32_offsets;
    use super::fsst_train_compressor;
    use crate::array::FSSTArrayExt;

    /// Regression for #7833: the i32-vs-i64 codes-offsets decision must cross at
    /// `i32::MAX` against the worst-case bound `2 * total + 7 * non_null`.
    #[test]
    fn offset_width_boundary() {
        let m = i32::MAX as usize;
        assert!(fsst_output_fits_in_i32_offsets(m / 2 - 7, 1));
        assert!(!fsst_output_fits_in_i32_offsets(m / 2, 1));
        assert!(fsst_output_fits_in_i32_offsets(0, 0));
        assert!(!fsst_output_fits_in_i32_offsets(usize::MAX, 1));
    }

    /// Small inputs fit the i32 bound, so `fsst_compress` must pick i32 offsets.
    /// The i64 branch is covered by `tests::fsst_compress_offsets_overflow_i32`.
    #[test]
    fn codes_offsets_dtype_small_input_is_i32() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_str(["hello", "world", "fsst encoded"]);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let compressor = fsst_train_compressor(&array, &mut ctx)?;
        let fsst = fsst_compress(&array, &compressor, &mut ctx)?;
        assert_eq!(fsst.codes().offsets().dtype().as_ptype(), PType::I32);
        Ok(())
    }
}
