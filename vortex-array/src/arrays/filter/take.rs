// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::Filter;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::dict::TakeExecute;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::executor::ExecutionCtx;
use crate::kernel::ParentKernelSet;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::validity::Validity;

pub(super) const PARENT_KERNELS: ParentKernelSet<Filter> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(Filter))]);

const NULLABLE_FIXED_WIDTH_FULL_TAKE_FALLBACK_LEN: usize = 4096;

fn take_impl(
    array: ArrayView<'_, Filter>,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let indices_validity = indices.validity()?.execute_mask(indices.len(), ctx)?;
    if let Some(taken) = take_primitive_fast_path(array, indices, &indices_validity)? {
        return Ok(taken);
    }
    if let Some(taken) = take_decimal_fast_path(array, indices, &indices_validity)? {
        return Ok(taken);
    }

    if indices_validity.all_true() {
        let translated = translate_indices_fast(array.filter_mask(), indices, array.len())?;
        let translated_indices = PrimitiveArray::new(
            translated,
            Validity::from_mask(indices_validity, indices.dtype().nullability()),
        )
        .into_array();

        return array.child().take(translated_indices);
    }

    let translated = translate_nullable_indices_fast(
        array.filter_mask(),
        indices,
        &indices_validity,
        array.len(),
    )?;
    let translated_indices = PrimitiveArray::new(
        translated,
        Validity::from_mask(indices_validity, indices.dtype().nullability()),
    )
    .into_array();

    array.child().take(translated_indices)
}

fn should_fallback_nullable_fixed_width_full_take(
    array: ArrayView<'_, Filter>,
    indices: &ArrayRef,
) -> VortexResult<bool> {
    if indices.len() < NULLABLE_FIXED_WIDTH_FULL_TAKE_FALLBACK_LEN
        || indices.len() < array.len()
        || !indices.dtype().is_nullable()
    {
        return Ok(false);
    }

    // For large nullable full-or-larger fixed-width takes with nullable children, materializing the
    // filter first and then taking the child beats translating every nullable rank through the
    // parent.
    if array.child().dtype().is_decimal() || array.child().dtype().is_primitive() {
        return Ok(!array.child().validity()?.no_nulls());
    }

    Ok(false)
}

fn translate_nullable_indices_fast(
    filter: &Mask,
    indices: &PrimitiveArray,
    indices_validity: &Mask,
    filtered_len: usize,
) -> VortexResult<Buffer<u64>> {
    match_each_integer_ptype!(indices.ptype(), |P| {
        translate_nullable_ranks_fast(
            filter,
            indices.as_slice::<P>(),
            indices_validity,
            filtered_len,
        )
    })
}

fn translate_nullable_ranks_fast<P: IntegerPType>(
    filter: &Mask,
    ranks: &[P],
    indices_validity: &Mask,
    filtered_len: usize,
) -> VortexResult<Buffer<u64>> {
    if indices_validity.all_true() {
        return translate_ranks_fast(filter, ranks, filtered_len);
    }
    if indices_validity.all_false() {
        return Ok(Buffer::zeroed(ranks.len()));
    }

    if let Some(start) = contiguous_filter_start(filter, filtered_len) {
        return translate_nullable_ranks_with_offset(ranks, indices_validity, filtered_len, start);
    }

    match filter.indices() {
        AllOr::All => translate_nullable_ranks_identity(ranks, indices_validity, filtered_len),
        AllOr::None => unreachable!("empty filters are handled by take preconditions"),
        AllOr::Some(filter_indices) => translate_nullable_ranks_with_indices(
            ranks,
            indices_validity,
            filtered_len,
            filter_indices,
        ),
    }
}

fn translate_nullable_ranks_with_offset<P: IntegerPType>(
    ranks: &[P],
    indices_validity: &Mask,
    filtered_len: usize,
    start: usize,
) -> VortexResult<Buffer<u64>> {
    let mut translated = BufferMut::<u64>::with_capacity(ranks.len());
    let translated_ptr = translated.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    for (idx, rank) in ranks.iter().enumerate() {
        let translated_rank = if indices_validity.value(idx) {
            let rank = validate_rank(*rank, filtered_len)?;
            u64::try_from(start + rank)?
        } else {
            0
        };

        // SAFETY: `translated` has capacity for all ranks and this loop initializes each
        // output slot once.
        unsafe { translated_ptr.add(idx).write(translated_rank) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { translated.set_len(ranks.len()) };
    Ok(translated.freeze())
}

fn translate_nullable_ranks_identity<P: IntegerPType>(
    ranks: &[P],
    indices_validity: &Mask,
    filtered_len: usize,
) -> VortexResult<Buffer<u64>> {
    let mut translated = BufferMut::<u64>::with_capacity(ranks.len());
    let translated_ptr = translated.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    for (idx, rank) in ranks.iter().enumerate() {
        let translated_rank = if indices_validity.value(idx) {
            u64::try_from(validate_rank(*rank, filtered_len)?)?
        } else {
            0
        };

        // SAFETY: `translated` has capacity for all ranks and this loop initializes each
        // output slot once.
        unsafe { translated_ptr.add(idx).write(translated_rank) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { translated.set_len(ranks.len()) };
    Ok(translated.freeze())
}

fn translate_nullable_ranks_with_indices<P: IntegerPType>(
    ranks: &[P],
    indices_validity: &Mask,
    filtered_len: usize,
    filter_indices: &[usize],
) -> VortexResult<Buffer<u64>> {
    let mut translated = BufferMut::<u64>::with_capacity(ranks.len());
    let translated_ptr = translated.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    for (idx, rank) in ranks.iter().enumerate() {
        let translated_rank = if indices_validity.value(idx) {
            let rank = validate_rank(*rank, filtered_len)?;
            // SAFETY: `rank` was checked against the filtered length, so it is in bounds for
            // `filter_indices`; filter indices are valid child positions by construction.
            unsafe { u64::try_from(*filter_indices.get_unchecked(rank))? }
        } else {
            0
        };

        // SAFETY: `translated` has capacity for all ranks and this loop initializes each
        // output slot once.
        unsafe { translated_ptr.add(idx).write(translated_rank) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { translated.set_len(ranks.len()) };
    Ok(translated.freeze())
}

fn translate_indices_fast(
    filter: &Mask,
    indices: &PrimitiveArray,
    filtered_len: usize,
) -> VortexResult<Buffer<u64>> {
    match_each_integer_ptype!(indices.ptype(), |P| {
        translate_ranks_fast(filter, indices.as_slice::<P>(), filtered_len)
    })
}

fn translate_ranks_fast<P: IntegerPType>(
    filter: &Mask,
    ranks: &[P],
    filtered_len: usize,
) -> VortexResult<Buffer<u64>> {
    let mut translated = BufferMut::<u64>::with_capacity(ranks.len());
    let translated_ptr = translated.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    if let Some(start) = contiguous_filter_start(filter, filtered_len) {
        for (idx, rank) in ranks.iter().enumerate() {
            let Some(rank) = rank.to_usize() else {
                vortex_bail!(OutOfBounds: 0, 0, filtered_len);
            };
            if rank >= filtered_len {
                vortex_bail!(OutOfBounds: rank, 0, filtered_len);
            }

            // SAFETY: `translated` has capacity for all ranks and this loop initializes each
            // output slot once.
            unsafe { translated_ptr.add(idx).write(u64::try_from(start + rank)?) };
        }
    } else {
        let filter_indices = match filter.indices() {
            AllOr::All => {
                for (idx, rank) in ranks.iter().enumerate() {
                    let Some(rank) = rank.to_usize() else {
                        vortex_bail!(OutOfBounds: 0, 0, filtered_len);
                    };
                    if rank >= filtered_len {
                        vortex_bail!(OutOfBounds: rank, 0, filtered_len);
                    }

                    // SAFETY: `translated` has capacity for all ranks and this loop initializes
                    // each output slot once.
                    unsafe { translated_ptr.add(idx).write(u64::try_from(rank)?) };
                }

                // SAFETY: The loop writes exactly `ranks.len()` initialized values.
                unsafe { translated.set_len(ranks.len()) };
                return Ok(translated.freeze());
            }
            AllOr::None => unreachable!("empty filters are handled by take preconditions"),
            AllOr::Some(filter_indices) => filter_indices,
        };

        for (idx, rank) in ranks.iter().enumerate() {
            let Some(rank) = rank.to_usize() else {
                vortex_bail!(OutOfBounds: 0, 0, filtered_len);
            };
            if rank >= filtered_len {
                vortex_bail!(OutOfBounds: rank, 0, filtered_len);
            }

            // SAFETY: `translated` has capacity for all ranks. `rank` was checked against the
            // filtered length, and filter indices are valid child positions by construction.
            unsafe {
                translated_ptr
                    .add(idx)
                    .write(u64::try_from(*filter_indices.get_unchecked(rank))?)
            };
        }
    }

    // SAFETY: Each loop path writes exactly `ranks.len()` initialized values.
    unsafe { translated.set_len(ranks.len()) };
    Ok(translated.freeze())
}

fn validate_rank<P: IntegerPType>(rank: P, filtered_len: usize) -> VortexResult<usize> {
    let Some(rank) = rank.to_usize() else {
        vortex_bail!(OutOfBounds: 0, 0, filtered_len);
    };
    if rank >= filtered_len {
        vortex_bail!(OutOfBounds: rank, 0, filtered_len);
    }
    Ok(rank)
}

fn take_primitive_fast_path(
    array: ArrayView<'_, Filter>,
    indices: &PrimitiveArray,
    indices_validity: &Mask,
) -> VortexResult<Option<ArrayRef>> {
    let Some(child) = array.child().as_opt::<Primitive>() else {
        return Ok(None);
    };

    let child_validity = child.validity()?;
    if !child_validity.no_nulls() {
        return Ok(None);
    }

    let output_validity =
        Validity::from_mask(indices_validity.clone(), indices.dtype().nullability());
    match_each_native_ptype!(child.ptype(), |T| {
        match_each_integer_ptype!(indices.ptype(), |P| {
            take_primitive_fast_path_typed::<T, P>(
                child,
                array.filter_mask(),
                array.len(),
                indices,
                indices_validity,
                output_validity,
            )
            .map(Some)
        })
    })
}

fn take_decimal_fast_path(
    array: ArrayView<'_, Filter>,
    indices: &PrimitiveArray,
    indices_validity: &Mask,
) -> VortexResult<Option<ArrayRef>> {
    let Some(child) = array.child().as_opt::<Decimal>() else {
        return Ok(None);
    };

    let child_validity = child.validity()?;
    if !child_validity.no_nulls() {
        return Ok(None);
    }

    let output_validity =
        Validity::from_mask(indices_validity.clone(), indices.dtype().nullability());
    match_each_decimal_value_type!(child.values_type(), |T| {
        match_each_integer_ptype!(indices.ptype(), |P| {
            take_decimal_fast_path_typed::<T, P>(
                child,
                array.filter_mask(),
                array.len(),
                indices,
                indices_validity,
                output_validity,
            )
            .map(Some)
        })
    })
}

fn take_decimal_fast_path_typed<T, P>(
    child: ArrayView<'_, Decimal>,
    filter: &Mask,
    filtered_len: usize,
    indices: &PrimitiveArray,
    indices_validity: &Mask,
    output_validity: Validity,
) -> VortexResult<ArrayRef>
where
    T: NativeDecimalType,
    P: IntegerPType,
{
    let ranks = indices.as_slice::<P>();
    let decimal_dtype = child.decimal_dtype();

    if indices_validity.all_true() {
        if let Some((start, end)) = contiguous_sequential_take_range(filter, ranks, filtered_len)? {
            let values = child.buffer_handle().slice_typed::<T>(start..end);
            // SAFETY: The values are sliced from an existing valid decimal array, and the output
            // validity was built for exactly the sliced take length.
            return Ok(unsafe {
                DecimalArray::new_unchecked_handle(
                    values,
                    T::DECIMAL_TYPE,
                    decimal_dtype,
                    output_validity,
                )
            }
            .into_array());
        }

        let taken = take_filtered_values::<T, P>(
            child.buffer::<T>().as_slice(),
            filter,
            ranks,
            filtered_len,
        )?;
        // SAFETY: Taking existing decimal values preserves the decimal dtype invariants, and the
        // output validity was built for the take length.
        return Ok(
            unsafe { DecimalArray::new_unchecked(taken, decimal_dtype, output_validity) }
                .into_array(),
        );
    }

    let taken = take_filtered_values_nullable::<T, P>(
        child.buffer::<T>().as_slice(),
        filter,
        ranks,
        indices_validity,
        filtered_len,
    )?;
    // SAFETY: Valid ranks copy existing decimal values, null ranks write default placeholders that
    // are hidden by output validity, and the output validity was built for the take length.
    Ok(unsafe { DecimalArray::new_unchecked(taken, decimal_dtype, output_validity) }.into_array())
}

fn take_primitive_fast_path_typed<T, P>(
    child: ArrayView<'_, Primitive>,
    filter: &Mask,
    filtered_len: usize,
    indices: &PrimitiveArray,
    indices_validity: &Mask,
    output_validity: Validity,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
    P: IntegerPType,
{
    let ranks = indices.as_slice::<P>();

    if indices_validity.all_true() {
        return take_primitive_fast_path_all_valid::<T, P>(
            child,
            filter,
            filtered_len,
            ranks,
            output_validity,
        );
    }

    take_primitive_fast_path_nullable::<T, P>(
        child.as_slice::<T>(),
        filter,
        filtered_len,
        ranks,
        indices_validity,
        output_validity,
    )
}

fn take_primitive_fast_path_all_valid<T, P>(
    child: ArrayView<'_, Primitive>,
    filter: &Mask,
    filtered_len: usize,
    ranks: &[P],
    output_validity: Validity,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
    P: IntegerPType,
{
    if let Some((start, end)) = contiguous_sequential_take_range(filter, ranks, filtered_len)? {
        return Ok(PrimitiveArray::from_buffer_handle(
            child.buffer_handle().slice_typed::<T>(start..end),
            T::PTYPE,
            output_validity,
        )
        .into_array());
    }

    let taken = take_filtered_values::<T, P>(child.as_slice::<T>(), filter, ranks, filtered_len)?;
    Ok(PrimitiveArray::new(taken, output_validity).into_array())
}

fn take_primitive_fast_path_nullable<T, P>(
    values: &[T],
    filter: &Mask,
    filtered_len: usize,
    ranks: &[P],
    indices_validity: &Mask,
    output_validity: Validity,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
    P: IntegerPType,
{
    let taken = take_filtered_values_nullable::<T, P>(
        values,
        filter,
        ranks,
        indices_validity,
        filtered_len,
    )?;
    Ok(PrimitiveArray::new(taken, output_validity).into_array())
}

fn take_filtered_values_nullable<T, P>(
    values: &[T],
    filter: &Mask,
    ranks: &[P],
    indices_validity: &Mask,
    filtered_len: usize,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    if indices_validity.all_false() {
        return Ok(Buffer::zeroed(ranks.len()));
    }

    if let Some(start) = contiguous_filter_start(filter, filtered_len) {
        let mut out = BufferMut::<T>::with_capacity(ranks.len());
        let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
        for (idx, rank) in ranks.iter().enumerate() {
            let value = if indices_validity.value(idx) {
                let rank = validate_rank(*rank, filtered_len)?;
                // SAFETY: `rank` was checked against the contiguous filtered length.
                unsafe { *values.get_unchecked(start + rank) }
            } else {
                T::default()
            };

            // SAFETY: `out` has capacity for all ranks and this loop initializes each output slot
            // once.
            unsafe { out_ptr.add(idx).write(value) };
        }

        // SAFETY: The loop writes exactly `ranks.len()` initialized values.
        unsafe { out.set_len(ranks.len()) };
        return Ok(out.freeze());
    }

    let indices = match filter.indices() {
        AllOr::All => {
            return take_values_by_rank_nullable(values, ranks, indices_validity, filtered_len);
        }
        AllOr::None => unreachable!("empty filters are handled by take preconditions"),
        AllOr::Some(indices) => indices,
    };

    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let value = if indices_validity.value(idx) {
            let rank = validate_rank(*rank, filtered_len)?;
            // SAFETY: `rank` was bounds-checked against `indices`, whose values are valid
            // positions in `values`.
            unsafe { *values.get_unchecked(*indices.get_unchecked(rank)) }
        } else {
            T::default()
        };

        // SAFETY: `out` has capacity for all ranks and this loop initializes each output slot
        // once.
        unsafe { out_ptr.add(idx).write(value) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}

fn take_values_by_rank_nullable<T, P>(
    values: &[T],
    ranks: &[P],
    indices_validity: &Mask,
    filtered_len: usize,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let value = if indices_validity.value(idx) {
            let rank = validate_rank(*rank, filtered_len)?;
            // SAFETY: `rank` was bounds-checked.
            unsafe { *values.get_unchecked(rank) }
        } else {
            T::default()
        };

        // SAFETY: `out` has capacity for all ranks and this loop initializes each output slot
        // once.
        unsafe { out_ptr.add(idx).write(value) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}

fn contiguous_sequential_take_range<P: IntegerPType>(
    filter: &Mask,
    ranks: &[P],
    filtered_len: usize,
) -> VortexResult<Option<(usize, usize)>> {
    let Some(start) = contiguous_filter_start(filter, filtered_len) else {
        return Ok(None);
    };

    for (idx, rank) in ranks.iter().enumerate() {
        let Some(rank) = rank.to_usize() else {
            vortex_bail!(OutOfBounds: 0, 0, filtered_len);
        };
        if rank >= filtered_len {
            vortex_bail!(OutOfBounds: rank, 0, filtered_len);
        }
        if rank != idx {
            return Ok(None);
        }
    }

    Ok(Some((start, start + ranks.len())))
}

fn take_filtered_values<T, P>(
    values: &[T],
    filter: &Mask,
    ranks: &[P],
    filtered_len: usize,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    if let Some(start) = contiguous_filter_start(filter, filtered_len) {
        let mut out = BufferMut::<T>::with_capacity(ranks.len());
        let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
        for (idx, rank) in ranks.iter().enumerate() {
            let Some(rank) = rank.to_usize() else {
                vortex_bail!(OutOfBounds: 0, 0, filtered_len);
            };
            if rank >= filtered_len {
                vortex_bail!(OutOfBounds: rank, 0, filtered_len);
            }

            // SAFETY: `out` has capacity for all ranks. The filter is contiguous with
            // `filtered_len` values starting at `start`, and `rank` was checked above.
            unsafe { out_ptr.add(idx).write(*values.get_unchecked(start + rank)) };
        }

        // SAFETY: The loop writes exactly `ranks.len()` initialized values.
        unsafe { out.set_len(ranks.len()) };
        return Ok(out.freeze());
    }

    let indices = match filter.indices() {
        AllOr::All => return take_values_by_rank(values, ranks, filtered_len),
        AllOr::None => unreachable!("empty filters are handled by take preconditions"),
        AllOr::Some(indices) => indices,
    };

    if ranks.len() == filtered_len && !first_rank_is_zero(ranks, filtered_len)? {
        let filtered = gather_values_by_indices(values, indices);
        return take_values_by_rank(filtered.as_slice(), ranks, filtered_len);
    }

    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let Some(rank) = rank.to_usize() else {
            vortex_bail!(OutOfBounds: 0, 0, filtered_len);
        };
        if rank >= filtered_len {
            vortex_bail!(OutOfBounds: rank, 0, filtered_len);
        }

        // SAFETY: `out` has capacity for all ranks. `rank` was bounds-checked against
        // `indices`, whose values are valid positions in `values`.
        unsafe {
            out_ptr
                .add(idx)
                .write(*values.get_unchecked(*indices.get_unchecked(rank)))
        };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}

fn gather_values_by_indices<T>(values: &[T], indices: &[usize]) -> Buffer<T>
where
    T: Copy + Default,
{
    let mut out = BufferMut::<T>::with_capacity(indices.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();

    for (idx, &value_idx) in indices.iter().enumerate() {
        // SAFETY: `out` has capacity for all indices and mask indices are valid positions in the
        // child values buffer by construction.
        unsafe { out_ptr.add(idx).write(*values.get_unchecked(value_idx)) };
    }

    // SAFETY: The loop writes exactly `indices.len()` initialized values.
    unsafe { out.set_len(indices.len()) };
    out.freeze()
}

fn first_rank_is_zero<P: IntegerPType>(ranks: &[P], filtered_len: usize) -> VortexResult<bool> {
    let Some(first) = ranks.first() else {
        return Ok(false);
    };
    let Some(first) = first.to_usize() else {
        vortex_bail!(OutOfBounds: 0, 0, filtered_len);
    };
    if first >= filtered_len {
        vortex_bail!(OutOfBounds: first, 0, filtered_len);
    }
    Ok(first == 0)
}

fn take_values_by_rank<T, P>(
    values: &[T],
    ranks: &[P],
    filtered_len: usize,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let Some(rank) = rank.to_usize() else {
            vortex_bail!(OutOfBounds: 0, 0, filtered_len);
        };
        if rank >= filtered_len {
            vortex_bail!(OutOfBounds: rank, 0, filtered_len);
        }

        // SAFETY: `out` has capacity for all ranks and `rank` was bounds-checked.
        unsafe { out_ptr.add(idx).write(*values.get_unchecked(rank)) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}

fn contiguous_filter_start(filter: &Mask, filtered_len: usize) -> Option<usize> {
    let start = filter.first()?;
    let end = filter.last()?.checked_add(1)?;
    (end - start == filtered_len).then_some(start)
}

impl TakeExecute for Filter {
    fn take(
        array: ArrayView<'_, Filter>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Bool filtering is already very cheap. Translating take indices through the filter adds
        // overhead without improving the downstream bool take, so leave bool children on the
        // regular filter path.
        if array.child().dtype().is_boolean() {
            return Ok(None);
        }

        let DType::Primitive(ptype, nullability) = indices.dtype() else {
            vortex_bail!("Invalid indices dtype: {}", indices.dtype())
        };

        if should_fallback_nullable_fixed_width_full_take(array, indices)? {
            return Ok(None);
        }

        let unsigned_indices = if ptype.is_unsigned_int() {
            indices.clone().execute::<PrimitiveArray>(ctx)?
        } else {
            indices
                .clone()
                .cast(DType::Primitive(ptype.to_unsigned(), *nullability))?
                .execute::<PrimitiveArray>(ctx)?
        };

        take_impl(array, &unsigned_indices, ctx).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::RecursiveCanonical;
    use crate::arrays::BoolArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::FilterArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DecimalDType;
    use crate::dtype::FieldNames;
    use crate::executor::ExecutionCtx;
    use crate::validity::Validity;

    #[test]
    fn test_take_execute_kernel_maps_indices_through_filter() -> VortexResult<()> {
        let filter = FilterArray::new(
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30), Some(40), None])
                .into_array(),
            Mask::from_iter([true, false, true, true, false]),
        )
        .into_array();
        let parent = DictArray::try_new(
            PrimitiveArray::new(
                buffer![2u64, 100, 0],
                Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
            )
            .into_array(),
            filter.clone(),
        )?
        .into_array();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = filter
            .execute_parent(&parent, 1, &mut ctx)?
            .expect("filter child should execute its take parent");

        assert!(result.as_opt::<Dict>().is_some());
        assert_arrays_eq!(
            result.execute::<RecursiveCanonical>(&mut ctx)?.0,
            PrimitiveArray::from_option_iter([Some(40i32), None, Some(10)]).into_array()
        );
        Ok(())
    }

    #[test]
    fn test_take_execute_kernel_nullable_fast_path_maps_indices_through_filter() -> VortexResult<()>
    {
        let filter = FilterArray::new(
            buffer![10i32, 20, 30, 40, 50].into_array(),
            Mask::from_slices(5, vec![(1, 4)]),
        )
        .into_array();
        let parent = DictArray::try_new(
            PrimitiveArray::new(
                buffer![2u64, 100, 0],
                Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
            )
            .into_array(),
            filter.clone(),
        )?
        .into_array();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = filter
            .execute_parent(&parent, 1, &mut ctx)?
            .expect("filter child should execute its take parent");

        assert!(result.as_opt::<Primitive>().is_some());
        assert_arrays_eq!(
            result.execute::<RecursiveCanonical>(&mut ctx)?.0,
            PrimitiveArray::from_option_iter([Some(40i32), None, Some(20)]).into_array()
        );
        Ok(())
    }

    #[test]
    fn test_take_execute_kernel_fast_path_maps_indices_through_filter() -> VortexResult<()> {
        let filter = FilterArray::new(
            buffer![10i32, 20, 30, 40, 50, 60].into_array(),
            Mask::from_indices(6, vec![1, 3, 4, 5]),
        )
        .into_array();
        let parent =
            DictArray::try_new(buffer![2u64, 0, 3].into_array(), filter.clone())?.into_array();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = filter
            .execute_parent(&parent, 1, &mut ctx)?
            .expect("filter child should execute its take parent");

        assert!(result.as_opt::<Primitive>().is_some());
        assert_arrays_eq!(
            result.execute::<RecursiveCanonical>(&mut ctx)?.0,
            PrimitiveArray::from_iter([50i32, 20, 60]).into_array()
        );
        Ok(())
    }

    fn assert_take_execute_maps_child_dtype(
        child: crate::ArrayRef,
        expected: crate::ArrayRef,
    ) -> VortexResult<()> {
        let filter =
            FilterArray::new(child, Mask::from_iter([true, false, true, true, false])).into_array();
        let parent =
            DictArray::try_new(buffer![2u64, 0, 1].into_array(), filter.clone())?.into_array();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = filter
            .execute_parent(&parent, 1, &mut ctx)?
            .expect("filter child should execute its take parent");

        assert_arrays_eq!(result.execute::<RecursiveCanonical>(&mut ctx)?.0, expected);
        Ok(())
    }

    #[test]
    fn test_take_execute_kernel_skips_bool_filter_child() -> VortexResult<()> {
        let filter = FilterArray::new(
            BoolArray::from_iter([true, false, true, true, false]).into_array(),
            Mask::from_iter([true, false, true, true, false]),
        )
        .into_array();
        let parent =
            DictArray::try_new(buffer![2u64, 0, 1].into_array(), filter.clone())?.into_array();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = filter.execute_parent(&parent, 1, &mut ctx)?;

        assert!(result.is_none());
        Ok(())
    }

    fn execute_large_nullable_fixed_width_take(
        child: crate::ArrayRef,
    ) -> VortexResult<Option<crate::ArrayRef>> {
        let filter =
            FilterArray::new(child, Mask::from_iter([true, false, true, true, false])).into_array();
        let indices = PrimitiveArray::from_option_iter(
            (0..=super::NULLABLE_FIXED_WIDTH_FULL_TAKE_FALLBACK_LEN)
                .map(|idx| Some((idx % 3) as u64)),
        )
        .into_array();
        let parent = DictArray::try_new(indices, filter.clone())?.into_array();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        filter.execute_parent(&parent, 1, &mut ctx)
    }

    #[test]
    fn test_take_execute_kernel_handles_large_nullable_primitive_take_without_child_nulls()
    -> VortexResult<()> {
        let result =
            execute_large_nullable_fixed_width_take(buffer![10i32, 20, 30, 40, 50].into_array())?;

        assert_eq!(
            result
                .expect("non-null fixed-width child should stay on the fast path")
                .len(),
            super::NULLABLE_FIXED_WIDTH_FULL_TAKE_FALLBACK_LEN + 1
        );
        Ok(())
    }

    #[test]
    fn test_take_execute_kernel_handles_large_nullable_decimal_take_without_child_nulls()
    -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(19, 2);

        let result = execute_large_nullable_fixed_width_take(
            DecimalArray::new(
                buffer![100i128, 200, 300, 400, 500],
                decimal_dtype,
                Validity::NonNullable,
            )
            .into_array(),
        )?;

        assert_eq!(
            result
                .expect("non-null fixed-width child should stay on the fast path")
                .len(),
            super::NULLABLE_FIXED_WIDTH_FULL_TAKE_FALLBACK_LEN + 1
        );
        Ok(())
    }

    #[test]
    fn test_take_execute_kernel_falls_back_for_large_nullable_primitive_take_with_child_nulls()
    -> VortexResult<()> {
        let result = execute_large_nullable_fixed_width_take(
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), None, Some(40), Some(50)])
                .into_array(),
        )?;

        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn test_take_execute_kernel_falls_back_for_large_nullable_decimal_take_with_child_nulls()
    -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(19, 2);
        let result = execute_large_nullable_fixed_width_take(
            DecimalArray::from_option_iter(
                [Some(100i128), Some(200), None, Some(400), Some(500)],
                decimal_dtype,
            )
            .into_array(),
        )?;

        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn test_take_execute_kernel_handles_decimal_filter_child() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(19, 2);

        assert_take_execute_maps_child_dtype(
            DecimalArray::new(
                buffer![100i128, 200, 300, 400, 500],
                decimal_dtype,
                Validity::NonNullable,
            )
            .into_array(),
            DecimalArray::new(
                buffer![400i128, 100, 300],
                decimal_dtype,
                Validity::NonNullable,
            )
            .into_array(),
        )
    }

    #[test]
    fn test_take_execute_kernel_handles_fixed_size_list_filter_child() -> VortexResult<()> {
        assert_take_execute_maps_child_dtype(
            FixedSizeListArray::new(
                buffer![10u32, 11, 20, 21, 30, 31, 40, 41, 50, 51].into_array(),
                2,
                Validity::NonNullable,
                5,
            )
            .into_array(),
            FixedSizeListArray::new(
                buffer![40u32, 41, 10, 11, 30, 31].into_array(),
                2,
                Validity::NonNullable,
                3,
            )
            .into_array(),
        )
    }

    #[test]
    fn test_take_execute_kernel_handles_list_filter_child() -> VortexResult<()> {
        assert_take_execute_maps_child_dtype(
            ListArray::try_new(
                buffer![10u32, 11, 20, 30, 31, 32, 40, 50, 51].into_array(),
                buffer![0u32, 2, 3, 6, 7, 9].into_array(),
                Validity::NonNullable,
            )?
            .into_array(),
            ListArray::try_new(
                buffer![40u32, 10, 11, 30, 31, 32].into_array(),
                buffer![0u32, 1, 3, 6].into_array(),
                Validity::NonNullable,
            )?
            .into_array(),
        )
    }

    #[test]
    fn test_take_execute_kernel_handles_string_filter_child() -> VortexResult<()> {
        assert_take_execute_maps_child_dtype(
            VarBinViewArray::from_iter_str(["a", "b", "c", "d", "e"]).into_array(),
            VarBinViewArray::from_iter_str(["d", "a", "c"]).into_array(),
        )
    }

    #[test]
    fn test_take_execute_kernel_handles_struct_filter_child() -> VortexResult<()> {
        assert_take_execute_maps_child_dtype(
            StructArray::try_new(
                FieldNames::from(["id", "value"]),
                vec![
                    buffer![10u32, 20, 30, 40, 50].into_array(),
                    buffer![100u64, 200, 300, 400, 500].into_array(),
                ],
                5,
                Validity::NonNullable,
            )?
            .into_array(),
            StructArray::try_new(
                FieldNames::from(["id", "value"]),
                vec![
                    buffer![40u32, 10, 30].into_array(),
                    buffer![400u64, 100, 300].into_array(),
                ],
                3,
                Validity::NonNullable,
            )?
            .into_array(),
        )
    }
}
