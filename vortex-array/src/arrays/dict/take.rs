// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexResult;

use super::Dict;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::ParentRef;
use crate::array::VTable;
use crate::arrays::ConstantArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::DictSlotsView;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar::Scalar;
use crate::stats::StatsSet;
use crate::validity::Validity;

pub trait TakeReduce: VTable {
    /// Take elements from an array at the given indices without reading buffers.
    ///
    /// This trait is for take implementations that can operate purely on array metadata and
    /// structure without needing to read or execute on the underlying buffers. Implementations
    /// should return `None` if taking requires buffer access.
    ///
    /// # Preconditions
    ///
    /// The indices are guaranteed to be non-empty.
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
}

pub trait TakeExecute: VTable {
    /// Take elements from an array at the given indices, potentially reading buffers.
    ///
    /// Unlike [`TakeReduce`], this trait is for take implementations that may need to read
    /// and execute on the underlying buffers to produce the result.
    ///
    /// # Preconditions
    ///
    /// The indices are guaranteed to be non-empty.
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Common preconditions for take operations that apply to all arrays.
///
/// Returns `Some(result)` if the precondition short-circuits the take operation,
/// or `None` if the take should proceed normally.
fn precondition<V: VTable>(array: ArrayView<'_, V>, indices: &ArrayRef) -> Option<ArrayRef> {
    // Fast-path for empty indices.
    if indices.is_empty() {
        let result_dtype = array
            .dtype()
            .clone()
            .union_nullability(indices.dtype().nullability());
        return Some(Canonical::empty(&result_dtype).into_array());
    }

    // Fast-path for empty arrays: all indices must be null, return all-invalid result.
    if array.is_empty() {
        return Some(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                .into_array(),
        );
    }

    None
}

#[derive(Default, Debug)]
pub struct TakeReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for TakeReduceAdaptor<V>
where
    V: TakeReduce,
{
    type Parent = Dict;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ArrayView<'_, Dict>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle the values child (index 1), not the codes child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        if let Some(result) = precondition::<V>(array, parent.codes()) {
            return Ok(Some(result));
        }
        let result = <V as TakeReduce>::take(array, parent.codes())?;
        if let Some(taken) = &result {
            propagate_take_stats(array.array(), taken, parent.codes())?;
        }
        Ok(result)
    }

    fn reduce_parent_ref(
        &self,
        array: ArrayView<'_, V>,
        parent: &ParentRef<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle the values child (index 1), not the codes child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let Some(parent) = parent.try_view::<Dict>() else {
            return Ok(None);
        };
        let codes = DictSlotsView::from_slots(parent.slots()).codes;
        if let Some(result) = precondition::<V>(array, codes) {
            return Ok(Some(result));
        }
        let result = <V as TakeReduce>::take(array, codes)?;
        if let Some(taken) = &result {
            propagate_take_stats(array.array(), taken, codes)?;
        }
        Ok(result)
    }
}

#[derive(Default, Debug)]
pub struct TakeExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for TakeExecuteAdaptor<V>
where
    V: TakeExecute,
{
    type Parent = Dict;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle the values child (index 1), not the codes child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        if let Some(result) = precondition::<V>(array, parent.codes()) {
            return Ok(Some(result));
        }
        let result = <V as TakeExecute>::take(array, parent.codes(), ctx)?;
        if let Some(taken) = &result {
            propagate_take_stats(array.array(), taken, parent.codes())?;
        }
        Ok(result)
    }
}

pub(crate) fn propagate_take_stats(
    source: &ArrayRef,
    target: &ArrayRef,
    indices: &ArrayRef,
) -> VortexResult<()> {
    let indices_all_valid = matches!(
        indices.validity()?,
        Validity::NonNullable | Validity::AllValid
    );
    target.statistics().with_mut_typed_stats_set(|mut st| {
        if indices_all_valid {
            let is_constant = source.statistics().get_as::<bool>(Stat::IsConstant);
            if is_constant == Some(Precision::Exact(true)) {
                // Any combination of elements from a constant array is still const
                st.set(Stat::IsConstant, Precision::exact(true));
            }
        }
        let inexact_min_max = [Stat::Min, Stat::Max]
            .into_iter()
            .filter_map(|stat| {
                source
                    .statistics()
                    .get(stat)
                    .and_then(|v| v.map(|s| s.into_value()).into_inexact().transpose())
                    .map(|sv| (stat, sv))
            })
            .collect::<SmallVec<_>>();
        st.combine_sets(
            &(unsafe { StatsSet::new_unchecked(inexact_min_max) }).as_typed_ref(source.dtype()),
        )
    })
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use crate::IntoArray;
    use crate::ParentRef;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;

    #[test]
    fn reduce_adaptor_handles_stack_backed_dict_parent() -> VortexResult<()> {
        let indices = PrimitiveArray::from_iter([0u32, 0, 0]).into_array();
        let values = ConstantArray::new(7i32, 1).into_array();
        let parts = DictArray::try_new_parts(indices, values)?;

        let Some(reduced) = ParentRef::from_parts(&parts).optimize()? else {
            vortex_bail!("take reduce adaptor should optimize stack-backed parent");
        };

        assert!(reduced.is::<Constant>());
        assert_eq!(reduced.len(), 3);
        Ok(())
    }
}
