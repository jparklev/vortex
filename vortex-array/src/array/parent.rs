// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stack-allocatable parent representation used by the `reduce_parent` dispatch chain.
//!
//! [`ParentRef`] either borrows an existing heap-allocated [`ArrayRef`], or borrows
//! stack-allocated construction state. The construction-side optimizer can borrow
//! `ArrayParts` before materializing an `ArrayInner`, so matchers and parent-reduce
//! rules can attempt reduction without first allocating an `Arc<ArrayInner<_>>`.
//!
//! Both cases expose the same borrowed metadata to rule dispatch. Heap-backed parents
//! also retain the original [`ArrayRef`] for compatibility with existing matchers that
//! still need an array allocation.

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;

use vortex_error::VortexResult;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::array::ArrayData;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::dtype::DType;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::kernels::ArrayKernels;

/// A parent array, possibly stack-allocated, used by the `reduce_parent` dispatch chain.
///
/// Optionally holds an `&ArrayRef` when the parent is heap-allocated. The remaining
/// metadata (encoding id, dtype, length, encoding-specific data, slots) is always
/// available so matchers and rule bodies can dispatch without distinguishing the two
/// cases.
#[derive(Clone, Copy)]
pub struct ParentRef<'a> {
    encoding_id: ArrayId,
    dtype: &'a DType,
    len: usize,
    data: ParentData<'a>,
    slots: &'a [Option<ArrayRef>],
    array_ref: Option<&'a ArrayRef>,
}

#[derive(Clone, Copy)]
enum ParentData<'a> {
    ArrayData(&'a dyn Any),
    Parts {
        vtable: &'a dyn Any,
        data: &'a dyn Any,
    },
}

impl<'a> ParentRef<'a> {
    /// Build a [`ParentRef`] borrowing a heap-allocated [`ArrayRef`].
    #[inline]
    pub fn from_array_ref(array: &'a ArrayRef) -> Self {
        let inner = array.inner();
        Self {
            array_ref: Some(array),
            encoding_id: inner.encoding_id,
            dtype: &inner.dtype,
            len: inner.len,
            data: ParentData::ArrayData(inner.data.as_any()),
            slots: &inner.slots,
        }
    }

    /// Build a [`ParentRef`] borrowing construction parts before materialization.
    #[inline]
    pub(crate) fn from_parts<V: VTable>(parts: &'a ArrayParts<V>) -> Self {
        Self {
            array_ref: None,
            encoding_id: parts.vtable.id(),
            dtype: &parts.dtype,
            len: parts.len,
            data: ParentData::Parts {
                vtable: &parts.vtable,
                data: &parts.data,
            },
            slots: &parts.slots,
        }
    }

    /// Try to optimize this parent using static parent-reduction rules.
    ///
    /// Returns `Some(reduced)` when a parent-reduction rule rewrites the parent, or
    /// `None` when no rule applies. This method does not materialize the original
    /// parent when no rule fires; callers that own stack construction state should
    /// materialize that state themselves.
    pub fn optimize(&self) -> VortexResult<Option<ArrayRef>> {
        self.optimize_inner(None)
    }

    /// Try to optimize this parent using static rules and session-registered kernels.
    ///
    /// Session kernels are checked before static parent-reduction rules.
    pub fn optimize_ctx(&self, session: &VortexSession) -> VortexResult<Option<ArrayRef>> {
        self.optimize_inner(Some(session))
    }

    fn optimize_inner(&self, session: Option<&VortexSession>) -> VortexResult<Option<ArrayRef>> {
        let kernels = session.and_then(|s| s.get_opt::<ArrayKernels>());

        for (slot_idx, slot) in self.slots.iter().enumerate() {
            let Some(child) = slot else { continue };

            // Session kernels take precedence over static `PARENT_RULES`, matching
            // the existing optimizer's ordering.
            if let Some(kernels) = &kernels
                && let Some(plugins) =
                    kernels.find_reduce_parent(self.encoding_id, child.encoding_id())
            {
                for plugin in plugins.as_ref() {
                    if let Some(reduced) = plugin(child, self, slot_idx)? {
                        return cascade(reduced, session).map(Some);
                    }
                }
            }

            if let Some(reduced) = child.reduce_parent(self, slot_idx)? {
                return cascade(reduced, session).map(Some);
            }
        }

        Ok(None)
    }

    /// Returns the encoding id of the parent.
    #[inline]
    pub fn encoding_id(&self) -> ArrayId {
        self.encoding_id
    }

    /// Returns the dtype of the parent.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the length of the parent.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the parent is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the slots of the parent.
    #[inline]
    pub fn slots(&self) -> &'a [Option<ArrayRef>] {
        self.slots
    }

    /// Returns the underlying [`ArrayRef`] if the parent is heap-allocated.
    ///
    /// Returns `None` when the parent only exists on the stack.
    #[inline]
    pub fn array_ref(&self) -> Option<&'a ArrayRef> {
        self.array_ref
    }

    /// Try to extract a [`ParentView`] for the parent's encoding `V`.
    ///
    /// Returns `None` if the parent's encoding is not `V`. Works for both heap-
    /// and stack-allocated parents.
    pub fn try_view<V: VTable>(&self) -> Option<ParentView<'a, V>> {
        let data = self.typed_data::<V>()?;
        Some(ParentView {
            data,
            dtype: self.dtype,
            len: self.len,
            slots: self.slots,
        })
    }

    #[inline]
    pub(crate) fn is_encoding<V: VTable>(&self) -> bool {
        match self.data {
            ParentData::ArrayData(data) => data.is::<ArrayData<V>>(),
            ParentData::Parts { vtable, .. } => vtable.is::<V>(),
        }
    }

    #[inline]
    pub(crate) fn typed_data<V: VTable>(&self) -> Option<&'a V::TypedArrayData> {
        if !self.is_encoding::<V>() {
            return None;
        }

        match self.data {
            ParentData::ArrayData(data) => data
                .downcast_ref::<ArrayData<V>>()
                .map(|array_data| &array_data.data),
            ParentData::Parts { data, .. } => data.downcast_ref::<V::TypedArrayData>(),
        }
    }

    /// Try to extract an [`ArrayView`] for the parent's encoding `V`.
    ///
    /// Returns `None` if the parent is not heap-allocated, or if its encoding is not
    /// `V`. This is useful for matcher implementations whose match type is an
    /// `ArrayView` and therefore requires an underlying `ArrayRef`.
    pub fn try_array_view<V: VTable>(&self) -> Option<ArrayView<'a, V>> {
        self.array_ref?.as_typed::<V>()
    }
}

impl Debug for ParentRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParentRef")
            .field("encoding", &self.encoding_id())
            .field("dtype", self.dtype())
            .field("len", &self.len())
            .field("heap_backed", &self.array_ref.is_some())
            .finish()
    }
}

impl<'a> From<&'a ArrayRef> for ParentRef<'a> {
    fn from(array: &'a ArrayRef) -> Self {
        Self::from_array_ref(array)
    }
}

#[inline]
fn cascade(reduced: ArrayRef, session: Option<&VortexSession>) -> VortexResult<ArrayRef> {
    match session {
        Some(s) => reduced.optimize_ctx(s),
        None => reduced.optimize(),
    }
}

/// A typed view of a parent array, possibly stack-allocated.
///
/// Provides the same surface as [`ArrayView`] for the parent data: `Deref` to
/// `V::ArrayData`, plus `slots`, `dtype`, and `len` accessors. It does not expose
/// an underlying `&ArrayRef`, since the parent may not have one.
pub struct ParentView<'a, V: VTable> {
    data: &'a V::TypedArrayData,
    dtype: &'a DType,
    len: usize,
    slots: &'a [Option<ArrayRef>],
}

impl<V: VTable> Copy for ParentView<'_, V> {}

impl<V: VTable> Clone for ParentView<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: VTable> ParentView<'a, V> {
    /// Returns the encoding-specific data.
    #[inline]
    pub fn data(&self) -> &'a V::TypedArrayData {
        self.data
    }

    /// Returns the dtype of the parent.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the length of the parent.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the parent is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the parent's slots.
    #[inline]
    pub fn slots(&self) -> &'a [Option<ArrayRef>] {
        self.slots
    }
}

impl<V: VTable> Deref for ParentView<'_, V> {
    type Target = V::TypedArrayData;

    fn deref(&self) -> &V::TypedArrayData {
        self.data
    }
}

impl<V: VTable> Debug for ParentView<'_, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParentView")
            .field("dtype", self.dtype)
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::ParentRef;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::Slice;
    use crate::arrays::SliceArray;

    #[test]
    fn parts_parent_ref_exposes_parent_view() -> VortexResult<()> {
        let child = BoolArray::from_iter([true, false, true]).into_array();
        let parts = SliceArray::try_new_parts(child, 1..3)?;
        let parent = ParentRef::from_parts(&parts);

        let view = parent
            .try_view::<Slice>()
            .expect("Slice parts should match a Slice parent view");

        assert_eq!(view.slice_range(), &(1..3));
        assert!(parent.try_array_view::<Slice>().is_none());

        Ok(())
    }
}
