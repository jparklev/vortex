// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metadata-only rewrite rules for the optimizer (Layers 1 and 2 of the execution model).
//!
//! Reduce rules are the cheapest transformations in the execution pipeline: they operate
//! purely on array structure and metadata without reading any data buffers.
//!
//! There are two kinds of reduce rules:
//!
//! - [`ArrayReduceRule`] (Layer 1) -- a self-rewrite where an array simplifies itself.
//!   Example: a `FilterArray` with an all-true mask removes the filter wrapper.
//!
//! - [`ArrayParentReduceRule`] (Layer 2) -- a child-driven rewrite where a child rewrites
//!   its parent. Example: a `DictArray` child of a `ScalarFnArray` pushes the scalar function
//!   into the dictionary values.
//!
//! Rules are collected into [`ReduceRuleSet`] and [`ParentRuleSet`] respectively, and
//! evaluated by the optimizer in a fixpoint loop until no more rules apply.

use std::any::type_name;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::ParentRef;
use crate::array::VTable;
use crate::matcher::Matcher;

/// A metadata-only rewrite rule that transforms an array based on its own structure (Layer 1).
///
/// These rules look only at the array's metadata and children types (not buffer contents)
/// and return a structurally simpler replacement, or `None` if the rule doesn't apply.
pub trait ArrayReduceRule<V: VTable>: Debug + Send + Sync + 'static {
    /// Attempt to rewrite this array.
    ///
    /// Returns:
    /// - `Ok(Some(new_array))` if the rule applied successfully
    /// - `Ok(None)` if the rule doesn't apply
    /// - `Err(e)` if an error occurred
    fn reduce(&self, array: ArrayView<'_, V>) -> VortexResult<Option<ArrayRef>>;
}

/// A metadata-only rewrite rule where a child encoding rewrites its parent (Layer 2).
///
/// The child sees the parent's type via the associated `Parent` [`Matcher`] and can return
/// a replacement for the parent. This enables optimizations like pushing operations through
/// compression layers (e.g., pushing a scalar function into dictionary values).
///
/// # Migration to `ParentRef`
///
/// New rules should override [`ArrayParentReduceRule::reduce_parent_ref`] directly so
/// they participate in the stack-backed dispatch driven by
/// [`ParentRef::optimize`](crate::array::ParentRef::optimize). Construction-side
/// callers borrow `ArrayParts` as a [`ParentRef`] and only materialize the `ArrayRef`
/// if no stack-backed parent rule fires. The default
/// `reduce_parent_ref` falls through to [`reduce_parent`](Self::reduce_parent) via
/// [`Matcher::try_match_parent`]. Whether that can match stack-allocated parents
/// depends on the matcher; for example the blanket `VTable` matcher returns an
/// [`ArrayView`] and therefore still requires a heap-allocated parent.
pub trait ArrayParentReduceRule<V: VTable>: Debug + Send + Sync + 'static {
    /// The parent array type this rule matches against.
    type Parent: Matcher;

    /// Attempt to rewrite this child array given information about its parent.
    ///
    /// Returns:
    /// - `Ok(Some(new_array))` if the rule applied successfully
    /// - `Ok(None)` if the rule doesn't apply
    /// - `Err(e)` if an error occurred
    ///
    /// # Deprecated
    ///
    /// Prefer overriding [`ArrayParentReduceRule::reduce_parent_ref`]. This method
    /// only sees the parent through [`Matcher::Match`], which for the blanket
    /// `impl<V: VTable> Matcher for V` is an [`ArrayView`] holding an `&ArrayRef` —
    /// in other words it only fires when the parent matcher can produce an
    /// `ArrayView`. Rules that override `reduce_parent_ref` instead can rewrite
    /// stack-allocated parents via [`ParentRef::optimize`](crate::array::ParentRef::optimize)
    /// without forcing an `Arc<ArrayInner<_>>` allocation.
    ///
    /// Existing implementations can stay; the default `reduce_parent_ref` continues
    /// to delegate here when the matcher can materialize its `Match` type.
    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;

    /// Attempt to rewrite the child given a [`ParentRef`].
    ///
    /// This is the dispatch entry point used by [`ParentRuleSet`]. The default
    /// implementation extracts the parent's typed view via
    /// [`Matcher::try_match_parent`] and then delegates to
    /// [`reduce_parent`](Self::reduce_parent).
    ///
    /// Override this when the rule can rewrite a stack-allocated parent, typically by
    /// extracting an encoding-specific view via [`ParentRef::try_view`]. Doing so lets
    /// callers using [`ParentRef::optimize`](crate::array::ParentRef::optimize) avoid
    /// materializing an `Arc<ArrayInner<_>>` when the rule fires.
    ///
    /// # Stability
    ///
    /// **Unstable.** This is the new `ParentRef`-based dispatch entry; the signature
    /// and contract are expected to change as more rules migrate off
    /// [`reduce_parent`](Self::reduce_parent). Treat overrides as opt-in for now.
    fn reduce_parent_ref(
        &self,
        array: ArrayView<'_, V>,
        parent: &ParentRef<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(parent_view) = <Self::Parent as Matcher>::try_match_parent(parent) else {
            return Ok(None);
        };
        self.reduce_parent(array, parent_view, child_idx)
    }
}

/// Type-erased version of [`ArrayParentReduceRule`] used for dynamic dispatch within
/// [`ParentRuleSet`].
pub trait DynArrayParentReduceRule<V: VTable>: Debug + Send + Sync {
    fn matches(&self, parent: &ParentRef<'_>) -> bool;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: &ParentRef<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Bridges a concrete [`ArrayParentReduceRule<V, R>`] to the type-erased
/// [`DynArrayParentReduceRule<V>`] trait. Created by [`ParentRuleSet::lift`].
pub struct ParentReduceRuleAdapter<V, R> {
    rule: R,
    _phantom: PhantomData<V>,
}

impl<V: VTable, R: ArrayParentReduceRule<V>> Debug for ParentReduceRuleAdapter<V, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayParentReduceRuleAdapter")
            .field("parent", &type_name::<R::Parent>())
            .field("rule", &self.rule)
            .finish()
    }
}

impl<V: VTable, K: ArrayParentReduceRule<V>> DynArrayParentReduceRule<V>
    for ParentReduceRuleAdapter<V, K>
{
    fn matches(&self, parent: &ParentRef<'_>) -> bool {
        K::Parent::matches_parent(parent)
    }

    fn reduce_parent(
        &self,
        child: ArrayView<'_, V>,
        parent: &ParentRef<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        self.rule.reduce_parent_ref(child, parent, child_idx)
    }
}

/// A collection of [`ArrayReduceRule`]s registered for a specific encoding.
///
/// During optimization, the optimizer calls [`evaluate`](Self::evaluate) which tries each rule
/// in order. The first rule that returns `Some` wins.
pub struct ReduceRuleSet<V: VTable> {
    rules: &'static [&'static dyn ArrayReduceRule<V>],
}

impl<V: VTable> ReduceRuleSet<V> {
    /// Create a new reduction rule set with the given rules.
    pub const fn new(rules: &'static [&'static dyn ArrayReduceRule<V>]) -> Self {
        Self { rules }
    }

    /// Evaluate the reduction rules on the given array.
    pub fn evaluate(&self, array: ArrayView<'_, V>) -> VortexResult<Option<ArrayRef>> {
        for rule in self.rules.iter() {
            if let Some(reduced) = rule.reduce(array)? {
                return Ok(Some(reduced));
            }
        }
        Ok(None)
    }
}

/// A set of parent reduction rules for a specific child array encoding.
pub struct ParentRuleSet<V: VTable> {
    rules: &'static [&'static dyn DynArrayParentReduceRule<V>],
}

impl<V: VTable> ParentRuleSet<V> {
    /// Create a new parent rule set with the given rules.
    ///
    /// Use [`ParentRuleSet::lift`] to lift static rules into dynamic trait objects.
    pub const fn new(rules: &'static [&'static dyn DynArrayParentReduceRule<V>]) -> Self {
        Self { rules }
    }

    /// Lift the given rule into a dynamic trait object.
    pub const fn lift<R: ArrayParentReduceRule<V>>(
        rule: &'static R,
    ) -> &'static dyn DynArrayParentReduceRule<V> {
        // Assert that self is zero-sized
        const {
            assert!(
                !(size_of::<R>() != 0),
                "Rule must be zero-sized to be lifted"
            );
        }
        unsafe { &*(rule as *const R as *const ParentReduceRuleAdapter<V, R>) }
    }

    /// Evaluate the parent reduction rules on the given child and parent arrays.
    pub fn evaluate(
        &self,
        child: ArrayView<'_, V>,
        parent: &ParentRef<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        for rule in self.rules.iter() {
            if !rule.matches(parent) {
                continue;
            }
            if let Some(reduced) = rule.reduce_parent(child, parent, child_idx)? {
                // Debug assertions because these checks are already run elsewhere.
                #[cfg(debug_assertions)]
                {
                    vortex_error::vortex_ensure!(
                        reduced.len() == parent.len(),
                        "Reduced array length mismatch from {:?}\nFrom:\n{}\nTo:\n{}",
                        rule,
                        parent.encoding_id(),
                        reduced.encoding_id()
                    );
                    vortex_error::vortex_ensure!(
                        reduced.dtype() == parent.dtype(),
                        "Reduced array dtype mismatch from {:?}\nFrom:\n{}\nTo:\n{}",
                        rule,
                        parent.encoding_id(),
                        reduced.encoding_id()
                    );
                }

                return Ok(Some(reduced));
            }
        }
        Ok(None)
    }
}
