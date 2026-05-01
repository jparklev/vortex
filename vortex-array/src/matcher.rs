// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::array::ParentRef;

/// Trait for matching array types.
pub trait Matcher {
    type Match<'a>;

    /// Check if the given array matches this matcher type.
    fn matches(array: &ArrayRef) -> bool {
        let parent = ParentRef::from_array_ref(array);
        Self::matches_parent(&parent)
    }

    /// Try to match the given array, returning the matched view type if successful.
    ///
    /// This compatibility entry point delegates through [`ParentRef::from_array_ref`].
    /// Implement [`Matcher::try_match_parent`] for new matchers.
    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        let parent = ParentRef::from_array_ref(array);
        Self::try_match_parent(&parent)
    }

    /// Try to match a [`ParentRef`].
    ///
    /// This is the primary matching entry point. Matchers whose `Match` type requires
    /// an [`ArrayRef`] can use [`ParentRef::array_ref`] or [`ParentRef::try_array_view`],
    /// while matchers that can handle stack-allocated parents should use
    /// [`ParentRef::try_view`].
    fn try_match_parent<'a>(parent: &ParentRef<'a>) -> Option<Self::Match<'a>>;

    /// Check if the given parent matches this matcher type.
    fn matches_parent(parent: &ParentRef<'_>) -> bool {
        Self::try_match_parent(parent).is_some()
    }
}

/// Matches any array type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArray;

impl Matcher for AnyArray {
    type Match<'a> = &'a ArrayRef;

    #[inline(always)]
    fn try_match_parent<'a>(parent: &ParentRef<'a>) -> Option<Self::Match<'a>> {
        parent.array_ref()
    }

    #[inline(always)]
    fn matches_parent(parent: &ParentRef<'_>) -> bool {
        parent.array_ref().is_some()
    }
}
