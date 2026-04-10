// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::PrimitiveArrayExt;
pub use array::PrimitiveData;
pub use array::PrimitiveDataParts;
pub use array::chunk_range;
pub use array::patch_chunk;
pub use vtable::PrimitiveArray;

pub(crate) mod compute;

mod vtable;
pub use compute::rules::PrimitiveMaskedValidityRule;
#[doc(hidden)]
pub use compute::take::bench_take_avx2;
#[doc(hidden)]
pub use compute::take::bench_take_mojo;
#[doc(hidden)]
pub use compute::take::bench_take_scalar;
pub use vtable::Primitive;

mod native_value;
pub use native_value::NativeValue;

#[cfg(test)]
mod tests;
