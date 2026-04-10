// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FFI bridge to AOT-compiled Mojo SIMD take kernels.
//!
//! The Mojo kernels are compiled during `build.rs` and statically linked. Each exported
//! symbol operates on raw pointer-width integers (`usize`) — Rust allocates the output
//! buffer, passes addresses as `usize`, and Mojo writes directly into the buffer.
//!
//! Value types are dispatched by byte width (1/2/4/8) since the gather operation is
//! agnostic to signedness. Rust reinterprets the slice pointers accordingly.

use std::mem::size_of;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::vtable::Primitive;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

use super::TakeImpl;

// ---------------------------------------------------------------------------
// Mojo extern declarations — pointers passed as usize (Mojo `Int`).
// One symbol per (value_byte_width, index_type) pair.
// ---------------------------------------------------------------------------

unsafe extern "C" {
    // 1-byte values
    fn vortex_take_1byte_u8idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_1byte_u16idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_1byte_u32idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_1byte_u64idx(src: usize, idx: usize, dst: usize, len: usize);

    // 2-byte values
    fn vortex_take_2byte_u8idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_2byte_u16idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_2byte_u32idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_2byte_u64idx(src: usize, idx: usize, dst: usize, len: usize);

    // 4-byte values
    fn vortex_take_4byte_u8idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_4byte_u16idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_4byte_u32idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_4byte_u64idx(src: usize, idx: usize, dst: usize, len: usize);

    // 8-byte values
    fn vortex_take_8byte_u8idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_8byte_u16idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_8byte_u32idx(src: usize, idx: usize, dst: usize, len: usize);
    fn vortex_take_8byte_u64idx(src: usize, idx: usize, dst: usize, len: usize);
}

pub(super) struct TakeKernelMojo;

impl TakeImpl for TakeKernelMojo {
    fn take(
        &self,
        array: ArrayView<'_, Primitive>,
        indices: ArrayView<'_, Primitive>,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        match_each_native_ptype!(array.ptype(), |V| {
            match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
                let buffer = take_mojo::<V, I>(array.as_slice(), indices.as_slice());
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            })
        })
    }
}

/// Dispatch to the appropriate Mojo kernel based on value byte width and index type.
fn take_mojo<V: NativePType, I: UnsignedPType>(values: &[V], indices: &[I]) -> Buffer<V> {
    let len = indices.len();
    let mut buffer = BufferMut::<V>::with_capacity(len);

    let dst = buffer.spare_capacity_mut().as_mut_ptr().cast::<V>();
    let src = values.as_ptr();
    let idx = indices.as_ptr();

    // SAFETY: All three pointers are valid for their respective lengths. The Mojo kernel
    // writes exactly `len` elements to `dst`, which has capacity for `len` elements.
    // We dispatch by value byte-width since the gather is signedness-agnostic.
    unsafe {
        match (size_of::<V>(), I::PTYPE) {
            (1, PType::U8) => {
                vortex_take_1byte_u8idx(src as usize, idx as usize, dst as usize, len)
            }
            (1, PType::U16) => {
                vortex_take_1byte_u16idx(src as usize, idx as usize, dst as usize, len)
            }
            (1, PType::U32) => {
                vortex_take_1byte_u32idx(src as usize, idx as usize, dst as usize, len)
            }
            (1, PType::U64) => {
                vortex_take_1byte_u64idx(src as usize, idx as usize, dst as usize, len)
            }

            (2, PType::U8) => {
                vortex_take_2byte_u8idx(src as usize, idx as usize, dst as usize, len)
            }
            (2, PType::U16) => {
                vortex_take_2byte_u16idx(src as usize, idx as usize, dst as usize, len)
            }
            (2, PType::U32) => {
                vortex_take_2byte_u32idx(src as usize, idx as usize, dst as usize, len)
            }
            (2, PType::U64) => {
                vortex_take_2byte_u64idx(src as usize, idx as usize, dst as usize, len)
            }

            (4, PType::U8) => {
                vortex_take_4byte_u8idx(src as usize, idx as usize, dst as usize, len)
            }
            (4, PType::U16) => {
                vortex_take_4byte_u16idx(src as usize, idx as usize, dst as usize, len)
            }
            (4, PType::U32) => {
                vortex_take_4byte_u32idx(src as usize, idx as usize, dst as usize, len)
            }
            (4, PType::U64) => {
                vortex_take_4byte_u64idx(src as usize, idx as usize, dst as usize, len)
            }

            (8, PType::U8) => {
                vortex_take_8byte_u8idx(src as usize, idx as usize, dst as usize, len)
            }
            (8, PType::U16) => {
                vortex_take_8byte_u16idx(src as usize, idx as usize, dst as usize, len)
            }
            (8, PType::U32) => {
                vortex_take_8byte_u32idx(src as usize, idx as usize, dst as usize, len)
            }
            (8, PType::U64) => {
                vortex_take_8byte_u64idx(src as usize, idx as usize, dst as usize, len)
            }

            _ => unreachable!("unsupported value size / index type combination"),
        }

        buffer.set_len(len);
    }

    buffer.freeze()
}
