# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Mojo AOT-compiled SIMD take (gather) kernels for Vortex.
#
# Each exported function gathers values from `src` at positions given by `indices`
# and writes them into `dst`. The caller (Rust) owns all three buffers — the Mojo
# side performs zero allocation.
#
# Pointers are passed as `Int` (pointer-width integer) because Mojo 0.26's
# `UnsafePointer` carries origin/mutability parameters that make it incompatible
# with `@export`. Inside each function we reconstruct typed `UnsafePointer`s via
# the `type_of` anchor pattern.
#
# SIMD width is hardcoded to 8 lanes for 4-byte types and 4 lanes for 8-byte
# types (matching AVX2 register width). The compiler will use the best available
# ISA (AVX-512, AVX2, NEON) for the gather instructions.

from std.memory import UnsafePointer

# SIMD lane counts matching 256-bit registers (AVX2 baseline).
comptime W1: Int = 32  # 1-byte values: 32 lanes
comptime W2: Int = 16  # 2-byte values: 16 lanes
comptime W4: Int = 8   # 4-byte values: 8 lanes
comptime W8: Int = 4   # 8-byte values: 4 lanes


# ---------------------------------------------------------------------------
# Generic gather implementation
# ---------------------------------------------------------------------------

@always_inline
fn _take[VT: DType, IT: DType, W: Int](
    src_addr: Int,
    idx_addr: Int,
    dst_addr: Int,
    count: Int,
):
    """Gather `count` elements: dst[i] = src[indices[i]]."""
    var _v_anchor: Scalar[VT] = 0
    var _i_anchor: Scalar[IT] = 0
    comptime VP = type_of(UnsafePointer(to=_v_anchor))
    comptime IP = type_of(UnsafePointer(to=_i_anchor))

    var src = VP(unsafe_from_address=src_addr)
    var idx = IP(unsafe_from_address=idx_addr)
    var dst = VP(unsafe_from_address=dst_addr)

    var i = 0

    # SIMD gather loop — processes W elements per iteration.
    while i + W <= count:
        var idx_vec = idx.load[width=W](i).cast[DType.uint64]()
        var gathered = src.gather(idx_vec)
        dst.store[width=W](i, gathered)
        i += W

    # Scalar remainder.
    while i < count:
        dst[i] = src[Int(idx[i])]
        i += 1


# ---------------------------------------------------------------------------
# 4-byte value types (i32 / u32 / f32)
# ---------------------------------------------------------------------------

@export("vortex_take_4byte_u8idx")
fn take_4byte_u8idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint32, DType.uint8, W4](src, idx, dst, n)

@export("vortex_take_4byte_u16idx")
fn take_4byte_u16idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint32, DType.uint16, W4](src, idx, dst, n)

@export("vortex_take_4byte_u32idx")
fn take_4byte_u32idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint32, DType.uint32, W4](src, idx, dst, n)

@export("vortex_take_4byte_u64idx")
fn take_4byte_u64idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint32, DType.uint64, W4](src, idx, dst, n)


# ---------------------------------------------------------------------------
# 8-byte value types (i64 / u64 / f64)
# ---------------------------------------------------------------------------

@export("vortex_take_8byte_u8idx")
fn take_8byte_u8idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint64, DType.uint8, W8](src, idx, dst, n)

@export("vortex_take_8byte_u16idx")
fn take_8byte_u16idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint64, DType.uint16, W8](src, idx, dst, n)

@export("vortex_take_8byte_u32idx")
fn take_8byte_u32idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint64, DType.uint32, W8](src, idx, dst, n)

@export("vortex_take_8byte_u64idx")
fn take_8byte_u64idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint64, DType.uint64, W8](src, idx, dst, n)


# ---------------------------------------------------------------------------
# 2-byte value types (i16 / u16 / f16)
# ---------------------------------------------------------------------------

@export("vortex_take_2byte_u8idx")
fn take_2byte_u8idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint16, DType.uint8, W2](src, idx, dst, n)

@export("vortex_take_2byte_u16idx")
fn take_2byte_u16idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint16, DType.uint16, W2](src, idx, dst, n)

@export("vortex_take_2byte_u32idx")
fn take_2byte_u32idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint16, DType.uint32, W2](src, idx, dst, n)

@export("vortex_take_2byte_u64idx")
fn take_2byte_u64idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint16, DType.uint64, W2](src, idx, dst, n)


# ---------------------------------------------------------------------------
# 1-byte value types (i8 / u8)
# ---------------------------------------------------------------------------

@export("vortex_take_1byte_u8idx")
fn take_1byte_u8idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint8, DType.uint8, W1](src, idx, dst, n)

@export("vortex_take_1byte_u16idx")
fn take_1byte_u16idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint8, DType.uint16, W1](src, idx, dst, n)

@export("vortex_take_1byte_u32idx")
fn take_1byte_u32idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint8, DType.uint32, W1](src, idx, dst, n)

@export("vortex_take_1byte_u64idx")
fn take_1byte_u64idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint8, DType.uint64, W1](src, idx, dst, n)
