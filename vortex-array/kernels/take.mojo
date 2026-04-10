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
    """Gather `count` elements: dst[i] = src[indices[i]].

    The inner loop is 4x unrolled to keep the CPU's gather pipeline fed with
    independent loads (critical for throughput on Intel Skylake+ and AMD Zen3+).
    """
    var _v_anchor: Scalar[VT] = 0
    var _i_anchor: Scalar[IT] = 0
    comptime VP = type_of(UnsafePointer(to=_v_anchor))
    comptime IP = type_of(UnsafePointer(to=_i_anchor))

    var src = VP(unsafe_from_address=src_addr)
    var idx = IP(unsafe_from_address=idx_addr)
    var dst = VP(unsafe_from_address=dst_addr)

    var i = 0

    # 4x unrolled SIMD gather — keeps gather units saturated with independent
    # loads for maximum instruction-level parallelism.
    while i + W * 4 <= count:
        var g0 = src.gather(idx.load[width=W](i))
        var g1 = src.gather(idx.load[width=W](i + W))
        var g2 = src.gather(idx.load[width=W](i + W * 2))
        var g3 = src.gather(idx.load[width=W](i + W * 3))

        dst.store[width=W](i, g0)
        dst.store[width=W](i + W, g1)
        dst.store[width=W](i + W * 2, g2)
        dst.store[width=W](i + W * 3, g3)
        i += W * 4

    # Single-vector remainder.
    while i + W <= count:
        dst.store[width=W](i, src.gather(idx.load[width=W](i)))
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


# ---------------------------------------------------------------------------
# Filter kernels (gather by usize indices from mask)
#
# These are used by the primitive filter path when the mask is sparse (<80%
# selectivity). The Rust side converts the bitmap to a &[usize] index array
# and passes it here. On x86_64 usize = u64, so these are gathers with
# u64 element indices.
# ---------------------------------------------------------------------------

@export("vortex_filter_1byte")
fn filter_1byte(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint8, DType.uint64, W1](src, idx, dst, n)

@export("vortex_filter_2byte")
fn filter_2byte(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint16, DType.uint64, W2](src, idx, dst, n)

@export("vortex_filter_4byte")
fn filter_4byte(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint32, DType.uint64, W4](src, idx, dst, n)

@export("vortex_filter_8byte")
fn filter_8byte(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint64, DType.uint64, W8](src, idx, dst, n)


# ---------------------------------------------------------------------------
# Run-end decode kernels (SIMD broadcast + store)
#
# Decodes run-end encoded arrays: for each run, broadcast the value into
# a SIMD register and write it to the output buffer. 3-4x faster than
# scalar fill for run_length >= 8.
#
# Parameters:
#   ends:     pointer to u32 run-end positions (monotonically increasing)
#   values:   pointer to values (one per run, same byte width as output)
#   dst:      pointer to output buffer (pre-allocated by Rust)
#   num_runs: number of runs
# ---------------------------------------------------------------------------

@always_inline
fn _runend_decode[VT: DType, W: Int](
    ends_addr: Int,
    values_addr: Int,
    dst_addr: Int,
    num_runs: Int,
):
    """Decode run-end encoded data using SIMD broadcast fill."""
    var _e: UInt32 = 0
    var _v: Scalar[VT] = 0
    comptime EP = type_of(UnsafePointer(to=_e))
    comptime VP = type_of(UnsafePointer(to=_v))

    var ends = EP(unsafe_from_address=ends_addr)
    var values = VP(unsafe_from_address=values_addr)
    var dst = VP(unsafe_from_address=dst_addr)

    var pos = 0
    for run in range(num_runs):
        var end = Int(ends[run])
        var val = values[run]
        var run_len = end - pos

        # SIMD broadcast fill
        var vec = SIMD[VT, W](val)
        var i = 0
        while i + W <= run_len:
            dst.store[width=W](pos + i, vec)
            i += W

        # Scalar remainder
        while i < run_len:
            dst[pos + i] = val
            i += 1

        pos = end


@export("vortex_runend_decode_1byte")
fn runend_decode_1byte(ends: Int, values: Int, dst: Int, num_runs: Int):
    _runend_decode[DType.uint8, W1](ends, values, dst, num_runs)

@export("vortex_runend_decode_2byte")
fn runend_decode_2byte(ends: Int, values: Int, dst: Int, num_runs: Int):
    _runend_decode[DType.uint16, W2](ends, values, dst, num_runs)

@export("vortex_runend_decode_4byte")
fn runend_decode_4byte(ends: Int, values: Int, dst: Int, num_runs: Int):
    _runend_decode[DType.uint32, W4](ends, values, dst, num_runs)

@export("vortex_runend_decode_8byte")
fn runend_decode_8byte(ends: Int, values: Int, dst: Int, num_runs: Int):
    _runend_decode[DType.uint64, W8](ends, values, dst, num_runs)
