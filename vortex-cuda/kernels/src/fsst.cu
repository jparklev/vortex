// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// FSST decompression. A thread decodes one string at a time.
//
// Byte-by-byte global writes; no per-thread output scratch and no
// alignment-aware stores yet. The 256-entry symbol table is cooperatively
// loaded into shared memory before decoding begins so every per-code
// lookup in the inner loop hits SRAM.
//
// The compressed code stream is staged in a per-thread register `chunk`
// (up to 8 bytes). Refills happen via hierarchical aligned loads: each
// step picks the largest naturally-aligned load that fits at the current
// fill position and in the remaining chunk capacity. CUDA u64 loads
// require natural 8-byte alignment, so reading `*(uint64_t*)(codes_bytes
// + in_pos)` directly raises CUDA_ERROR_MISALIGNED_ADDRESS for
// non-8-aligned `in_pos`. The hierarchical fill walks up to a u64
// boundary with smaller aligned loads (mirroring `Scratch::drain` on the
// output side); once aligned and chunk is empty, every fill is a single
// u64 load.
//
//   load    gate                                       ptx
//   ------  -----------------------------------------  ----------------
//    u64    chunk_bytes == 0, fill_pos % 8 == 0        ld.global.u64
//    u32    chunk_bytes + 4 ≤ 8, fill_pos % 4 == 0     ld.global.u32
//    u16    chunk_bytes + 2 ≤ 8, fill_pos % 2 == 0     ld.global.u16
//    u8     (always)                                   ld.global.u8
//
// symbols[i] is the 8-byte symbol for code i, stored little-endian in a
// u64: byte 0 lives in bits 0-7, byte 1 in bits 8-15, etc.
// symbol_lengths[i] is the symbol's valid byte count (1-8). Code 255 is
// the escape marker: the next input byte is emitted as a literal.
//
// codes_offsets is templated over the four unsigned integer widths
// (u8/u16/u32/u64). output_offsets is uint64_t.

template <typename OffT>
struct FSSTArgs {
    // Compressed FSST code stream, contiguous across all strings. String
    // `sid`'s codes live in `[codes_offsets[sid], codes_offsets[sid + 1])`.
    const uint8_t *__restrict codes_bytes;
    // Per-string offsets into `codes_bytes`, length `num_strings + 1`.
    const OffT *__restrict codes_offsets;
    // FSST symbol table.
    const uint64_t *__restrict symbols;
    // Length in bytes (1..=8) of each entry in `symbols`. The remaining bits
    // are unspecified.
    const uint8_t *__restrict symbol_lengths;
    // Buffer to write decoded data into.
    uint8_t *__restrict output_bytes;
    // Per-string offsets into `output_bytes`, length `num_strings + 1`.
    const uint64_t *__restrict output_offsets;
    // Validity of each string.
    const uint8_t *__restrict validity_bits;
};

// Refill `chunk` from `codes_bytes + (in_pos + chunk_bytes)` using
// hierarchical aligned loads. Loop invariant: `chunk[0..chunk_bytes]` ==
// `codes_bytes[in_pos..in_pos + chunk_bytes]`. Each step picks the
// largest naturally-aligned load that fits the current fill position's
// alignment AND in the remaining 8 - chunk_bytes capacity, then ORs the
// loaded bytes into chunk at byte offset `chunk_bytes`. Once chunk is
// full (8 bytes) or we run out of input, the loop exits.
template <typename OffT>
__device__ inline void fsst_chunk_refill(const uint8_t *__restrict codes_bytes,
                                         OffT in_pos,
                                         OffT in_end,
                                         uint64_t &chunk,
                                         uint32_t &chunk_bytes) {
#pragma unroll 1
    while (chunk_bytes < 8) {
        const OffT fill_pos = in_pos + (OffT)chunk_bytes;
        if (fill_pos >= in_end) {
            break;
        }
        const int32_t remaining = (int32_t)(in_end - fill_pos);
        const uint32_t aln = (uint32_t)fill_pos & 7u;
        if (chunk_bytes == 0 && aln == 0 && remaining >= 8) {
            chunk = *reinterpret_cast<const uint64_t *>(codes_bytes + fill_pos);
            chunk_bytes = 8;
        } else if (chunk_bytes + 4u <= 8u && (aln & 3u) == 0 && remaining >= 4) {
            const uint64_t v = *reinterpret_cast<const uint32_t *>(codes_bytes + fill_pos);
            chunk |= v << (8u * chunk_bytes);
            chunk_bytes += 4;
        } else if (chunk_bytes + 2u <= 8u && (aln & 1u) == 0 && remaining >= 2) {
            const uint64_t v = *reinterpret_cast<const uint16_t *>(codes_bytes + fill_pos);
            chunk |= v << (8u * chunk_bytes);
            chunk_bytes += 2;
        } else {
            const uint64_t v = codes_bytes[fill_pos];
            chunk |= v << (8u * chunk_bytes);
            chunk_bytes += 1;
        }
    }
}

template <typename OffT>
__device__ inline void fsst_decode_string(const FSSTArgs<OffT> &args,
                                          const uint64_t *sm_symbols,
                                          const uint8_t *sm_symbol_lengths,
                                          uint64_t sid) {
    if (((args.validity_bits[sid >> 3] >> (sid & 7u)) & 1u) == 0u) {
        return;
    }

    OffT in_pos = args.codes_offsets[sid];
    const OffT in_end = args.codes_offsets[sid + 1];
    uint64_t out_pos = args.output_offsets[sid];

    // `chunk` holds the next up-to-8 bytes of the code stream in a register.
    // Byte 0 of `chunk` is always `codes_bytes[in_pos]`. `chunk_bytes` is the
    // count of valid bytes still in `chunk`.
    uint64_t chunk = 0;
    uint32_t chunk_bytes = 0;

    while (in_pos < in_end) {
        // Refill when we can't cover a worst-case 2-byte escape.
        if (chunk_bytes < 2) {
            fsst_chunk_refill<OffT>(args.codes_bytes, in_pos, in_end, chunk, chunk_bytes);
        }

        const uint8_t code = (uint8_t)(chunk & 0xFFu);
        if (code == 255) {
            // Escape: next byte is a literal, already sitting in chunk[1].
            args.output_bytes[out_pos] = (uint8_t)((chunk >> 8u) & 0xFFu);
            in_pos += (OffT)2;
            out_pos += 1;
            chunk >>= 16;
            chunk_bytes -= 2;
        } else {
            const uint64_t sym = sm_symbols[code];
            const uint8_t len = sm_symbol_lengths[code];
#pragma unroll 1
            for (uint8_t i = 0; i < len; ++i) {
                args.output_bytes[out_pos + i] = (uint8_t)(sym >> (8u * i));
            }
            in_pos += (OffT)1;
            out_pos += len;
            chunk >>= 8;
            chunk_bytes -= 1;
        }
    }
}

#define GENERATE_FSST_KERNEL(suffix, OffT)                                                                   \
    extern "C" __global__ void fsst_##suffix(const uint8_t *__restrict codes_bytes,                          \
                                             const OffT *__restrict codes_offsets,                           \
                                             const uint64_t *__restrict symbols,                             \
                                             const uint8_t *__restrict symbol_lengths,                       \
                                             const uint64_t *__restrict output_offsets,                      \
                                             const uint8_t *__restrict validity_bits,                        \
                                             uint8_t *__restrict output_bytes,                               \
                                             uint64_t num_strings) {                                         \
        const FSSTArgs<OffT> args = {                                                                        \
            codes_bytes,                                                                                     \
            codes_offsets,                                                                                   \
            symbols,                                                                                         \
            symbol_lengths,                                                                                  \
            output_bytes,                                                                                    \
            output_offsets,                                                                                  \
            validity_bits,                                                                                   \
        };                                                                                                   \
                                                                                                             \
        __shared__ uint64_t sm_symbols[256];                                                                 \
        __shared__ uint8_t sm_symbol_lengths[256];                                                           \
        for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) {                                           \
            sm_symbols[i] = symbols[i];                                                                      \
            sm_symbol_lengths[i] = symbol_lengths[i];                                                        \
        }                                                                                                    \
        __syncthreads();                                                                                     \
                                                                                                             \
        const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;                      \
        const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;                              \
        const uint64_t block_end = (block_start + elements_per_block < num_strings)                          \
                                       ? (block_start + elements_per_block)                                  \
                                       : num_strings;                                                        \
                                                                                                             \
        for (uint64_t sid = block_start + threadIdx.x; sid < block_end; sid += blockDim.x) {                 \
            fsst_decode_string<OffT>(args, sm_symbols, sm_symbol_lengths, sid);                              \
        }                                                                                                    \
    }

GENERATE_FSST_KERNEL(u8, uint8_t)
GENERATE_FSST_KERNEL(u16, uint16_t)
GENERATE_FSST_KERNEL(u32, uint32_t)
GENERATE_FSST_KERNEL(u64, uint64_t)
