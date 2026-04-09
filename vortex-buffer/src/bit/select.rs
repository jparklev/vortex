// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::count_ones::align_offset_len;

/// Returns the position of the `nth` set bit (0-indexed) within the logical range
/// `[offset, offset + len)` of the given byte slice.
///
/// The returned position is relative to the logical start (i.e., 0-indexed from `offset`).
///
/// Uses architecture-specific optimizations:
/// - **aarch64**: NEON `vcnt`-based popcount for the word-level scan.
/// - **x86_64 + BMI2**: `pdep` + `tzcnt` for the final in-word select.
/// - **Scalar fallback**: 4× unrolled word scan with `count_ones`, byte-level narrowing.
#[inline]
pub fn bit_select(bytes: &[u8], offset: usize, len: usize, nth: usize) -> usize {
    let (head, middle, tail) = align_offset_len(bytes, offset, len);
    let mut remaining = nth;
    let mut pos = 0usize;

    // ── partial first byte ──────────────────────────────────────────────
    if let Some(head) = head {
        let count = head.count_ones() as usize;
        if remaining < count {
            return select_in_byte(head, remaining);
        }
        remaining -= count;
        let start_bit = offset % 8;
        pos = (8 - start_bit).min(len);
    }

    // ── aligned middle bytes ────────────────────────────────────────────
    if !middle.is_empty() {
        let (words, tail_bytes) = middle.as_chunks::<8>();

        let (rem, new_pos, word_idx) = scan_words(words, remaining, pos);
        remaining = rem;
        pos = new_pos;

        if word_idx < words.len() {
            let word = u64::from_le_bytes(words[word_idx]);
            return pos + select_in_word(word, remaining);
        }

        // Remaining aligned bytes that don't fill a full u64.
        for &byte in tail_bytes {
            let count = byte.count_ones() as usize;
            if remaining < count {
                return pos + select_in_byte(byte, remaining);
            }
            remaining -= count;
            pos += 8;
        }
    }

    // ── partial last byte ───────────────────────────────────────────────
    if let Some(tail) = tail {
        debug_assert!(
            remaining < tail.count_ones() as usize,
            "bit_select: nth={nth} out of bounds"
        );
        return pos + select_in_byte(tail, remaining);
    }

    unreachable!("bit_select: nth={nth} exceeds set bit count")
}

// ── Batch select (sorted ranks, single pass) ───────────────────────────

/// Select positions for multiple ranks in a single pass over the bitmap.
///
/// `sorted_ranks` must be sorted in non-decreasing order; each value must be
/// less than the total number of set bits in `[offset, offset+len)`.
/// Results are written to `out[0..sorted_ranks.len()]`.
pub fn bit_select_sorted_batch(
    bytes: &[u8],
    offset: usize,
    len: usize,
    sorted_ranks: &[usize],
    out: &mut [usize],
) {
    debug_assert!(out.len() >= sorted_ranks.len());
    if sorted_ranks.is_empty() {
        return;
    }

    let (head, middle, tail) = align_offset_len(bytes, offset, len);
    let mut cumul = 0usize;
    let mut pos = 0usize;
    let mut ri = 0usize; // index into sorted_ranks / out

    // ── head byte ───────────────────────────────────────────────────
    if let Some(head) = head {
        let count = head.count_ones() as usize;
        while ri < sorted_ranks.len() && sorted_ranks[ri] < cumul + count {
            out[ri] = select_in_byte(head, sorted_ranks[ri] - cumul);
            ri += 1;
        }
        cumul += count;
        let start_bit = offset % 8;
        pos = (8 - start_bit).min(len);
        if ri >= sorted_ranks.len() {
            return;
        }
    }

    // ── middle bytes ────────────────────────────────────────────────
    if !middle.is_empty() {
        let (words, tail_bytes) = middle.as_chunks::<8>();

        scan_words_batch_impl(words, sorted_ranks, out, &mut cumul, &mut pos, &mut ri);
        if ri >= sorted_ranks.len() {
            return;
        }

        for &byte in tail_bytes {
            let count = byte.count_ones() as usize;
            while ri < sorted_ranks.len() && sorted_ranks[ri] < cumul + count {
                out[ri] = pos + select_in_byte(byte, sorted_ranks[ri] - cumul);
                ri += 1;
            }
            cumul += count;
            pos += 8;
            if ri >= sorted_ranks.len() {
                return;
            }
        }
    }

    // ── tail byte ───────────────────────────────────────────────────
    if let Some(tail) = tail {
        let count = tail.count_ones() as usize;
        while ri < sorted_ranks.len() && sorted_ranks[ri] < cumul + count {
            out[ri] = pos + select_in_byte(tail, sorted_ranks[ri] - cumul);
            ri += 1;
        }
    }
}

// ── aarch64 NEON batch scan ─────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[allow(clippy::cast_possible_truncation)]
fn scan_words_batch_impl(
    words: &[[u8; 8]],
    sorted_ranks: &[usize],
    out: &mut [usize],
    cumul: &mut usize,
    pos: &mut usize,
    ri: &mut usize,
) {
    use std::arch::aarch64::vcntq_u8;
    use std::arch::aarch64::vgetq_lane_u64;
    use std::arch::aarch64::vld1q_u8;
    use std::arch::aarch64::vpaddlq_u8;
    use std::arch::aarch64::vpaddlq_u16;
    use std::arch::aarch64::vpaddlq_u32;

    let mut wi = 0;

    // 4-word NEON blocks — skip entire blocks when no targets inside.
    while wi + 4 <= words.len() && *ri < sorted_ranks.len() {
        let ptr = words[wi].as_ptr();
        // SAFETY: wi + 4 <= words.len() guarantees 32 contiguous bytes.
        let (c0, c1, c2, c3) = unsafe {
            let pop_lo = vcntq_u8(vld1q_u8(ptr));
            let pop_hi = vcntq_u8(vld1q_u8(ptr.add(16)));
            let sums_lo = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(pop_lo)));
            let sums_hi = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(pop_hi)));
            (
                vgetq_lane_u64::<0>(sums_lo) as usize,
                vgetq_lane_u64::<1>(sums_lo) as usize,
                vgetq_lane_u64::<0>(sums_hi) as usize,
                vgetq_lane_u64::<1>(sums_hi) as usize,
            )
        };
        let total = c0 + c1 + c2 + c3;

        if sorted_ranks[*ri] >= *cumul + total {
            *cumul += total;
            *pos += 256;
            wi += 4;
            continue;
        }

        // At least one target in this block — emit per word.
        let counts = [c0, c1, c2, c3];
        for (j, &count) in counts.iter().enumerate() {
            while *ri < sorted_ranks.len() && sorted_ranks[*ri] < *cumul + count {
                let word = u64::from_le_bytes(words[wi + j]);
                out[*ri] = *pos + select_in_word(word, sorted_ranks[*ri] - *cumul);
                *ri += 1;
            }
            *cumul += count;
            *pos += 64;
        }
        wi += 4;
    }

    // Remaining words, scalar.
    while wi < words.len() && *ri < sorted_ranks.len() {
        let word = u64::from_le_bytes(words[wi]);
        let count = word.count_ones() as usize;
        while *ri < sorted_ranks.len() && sorted_ranks[*ri] < *cumul + count {
            out[*ri] = *pos + select_in_word(word, sorted_ranks[*ri] - *cumul);
            *ri += 1;
        }
        *cumul += count;
        *pos += 64;
        wi += 1;
    }
}

// ── Scalar batch scan ───────────────────────────────────────────────────

#[cfg(not(target_arch = "aarch64"))]
fn scan_words_batch_impl(
    words: &[[u8; 8]],
    sorted_ranks: &[usize],
    out: &mut [usize],
    cumul: &mut usize,
    pos: &mut usize,
    ri: &mut usize,
) {
    let mut wi = 0;

    while wi + 4 <= words.len() && *ri < sorted_ranks.len() {
        let c0 = u64::from_le_bytes(words[wi]).count_ones() as usize;
        let c1 = u64::from_le_bytes(words[wi + 1]).count_ones() as usize;
        let c2 = u64::from_le_bytes(words[wi + 2]).count_ones() as usize;
        let c3 = u64::from_le_bytes(words[wi + 3]).count_ones() as usize;
        let total = c0 + c1 + c2 + c3;

        if sorted_ranks[*ri] >= *cumul + total {
            *cumul += total;
            *pos += 256;
            wi += 4;
            continue;
        }

        let counts = [c0, c1, c2, c3];
        for (j, &count) in counts.iter().enumerate() {
            while *ri < sorted_ranks.len() && sorted_ranks[*ri] < *cumul + count {
                let word = u64::from_le_bytes(words[wi + j]);
                out[*ri] = *pos + select_in_word(word, sorted_ranks[*ri] - *cumul);
                *ri += 1;
            }
            *cumul += count;
            *pos += 64;
        }
        wi += 4;
    }

    while wi < words.len() && *ri < sorted_ranks.len() {
        let word = u64::from_le_bytes(words[wi]);
        let count = word.count_ones() as usize;
        while *ri < sorted_ranks.len() && sorted_ranks[*ri] < *cumul + count {
            out[*ri] = *pos + select_in_word(word, sorted_ranks[*ri] - *cumul);
            *ri += 1;
        }
        *cumul += count;
        *pos += 64;
        wi += 1;
    }
}

// ── Word-level scan ─────────────────────────────────────────────────────

/// Scan `words` accumulating popcounts. Returns `(remaining, position, word_index)`.
///
/// If `word_index < words.len()`, the target bit is inside that word and `remaining`
/// is the rank *within* that word. Otherwise all words were consumed.
#[inline]
fn scan_words(words: &[[u8; 8]], remaining: usize, pos: usize) -> (usize, usize, usize) {
    scan_words_impl(words, remaining, pos)
}

// ── aarch64 NEON scan ───────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[allow(clippy::cast_possible_truncation)] // u64 → usize is lossless on aarch64 (64-bit)
#[inline]
fn scan_words_impl(
    words: &[[u8; 8]],
    mut remaining: usize,
    mut pos: usize,
) -> (usize, usize, usize) {
    use std::arch::aarch64::vcntq_u8;
    use std::arch::aarch64::vgetq_lane_u64;
    use std::arch::aarch64::vld1q_u8;
    use std::arch::aarch64::vpaddlq_u8;
    use std::arch::aarch64::vpaddlq_u16;
    use std::arch::aarch64::vpaddlq_u32;

    let mut idx = 0;

    // Process 4 u64 words at a time using two 128-bit NEON registers.
    while idx + 4 <= words.len() {
        let ptr = words[idx].as_ptr();
        // SAFETY: idx + 4 <= words.len() guarantees 32 contiguous bytes from ptr.
        // NEON vld1q_u8 supports unaligned access.
        let (count_0, count_1, count_2, count_3) = unsafe {
            let pop_lo = vcntq_u8(vld1q_u8(ptr));
            let pop_hi = vcntq_u8(vld1q_u8(ptr.add(16)));
            let sums_lo = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(pop_lo)));
            let sums_hi = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(pop_hi)));
            (
                vgetq_lane_u64::<0>(sums_lo) as usize,
                vgetq_lane_u64::<1>(sums_lo) as usize,
                vgetq_lane_u64::<0>(sums_hi) as usize,
                vgetq_lane_u64::<1>(sums_hi) as usize,
            )
        };

        let total = count_0 + count_1 + count_2 + count_3;
        if remaining >= total {
            remaining -= total;
            pos += 256;
            idx += 4;
            continue;
        }

        // Narrow down to the exact word.
        if remaining < count_0 {
            return (remaining, pos, idx);
        }
        remaining -= count_0;
        pos += 64;
        if remaining < count_1 {
            return (remaining, pos, idx + 1);
        }
        remaining -= count_1;
        pos += 64;
        if remaining < count_2 {
            return (remaining, pos, idx + 2);
        }
        remaining -= count_2;
        pos += 64;
        return (remaining, pos, idx + 3);
    }

    // Process pairs.
    while idx + 2 <= words.len() {
        let ptr = words[idx].as_ptr();
        // SAFETY: idx + 2 <= words.len() guarantees 16 contiguous bytes.
        let (count_0, count_1) = unsafe {
            let pop = vcntq_u8(vld1q_u8(ptr));
            let sums = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(pop)));
            (
                vgetq_lane_u64::<0>(sums) as usize,
                vgetq_lane_u64::<1>(sums) as usize,
            )
        };
        let total = count_0 + count_1;
        if remaining < total {
            if remaining < count_0 {
                return (remaining, pos, idx);
            }
            return (remaining - count_0, pos + 64, idx + 1);
        }
        remaining -= total;
        pos += 128;
        idx += 2;
    }

    // Single trailing word.
    if idx < words.len() {
        let word = u64::from_le_bytes(words[idx]);
        let count = word.count_ones() as usize;
        if remaining < count {
            return (remaining, pos, idx);
        }
        remaining -= count;
        pos += 64;
        idx += 1;
    }

    (remaining, pos, idx)
}

// ── Scalar scan (x86_64 / generic) ─────────────────────────────────────

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn scan_words_impl(
    words: &[[u8; 8]],
    mut remaining: usize,
    mut pos: usize,
) -> (usize, usize, usize) {
    let mut idx = 0;

    // 4× unrolled: the four independent `count_ones` calls pipeline well.
    while idx + 4 <= words.len() {
        let count_0 = u64::from_le_bytes(words[idx]).count_ones() as usize;
        let count_1 = u64::from_le_bytes(words[idx + 1]).count_ones() as usize;
        let count_2 = u64::from_le_bytes(words[idx + 2]).count_ones() as usize;
        let count_3 = u64::from_le_bytes(words[idx + 3]).count_ones() as usize;
        let total = count_0 + count_1 + count_2 + count_3;

        if remaining >= total {
            remaining -= total;
            pos += 256;
            idx += 4;
            continue;
        }

        if remaining < count_0 {
            return (remaining, pos, idx);
        }
        remaining -= count_0;
        pos += 64;
        if remaining < count_1 {
            return (remaining, pos, idx + 1);
        }
        remaining -= count_1;
        pos += 64;
        if remaining < count_2 {
            return (remaining, pos, idx + 2);
        }
        remaining -= count_2;
        pos += 64;
        return (remaining, pos, idx + 3);
    }

    while idx < words.len() {
        let word = u64::from_le_bytes(words[idx]);
        let count = word.count_ones() as usize;
        if remaining < count {
            return (remaining, pos, idx);
        }
        remaining -= count;
        pos += 64;
        idx += 1;
    }

    (remaining, pos, idx)
}

// ── In-word select ──────────────────────────────────────────────────────

/// Position of the `nth` set bit inside a u64 (0-indexed, little-endian bit order).
#[inline]
fn select_in_word(word: u64, nth: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            // SAFETY: runtime detection guarantees the required target feature.
            return unsafe { select_in_word_bmi2(word, nth) };
        }
    }
    select_in_word_scalar(word, nth)
}

/// BMI2: deposit a single bit at the nth set-bit position, then count trailing zeros.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn select_in_word_bmi2(word: u64, nth: usize) -> usize {
    use std::arch::x86_64::_pdep_u64;
    use std::arch::x86_64::_tzcnt_u64;

    use vortex_error::VortexExpect;

    usize::try_from(unsafe { _tzcnt_u64(_pdep_u64(1u64 << nth, word)) })
        .vortex_expect("safe to convert tzcnt result to usize")
}

/// Scalar: narrow to the correct byte, then clear `nth` lowest set bits and trailing-zeros.
#[inline]
fn select_in_word_scalar(word: u64, mut nth: usize) -> usize {
    let bytes = word.to_le_bytes();
    let mut bit_offset = 0usize;
    for &byte in &bytes {
        let count = byte.count_ones() as usize;
        if nth < count {
            return bit_offset + select_in_byte(byte, nth);
        }
        nth -= count;
        bit_offset += 8;
    }
    unreachable!("select_in_word: nth exceeds popcount")
}

// ── In-byte select ──────────────────────────────────────────────────────

/// Position of the `nth` set bit inside a byte (0-indexed, LSB-first).
///
/// Clears the lowest `nth` set bits, then uses `trailing_zeros`.
#[inline]
fn select_in_byte(byte: u8, nth: usize) -> usize {
    debug_assert!(nth < byte.count_ones() as usize);
    let mut bits = u32::from(byte);
    for _ in 0..nth {
        bits &= bits - 1; // clear lowest set bit
    }
    bits.trailing_zeros() as usize
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation)]

    use rstest::rstest;

    use super::*;

    #[test]
    fn test_select_all_set() {
        // Every bit is set — select(n) == n.
        let buf = [0xFFu8; 16]; // 128 bits, all set
        for nth in 0..128 {
            assert_eq!(bit_select(&buf, 0, 128, nth), nth, "nth={nth}");
        }
    }

    #[test]
    fn test_select_every_other() {
        // 0b01010101 repeated: bits 0,2,4,6 of each byte are set.
        let buf = [0x55u8; 16]; // 128 bits, 64 set
        for nth in 0..64 {
            assert_eq!(bit_select(&buf, 0, 128, nth), nth * 2, "nth={nth}");
        }
    }

    #[test]
    fn test_select_single_bit() {
        // Only bit 42 is set.
        let mut buf = [0u8; 16];
        buf[42 / 8] |= 1 << (42 % 8);
        assert_eq!(bit_select(&buf, 0, 128, 0), 42);
    }

    #[rstest]
    #[case(0, 128)]
    #[case(3, 100)]
    #[case(7, 50)]
    #[case(1, 7)]
    #[case(5, 5)]
    #[case(0, 1)]
    #[case(0, 64)]
    #[case(1, 64)]
    #[case(0, 65)]
    #[case(3, 256)]
    fn test_select_agrees_with_naive(#[case] offset: usize, #[case] len: usize) {
        let total_bits = offset + len;
        let total_bytes = total_bits.div_ceil(8);
        // Deterministic pattern with moderate density.
        let buf: Vec<u8> = (0..total_bytes)
            .map(|i| ((i.wrapping_mul(0x9E) ^ 0xA5) & 0xFF) as u8)
            .collect();

        // Collect set-bit positions naively.
        let expected: Vec<usize> = (0..len)
            .filter(|&i| {
                let phys = offset + i;
                (buf[phys / 8] >> (phys % 8)) & 1 == 1
            })
            .collect();

        for (nth, &expected_pos) in expected.iter().enumerate() {
            assert_eq!(
                bit_select(&buf, offset, len, nth),
                expected_pos,
                "offset={offset} len={len} nth={nth}"
            );
        }
    }

    #[test]
    fn test_select_large_buffer() {
        // ~64 KB buffer, ~50% density.
        let len = 65_536 * 8;
        let buf: Vec<u8> = (0u32..65_536)
            .map(|i| ((i.wrapping_mul(0x37) ^ 0xBC) & 0xFF) as u8)
            .collect();

        let true_count = buf.iter().map(|b| b.count_ones() as usize).sum::<usize>();

        // Spot-check a few positions.
        let first = bit_select(&buf, 0, len, 0);
        let last = bit_select(&buf, 0, len, true_count - 1);
        assert!(first < len);
        assert!(last < len);
        assert!(first <= last);

        // Verify the found positions are actually set.
        assert_ne!(buf[first / 8] & (1 << (first % 8)), 0);
        assert_ne!(buf[last / 8] & (1 << (last % 8)), 0);
    }

    #[test]
    fn test_select_in_word_basic() {
        // 0b1010_1010 = 0xAA — bits 1,3,5,7 are set.
        let word = 0x00000000_000000AAu64;
        assert_eq!(select_in_word(word, 0), 1);
        assert_eq!(select_in_word(word, 1), 3);
        assert_eq!(select_in_word(word, 2), 5);
        assert_eq!(select_in_word(word, 3), 7);
    }

    #[test]
    fn test_select_in_word_all_set() {
        let word = u64::MAX;
        for nth in 0..64 {
            assert_eq!(select_in_word(word, nth), nth, "nth={nth}");
        }
    }

    #[test]
    fn test_select_in_byte_basic() {
        assert_eq!(select_in_byte(0b1010_1010, 0), 1);
        assert_eq!(select_in_byte(0b1010_1010, 1), 3);
        assert_eq!(select_in_byte(0b1010_1010, 2), 5);
        assert_eq!(select_in_byte(0b1010_1010, 3), 7);
        assert_eq!(select_in_byte(0b0000_0001, 0), 0);
        assert_eq!(select_in_byte(0b1000_0000, 0), 7);
        assert_eq!(select_in_byte(0xFF, 7), 7);
    }

    // ── batch select tests ──────────────────────────────────────────

    #[test]
    fn test_batch_select_all_set() {
        let buf = [0xFFu8; 16]; // 128 bits, all set
        let ranks: Vec<usize> = (0..128).collect();
        let mut out = vec![0usize; 128];
        bit_select_sorted_batch(&buf, 0, 128, &ranks, &mut out);
        for (nth, &pos) in out.iter().enumerate() {
            assert_eq!(pos, nth, "nth={nth}");
        }
    }

    #[test]
    fn test_batch_select_every_other() {
        let buf = [0x55u8; 16]; // 128 bits, 64 set
        let ranks: Vec<usize> = (0..64).collect();
        let mut out = vec![0usize; 64];
        bit_select_sorted_batch(&buf, 0, 128, &ranks, &mut out);
        for (nth, &pos) in out.iter().enumerate() {
            assert_eq!(pos, nth * 2, "nth={nth}");
        }
    }

    #[test]
    fn test_batch_select_sparse_ranks() {
        let buf = [0xFFu8; 16]; // 128 set
        let ranks = [0, 10, 50, 100, 127];
        let mut out = [0usize; 5];
        bit_select_sorted_batch(&buf, 0, 128, &ranks, &mut out);
        assert_eq!(out, [0, 10, 50, 100, 127]);
    }

    #[test]
    fn test_batch_select_empty() {
        let buf = [0xFFu8; 4];
        let mut out = [];
        bit_select_sorted_batch(&buf, 0, 32, &[], &mut out);
    }

    #[rstest]
    #[case(0, 128)]
    #[case(3, 100)]
    #[case(7, 50)]
    #[case(0, 65)]
    #[case(3, 256)]
    fn test_batch_select_agrees_with_individual(#[case] offset: usize, #[case] len: usize) {
        let total_bytes = (offset + len).div_ceil(8);
        let buf: Vec<u8> = (0..total_bytes)
            .map(|i| ((i.wrapping_mul(0x9E) ^ 0xA5) & 0xFF) as u8)
            .collect();

        // Get individual results.
        let true_count = (0..len)
            .filter(|&i| {
                let phys = offset + i;
                (buf[phys / 8] >> (phys % 8)) & 1 == 1
            })
            .count();

        let all_ranks: Vec<usize> = (0..true_count).collect();
        let individual: Vec<usize> = all_ranks
            .iter()
            .map(|&r| bit_select(&buf, offset, len, r))
            .collect();

        let mut batch = vec![0usize; true_count];
        bit_select_sorted_batch(&buf, offset, len, &all_ranks, &mut batch);

        assert_eq!(batch, individual, "offset={offset} len={len}");
    }
}
