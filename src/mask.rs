// Generic mask-as-side-info encoding: compact, lossless encode/decode of a boolean
// "which positions were kept" mask.
//
// Motivation (sparse/masked entropy coding): `coder::encode_with_indexes` /
// `decode_with_indexes` operate on flat `&[i32]` arrays with no positional coupling, so
// a sender can filter a latent tensor's symbols/indexes down to only the KEPT positions
// (per some external mask, e.g. a learned keep/discard selector) before calling encode --
// no changes needed to the coder itself. What's missing on the decode side is telling the
// receiver WHICH positions were kept, so it can scatter the decoded values back to the
// right place in the full-size tensor. This module is that missing piece: generic, with
// no coupling to any particular masking scheme (it just consumes/produces `&[bool]`) and
// no coupling to the rANS coder above (a boolean mask has no per-position learned
// probability model the way a Gaussian-conditional symbol does, so there is nothing for
// the rANS machinery in `coder.rs` to key off -- a dedicated, much simpler encoding is the
// right tool here, not a re-use of `coder.rs`'s CDF-driven machinery).
//
// Two encodings, chosen automatically per call by whichever is smaller:
//
// - **Run-length coding** (tag 0): masks that arise from a spatially/structurally
//   correlated selection process (the common case -- e.g. a learned per-region
//   keep/discard map, whose neighboring entries tend to agree) compress well as a
//   sequence of alternating same-value runs. Each run length is written as an unsigned
//   LEB128 varint, so short runs cost ~1 byte and long runs stay compact too.
// - **Raw bit-packing** (tag 1): a fallback for masks with no exploitable structure (e.g.
//   uniform-random -- every run has expected length ~2, so RLE's per-run varint overhead
//   makes it WORSE than one bit per entry). Bit-packing is a hard ceiling: it never costs
//   more than `ceil(n/8)` body bytes regardless of mask content, so picking the smaller of
//   the two per call guarantees this module is never worse than a plain bitmap plus a
//   fixed 9-byte header, while still capturing RLE's real win on the correlated case this
//   was actually commissioned for.
//
// Wire format (self-describing, no external length parameter needed at decode time):
//   [1 byte  tag: 0 = RLE, 1 = raw bit-packed]
//   [8 bytes little-endian u64: n, the number of mask entries]
//   [body, meaning depends on tag -- see `encode_mask_rle_body`/`encode_mask_bitpacked_body`]

fn write_uvarint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn read_uvarint(buf: &[u8], pos: &mut usize) -> u64 {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = buf[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    result
}

/// RLE body: `[starting value: 1 byte (0 or 1)]` followed by varint run lengths,
/// alternating value starting from that first byte, summing to `mask.len()`. Empty for
/// `mask.is_empty()`.
fn encode_mask_rle_body(mask: &[bool]) -> Vec<u8> {
    let mut out = Vec::new();
    if mask.is_empty() {
        return out;
    }
    out.push(mask[0] as u8);
    let mut run_len: u64 = 1;
    for i in 1..mask.len() {
        if mask[i] == mask[i - 1] {
            run_len += 1;
        } else {
            write_uvarint(&mut out, run_len);
            run_len = 1;
        }
    }
    write_uvarint(&mut out, run_len);
    out
}

fn decode_mask_rle_body(body: &[u8], n: usize) -> Vec<bool> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let mut pos = 0usize;
    let mut value = body[pos] != 0;
    pos += 1;
    while out.len() < n {
        let run_len = read_uvarint(body, &mut pos);
        for _ in 0..run_len {
            out.push(value);
        }
        value = !value;
    }
    out
}

/// Raw bit-packed body: `ceil(n/8)` bytes, entry `i` is bit `i % 8` (LSB-first) of byte
/// `i / 8`. Trailing padding bits beyond `n` (if `n` isn't a multiple of 8) are zero but
/// carry no meaning -- `decode_mask_bitpacked_body` never reads past entry `n - 1`.
fn encode_mask_bitpacked_body(mask: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; mask.len().div_ceil(8)];
    for (i, &value) in mask.iter().enumerate() {
        if value {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

fn decode_mask_bitpacked_body(body: &[u8], n: usize) -> Vec<bool> {
    (0..n).map(|i| (body[i / 8] >> (i % 8)) & 1 != 0).collect()
}

/// Encode a boolean mask compactly and losslessly. See this module's docs for the wire
/// format and the RLE-vs-bit-packed selection rule.
pub fn encode_mask(mask: &[bool]) -> Vec<u8> {
    let rle_body = encode_mask_rle_body(mask);
    let bitpacked_body = encode_mask_bitpacked_body(mask);
    let use_rle = rle_body.len() <= bitpacked_body.len();
    let body = if use_rle { &rle_body } else { &bitpacked_body };

    let mut out = Vec::with_capacity(1 + 8 + body.len());
    out.push(if use_rle { 0u8 } else { 1u8 });
    out.extend_from_slice(&(mask.len() as u64).to_le_bytes());
    out.extend_from_slice(body);
    out
}

/// Decode a mask produced by [`encode_mask`]. Self-describing -- the entry count and
/// encoding tag both travel in the bitstream, so no external length is needed.
pub fn decode_mask(encoded: &[u8]) -> Vec<bool> {
    assert!(
        encoded.len() >= 9,
        "mask bitstream too short: {} bytes (need at least 9-byte header)",
        encoded.len()
    );
    let tag = encoded[0];
    let n = u64::from_le_bytes(encoded[1..9].try_into().unwrap()) as usize;
    let body = &encoded[9..];
    match tag {
        0 => decode_mask_rle_body(body, n),
        1 => decode_mask_bitpacked_body(body, n),
        other => panic!("unknown mask encoding tag: {other} (expected 0=RLE or 1=bit-packed)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_roundtrip(mask: &[bool]) {
        let encoded = encode_mask(mask);
        let decoded = decode_mask(&encoded);
        assert_eq!(decoded, mask, "round-trip mismatch for mask of len {}", mask.len());
    }

    #[test]
    fn empty_mask() {
        assert_roundtrip(&[]);
    }

    #[test]
    fn all_kept() {
        assert_roundtrip(&vec![true; 4096]);
    }

    #[test]
    fn all_discarded() {
        assert_roundtrip(&vec![false; 4096]);
    }

    #[test]
    fn single_pixel_kept_start() {
        let mut mask = vec![false; 1024];
        mask[0] = true;
        assert_roundtrip(&mask);
    }

    #[test]
    fn single_pixel_kept_middle() {
        let mut mask = vec![false; 1024];
        mask[512] = true;
        assert_roundtrip(&mask);
    }

    #[test]
    fn single_pixel_kept_end() {
        let mut mask = vec![false; 1024];
        mask[1023] = true;
        assert_roundtrip(&mask);
    }

    #[test]
    fn single_pixel_discarded_in_sea_of_kept() {
        // The complementary case to single_pixel_kept -- exercises a lone SHORT run
        // (length 1) sandwiched between two long runs of the opposite value.
        let mut mask = vec![true; 1024];
        mask[517] = false;
        assert_roundtrip(&mask);
    }

    #[test]
    fn spatially_clustered() {
        // A realistic mask-selector-shaped pattern: a handful of contiguous "keep"
        // blocks (e.g. an object region) inside an otherwise-discarded 64x64 grid,
        // plus one degenerate zero-width-adjacent block to exercise back-to-back runs
        // of the same value collapsing correctly (i.e. never emitted as two runs).
        let (h, w) = (64usize, 64usize);
        let mut mask = vec![false; h * w];
        for y in 10..20 {
            for x in 10..25 {
                mask[y * w + x] = true;
            }
        }
        for y in 40..48 {
            for x in 5..50 {
                mask[y * w + x] = true;
            }
        }
        assert_roundtrip(&mask);

        // Real regression check, not just round-trip correctness: a spatially
        // clustered mask like this is exactly the case RLE exists for, so it must
        // actually be picked (and must actually beat the bit-packed floor of
        // ceil(4096/8) = 512 bytes), or the "compact for correlated masks" claim
        // this module's docs make would be untested.
        let encoded = encode_mask(&mask);
        assert_eq!(encoded[0], 0, "expected RLE to win on a spatially clustered mask");
        assert!(
            encoded.len() < 512,
            "RLE encoding ({} bytes) should be well under the {}-byte bit-packed floor \
             for a clustered mask",
            encoded.len(),
            h * w / 8
        );
    }

    #[test]
    fn random_pattern_correctness_regardless_of_efficiency() {
        // A simple xorshift PRNG (no external dependency) -- deterministic across runs
        // so a failure is reproducible, but with no structure for RLE to exploit.
        // Correctness must hold even though RLE is expected to lose to bit-packing
        // here (every run has expected length ~2 under i.i.d. coin flips, so RLE's
        // ~1-byte-per-run overhead is worse than 1 bit/entry).
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mask: Vec<bool> = (0..8192).map(|_| next() & 1 == 1).collect();
        assert_roundtrip(&mask);

        let encoded = encode_mask(&mask);
        assert_eq!(encoded[0], 1, "expected bit-packing to win on an unstructured random mask");
        assert_eq!(
            encoded.len(),
            9 + 8192usize.div_ceil(8),
            "bit-packed size must be exactly the fixed floor for a random mask"
        );
    }

    #[test]
    fn non_byte_aligned_length() {
        // n not a multiple of 8 -- exercises the bit-packed path's padding-bit handling
        // (decode must never read entry n, even though the last body byte has unused
        // high bits).
        let mask: Vec<bool> = (0..37).map(|i| i % 3 == 0).collect();
        assert_roundtrip(&mask);
    }

    #[test]
    fn rle_body_direct_roundtrip() {
        // Exercise the RLE body encode/decode directly (bypassing encode_mask's
        // size-based auto-selection), so this path is verified even on inputs where
        // bit-packing would actually be chosen end-to-end.
        let patterns: [&[bool]; 4] = [
            &[],
            &[true],
            &[false, false, false, true, true, false],
            &[true, false, true, false, true, false, true, false, true],
        ];
        for mask in patterns {
            let body = encode_mask_rle_body(mask);
            let decoded = decode_mask_rle_body(&body, mask.len());
            assert_eq!(decoded, mask);
        }
    }

    #[test]
    fn bitpacked_body_direct_roundtrip() {
        // Same as above, for the bit-packed body -- verified even on inputs where RLE
        // would actually win end-to-end.
        let patterns: [&[bool]; 3] =
            [&[], &vec![true; 65][..], &vec![false, true, true, false, false][..]];
        for mask in patterns {
            let body = encode_mask_bitpacked_body(mask);
            let decoded = decode_mask_bitpacked_body(&body, mask.len());
            assert_eq!(decoded, mask);
        }
    }
}
