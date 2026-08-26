// Faithful port of compressai's rans_interface.cpp (ryg_rans, 64-bit, precision=16).
#![allow(dead_code)]

const RANS64_L: u64 = 1u64 << 31;
const PRECISION: u32 = 16;
const BYPASS_PRECISION: u32 = 4;
const MAX_BYPASS_VAL: u32 = (1 << BYPASS_PRECISION) - 1;

struct RansSymbol {
    start: u16,
    range: u16,
    bypass: bool,
}

pub fn encode_with_indexes(
    symbols: &[i32],
    indexes: &[i32],
    cdfs: &[Vec<i32>],
    cdf_lengths: &[i32],
    offsets: &[i32],
) -> Vec<u8> {
    let mut syms: Vec<RansSymbol> = Vec::with_capacity(symbols.len());

    for i in 0..symbols.len() {
        let cdf_idx = indexes[i] as usize;
        let cdf = &cdfs[cdf_idx];
        let max_value = cdf_lengths[cdf_idx] - 2;

        let mut value = symbols[i] - offsets[cdf_idx];
        let mut raw_val: u32 = 0;
        if value < 0 {
            raw_val = (-2 * value - 1) as u32;
            value = max_value;
        } else if value >= max_value {
            raw_val = (2 * (value - max_value)) as u32;
            value = max_value;
        }

        let v = value as usize;
        syms.push(RansSymbol {
            start: cdf[v] as u16,
            range: (cdf[v + 1] - cdf[v]) as u16,
            bypass: false,
        });

        if value == max_value {
            let mut n_bypass = 0u32;
            while (raw_val >> (n_bypass * BYPASS_PRECISION)) != 0 {
                n_bypass += 1;
            }
            let mut val = n_bypass;
            while val >= MAX_BYPASS_VAL {
                syms.push(RansSymbol {
                    start: MAX_BYPASS_VAL as u16,
                    range: (MAX_BYPASS_VAL + 1) as u16,
                    bypass: true,
                });
                val -= MAX_BYPASS_VAL;
            }
            syms.push(RansSymbol {
                start: val as u16,
                range: (val + 1) as u16,
                bypass: true,
            });

            for j in 0..n_bypass {
                let val = (raw_val >> (j * BYPASS_PRECISION)) & MAX_BYPASS_VAL;
                syms.push(RansSymbol {
                    start: val as u16,
                    range: (val + 1) as u16,
                    bypass: true,
                });
            }
        }
    }

    let mut output: Vec<u32> = vec![0xCCCCCCCC; syms.len() + 16];
    let mut ptr = output.len();

    let mut rans: u64 = RANS64_L;

    for sym in syms.iter().rev() {
        if !sym.bypass {
            let start = sym.start as u32;
            let freq = sym.range as u32;
            let x_max = ((RANS64_L >> PRECISION) << 32) * freq as u64;
            let mut x = rans;
            if x >= x_max {
                ptr -= 1;
                output[ptr] = x as u32;
                x >>= 32;
            }
            rans = ((x / freq as u64) << PRECISION) + (x % freq as u64) + start as u64;
        } else {
            let nbits = BYPASS_PRECISION;
            let val = sym.start as u32;
            let mut x = rans;
            let freq = 1u32 << (16 - nbits);
            let x_max = ((RANS64_L >> 16) << 32) * freq as u64;
            if x >= x_max {
                ptr -= 1;
                output[ptr] = x as u32;
                x >>= 32;
            }
            rans = (x << nbits) | val as u64;
        }
    }

    ptr -= 2;
    output[ptr] = rans as u32;
    output[ptr + 1] = (rans >> 32) as u32;

    output[ptr..].iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub fn decode_with_indexes(
    encoded: &[u8],
    indexes: &[i32],
    cdfs: &[Vec<i32>],
    cdf_lengths: &[i32],
    offsets: &[i32],
) -> Vec<i32> {
    let words: Vec<u32> = encoded
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut wptr = 0usize;

    let mut rans: u64 = (words[0] as u64) | ((words[1] as u64) << 32);
    wptr += 2;

    let mut output = vec![0i32; indexes.len()];

    for i in 0..indexes.len() {
        let cdf_idx = indexes[i] as usize;
        let cdf = &cdfs[cdf_idx];
        let cdf_len = cdf_lengths[cdf_idx] as usize;
        let max_value = cdf_lengths[cdf_idx] - 2;
        let offset = offsets[cdf_idx];

        let cum_freq = (rans & ((1u64 << PRECISION) - 1)) as i32;

        let s = cdf[0..cdf_len].partition_point(|&v| v <= cum_freq) - 1;

        let start = cdf[s] as u64;
        let freq = (cdf[s + 1] - cdf[s]) as u64;
        let mask = (1u64 << PRECISION) - 1;
        let mut x = rans;
        x = freq * (x >> PRECISION) + (x & mask) - start;
        if x < RANS64_L {
            x = (x << 32) | words[wptr] as u64;
            wptr += 1;
        }
        rans = x;

        let mut value = s as i32;

        if value == max_value {
            let get_bits = |rans: &mut u64, wptr: &mut usize, nbits: u32| -> u32 {
                let x = *rans;
                let val = (x & ((1u64 << nbits) - 1)) as u32;
                let mut x = x >> nbits;
                if x < RANS64_L {
                    x = (x << 32) | words[*wptr] as u64;
                    *wptr += 1;
                }
                *rans = x;
                val
            };

            let mut val = get_bits(&mut rans, &mut wptr, BYPASS_PRECISION);
            let mut n_bypass = val;
            while val == MAX_BYPASS_VAL {
                val = get_bits(&mut rans, &mut wptr, BYPASS_PRECISION);
                n_bypass += val;
            }
            let mut raw_val: u32 = 0;
            for j in 0..n_bypass {
                let val = get_bits(&mut rans, &mut wptr, BYPASS_PRECISION);
                raw_val |= val << (j * BYPASS_PRECISION);
            }
            value = (raw_val >> 1) as i32;
            if raw_val & 1 != 0 {
                value = -value - 1;
            } else {
                value += max_value;
            }
        }

        output[i] = value + offset;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a CompressAI-style quantized CDF row from integer frequencies that must
    /// sum to exactly 1<<16. Returns (cdf, cdf_length, max_value).
    fn build_cdf(freqs: &[i32]) -> (Vec<i32>, i32) {
        assert_eq!(
            freqs.iter().sum::<i32>(),
            1 << 16,
            "frequencies must sum to 65536"
        );
        let mut cdf = vec![0i32];
        let mut acc = 0i32;
        for &f in freqs {
            acc += f;
            cdf.push(acc);
        }
        let cdf_length = cdf.len() as i32;
        (cdf, cdf_length)
    }

    fn roundtrip(
        symbols: &[i32],
        indexes: &[i32],
        cdfs: &[Vec<i32>],
        cdf_lengths: &[i32],
        offsets: &[i32],
    ) -> Vec<i32> {
        let encoded = encode_with_indexes(symbols, indexes, cdfs, cdf_lengths, offsets);
        decode_with_indexes(&encoded, indexes, cdfs, cdf_lengths, offsets)
    }

    #[test]
    fn roundtrip_single_uniform_row() {
        // 8 equally likely symbols, no outliers.
        let (cdf, cdf_length) = build_cdf(&[8192; 8]);
        let cdfs = vec![cdf];
        let cdf_lengths = vec![cdf_length];
        let offsets = vec![0];

        let symbols: Vec<i32> = (0..8).cycle().take(200).collect();
        let indexes = vec![0; symbols.len()];

        let decoded = roundtrip(&symbols, &indexes, &cdfs, &cdf_lengths, &offsets);
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn roundtrip_with_offset() {
        // Same distribution, but symbols are centered around a nonzero mean via offset.
        let (cdf, cdf_length) = build_cdf(&[8192; 8]);
        let cdfs = vec![cdf];
        let cdf_lengths = vec![cdf_length];
        let offsets = vec![-50]; // symbol range becomes [-50, -43]

        let symbols: Vec<i32> = (-50..=-43).cycle().take(100).collect();
        let indexes = vec![0; symbols.len()];

        let decoded = roundtrip(&symbols, &indexes, &cdfs, &cdf_lengths, &offsets);
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn roundtrip_multiple_cdf_rows() {
        // Simulates multiple quantized scale bins, each with its own distribution,
        // the way GaussianConditional indexes per-symbol CDFs by scale bucket.
        let (cdf_a, len_a) = build_cdf(&[32768, 32768]); // 2 symbols, sharply peaked-ish
        let (cdf_b, len_b) = build_cdf(&[16384; 4]); // 4 symbols, uniform
        let (cdf_c, len_c) = build_cdf(&[8192; 8]); // 8 symbols, uniform
        let cdfs = vec![cdf_a, cdf_b, cdf_c];
        let cdf_lengths = vec![len_a, len_b, len_c];
        let offsets = vec![0, 0, 0];

        let symbols = vec![0, 1, 0, 1, 2, 3, 0, 5, 7, 1, 0, 3, 2];
        let indexes = vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 1, 2];

        let decoded = roundtrip(&symbols, &indexes, &cdfs, &cdf_lengths, &offsets);
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn roundtrip_bypass_path_outliers() {
        // Small max_value so it's easy to land symbols far outside [0, max_value],
        // forcing the coder's bypass path (mirrors real Gaussian-conditional outliers).
        let (cdf, cdf_length) = build_cdf(&[16384; 4]);
        let cdfs = vec![cdf];
        let cdf_lengths = vec![cdf_length];
        let offsets = vec![0];

        let symbols = vec![0, 1, 2, 3, 1000, -1000, 50_000, -50_000, 2, 0];
        let indexes = vec![0; symbols.len()];

        let decoded = roundtrip(&symbols, &indexes, &cdfs, &cdf_lengths, &offsets);
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn roundtrip_empty() {
        let decoded = roundtrip(&[], &[], &[], &[], &[]);
        assert_eq!(decoded, Vec::<i32>::new());
    }

    #[test]
    fn roundtrip_large_random() {
        // Deterministic pseudo-random stream (no external RNG dependency), covering many
        // symbols across a distribution wide enough to mix normal and bypass-path values.
        let (cdf, cdf_length) = build_cdf(&[4096; 16]);
        let cdfs = vec![cdf];
        let cdf_lengths = vec![cdf_length];
        let offsets = vec![0];

        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let n = 10_000;
        let symbols: Vec<i32> = (0..n)
            .map(|_| {
                let r = next() % 1000;
                match r {
                    0..=979 => (next() % 16) as i32,     // normal path
                    980..=989 => (next() % 5000) as i32, // positive outlier -> bypass
                    _ => -((next() % 5000) as i32),      // negative outlier -> bypass
                }
            })
            .collect();
        let indexes = vec![0; symbols.len()];

        let decoded = roundtrip(&symbols, &indexes, &cdfs, &cdf_lengths, &offsets);
        assert_eq!(decoded, symbols);
    }
}
