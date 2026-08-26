// Cross-interop correctness harness: checks the Rust rANS port against real
// bytes produced by CompressAI's Python/C++ coder (not just self-consistency).
use std::fs::File;
use std::io::Read;

use kompressor::coder::{decode_with_indexes, encode_with_indexes};

struct Case {
    name: String,
    symbols: Vec<i32>,
    indexes: Vec<i32>,
    cdfs: Vec<Vec<i32>>,
    cdf_lengths: Vec<i32>,
    offsets: Vec<i32>,
    py_bytes: Vec<u8>,
    py_decoded_symbols: Vec<i32>,
}

fn read_i32_vec(buf: &[u8], pos: &mut usize) -> Vec<i32> {
    let n = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap()) as usize;
    *pos += 4;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let off = *pos + i * 4;
        v.push(i32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
    }
    *pos += n * 4;
    v
}

fn read_string(buf: &[u8], pos: &mut usize) -> String {
    let n = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap()) as usize;
    *pos += 4;
    let s = String::from_utf8(buf[*pos..*pos + n].to_vec()).unwrap();
    *pos += n;
    s
}

fn read_bytes(buf: &[u8], pos: &mut usize) -> Vec<u8> {
    let n = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap()) as usize;
    *pos += 4;
    let b = buf[*pos..*pos + n].to_vec();
    *pos += n;
    b
}

fn load_cases(path: &str) -> Vec<Case> {
    let mut f = File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let mut pos = 0usize;

    let n_cases = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    let mut cases = Vec::with_capacity(n_cases);
    for _ in 0..n_cases {
        let name = read_string(&buf, &mut pos);
        let symbols = read_i32_vec(&buf, &mut pos);
        let indexes = read_i32_vec(&buf, &mut pos);
        let n_cdf = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let mut cdfs = Vec::with_capacity(n_cdf);
        for _ in 0..n_cdf {
            cdfs.push(read_i32_vec(&buf, &mut pos));
        }
        let cdf_lengths = read_i32_vec(&buf, &mut pos);
        let offsets = read_i32_vec(&buf, &mut pos);
        let py_bytes = read_bytes(&buf, &mut pos);
        let py_decoded_symbols = read_i32_vec(&buf, &mut pos);
        cases.push(Case { name, symbols, indexes, cdfs, cdf_lengths, offsets, py_bytes, py_decoded_symbols });
    }
    cases
}

fn main() {
    let cases = load_cases("test_vectors.bin");
    let mut all_ok = true;

    for c in &cases {
        println!("=== case: {} ({} symbols) ===", c.name, c.symbols.len());
        let mut ok = true;

        // 1. INTEROP DECODE: can Rust decode bytes that Python actually produced?
        let rust_decoded = decode_with_indexes(&c.py_bytes, &c.indexes, &c.cdfs, &c.cdf_lengths, &c.offsets);
        if rust_decoded == c.symbols {
            println!("  [PASS] rust decode(python bytes) == original symbols");
        } else {
            println!("  [FAIL] rust decode(python bytes) != original symbols");
            let mismatches: Vec<(usize, i32, i32)> = rust_decoded.iter().zip(c.symbols.iter()).enumerate()
                .filter(|(_, (a, b))| a != b).map(|(i, (a, b))| (i, *a, *b)).take(5).collect();
            println!("        first mismatches (idx, rust, expected): {:?}", mismatches);
            ok = false;
        }
        if rust_decoded == c.py_decoded_symbols {
            println!("  [PASS] rust decode(python bytes) == python's own decode() output");
        } else {
            println!("  [FAIL] rust decode(python bytes) != python's own decode() output");
            ok = false;
        }

        // 2. INTEROP ENCODE: does Rust produce byte-IDENTICAL output to Python, not just same length?
        let rust_encoded = encode_with_indexes(&c.symbols, &c.indexes, &c.cdfs, &c.cdf_lengths, &c.offsets);
        if rust_encoded == c.py_bytes {
            println!("  [PASS] rust encode() == python bytes, byte-for-byte ({} bytes)", rust_encoded.len());
        } else {
            println!(
                "  [FAIL] rust encode() != python bytes (rust len={}, python len={})",
                rust_encoded.len(),
                c.py_bytes.len()
            );
            ok = false;
        }

        // 3. Rust self round-trip (sanity)
        let self_decoded = decode_with_indexes(&rust_encoded, &c.indexes, &c.cdfs, &c.cdf_lengths, &c.offsets);
        if self_decoded == c.symbols {
            println!("  [PASS] rust self round-trip (encode then decode)");
        } else {
            println!("  [FAIL] rust self round-trip");
            ok = false;
        }

        all_ok &= ok;
    }

    println!("\n{}", if all_ok { "ALL CASES PASSED" } else { "SOME CASES FAILED" });
    if !all_ok {
        std::process::exit(1);
    }
}
