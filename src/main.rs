use kompressor::coder::{decode_with_indexes, encode_with_indexes};

use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

struct Data {
    symbols: Vec<i32>,
    indexes: Vec<i32>,
    cdfs: Vec<Vec<i32>>,
    cdf_lengths: Vec<i32>,
    offsets: Vec<i32>,
    ref_bytes: Vec<u8>,
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

fn load_data(path: &str) -> Data {
    let mut f = File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let mut pos = 0usize;

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

    let n_bytes = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let ref_bytes = buf[pos..pos + n_bytes].to_vec();

    Data { symbols, indexes, cdfs, cdf_lengths, offsets, ref_bytes }
}

fn bench_one(label: &str, path: &str, chunk_counts: &[usize]) {
    let data = load_data(path);
    println!(
        "\n### {} — {} symbols, {} cdf rows, {} ref bytes",
        label,
        data.symbols.len(),
        data.cdfs.len(),
        data.ref_bytes.len()
    );

    // ---- correctness: byte-exact interop against the real python-produced bytes ----
    let encoded = encode_with_indexes(&data.symbols, &data.indexes, &data.cdfs, &data.cdf_lengths, &data.offsets);
    assert_eq!(
        encoded, data.ref_bytes,
        "{}: rust-encoded bytes differ from python reference bytes (byte content, not just length!)",
        label
    );
    let decoded = decode_with_indexes(&encoded, &data.indexes, &data.cdfs, &data.cdf_lengths, &data.offsets);
    assert_eq!(decoded, data.symbols, "{}: roundtrip mismatch!", label);
    let decoded_from_py_bytes = decode_with_indexes(&data.ref_bytes, &data.indexes, &data.cdfs, &data.cdf_lengths, &data.offsets);
    assert_eq!(decoded_from_py_bytes, data.symbols, "{}: rust decode(python bytes) mismatch!", label);
    println!("  correctness: byte-exact match vs python AND rust decodes python's own bytes correctly");

    let n_iters = 20;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = encode_with_indexes(&data.symbols, &data.indexes, &data.cdfs, &data.cdf_lengths, &data.offsets);
    }
    let t_encode = t0.elapsed().as_secs_f64() / n_iters as f64 * 1000.0;
    println!("  single-thread encode: {:.3} ms  ({:.2} ns/symbol)", t_encode, t_encode * 1e6 / data.symbols.len() as f64);

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = decode_with_indexes(&encoded, &data.indexes, &data.cdfs, &data.cdf_lengths, &data.offsets);
    }
    let t_decode = t0.elapsed().as_secs_f64() / n_iters as f64 * 1000.0;
    println!("  single-thread decode: {:.3} ms  ({:.2} ns/symbol)", t_decode, t_decode * 1e6 / data.symbols.len() as f64);

    for &n_chunks in chunk_counts {
        let n = data.symbols.len();
        if n < n_chunks {
            continue;
        }
        let chunk_size = (n + n_chunks - 1) / n_chunks;
        let sym_chunks: Vec<&[i32]> = data.symbols.chunks(chunk_size).collect();
        let idx_chunks: Vec<&[i32]> = data.indexes.chunks(chunk_size).collect();

        let t0 = Instant::now();
        let encoded_chunks: Vec<Vec<u8>> = (0..sym_chunks.len())
            .into_par_iter()
            .map(|i| encode_with_indexes(sym_chunks[i], idx_chunks[i], &data.cdfs, &data.cdf_lengths, &data.offsets))
            .collect();
        let t_par_encode = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let decoded_chunks: Vec<Vec<i32>> = (0..encoded_chunks.len())
            .into_par_iter()
            .map(|i| decode_with_indexes(&encoded_chunks[i], idx_chunks[i], &data.cdfs, &data.cdf_lengths, &data.offsets))
            .collect();
        let t_par_decode = t0.elapsed().as_secs_f64() * 1000.0;

        for (i, dc) in decoded_chunks.iter().enumerate() {
            assert_eq!(dc.as_slice(), sym_chunks[i], "{}: chunk {} mismatch at n_chunks={}", label, i, n_chunks);
        }

        println!(
            "  chunks={:2}  parallel encode: {:7.3} ms (speedup {:.2}x)   parallel decode: {:7.3} ms (speedup {:.2}x)",
            n_chunks,
            t_par_encode,
            t_encode / t_par_encode,
            t_par_decode,
            t_decode / t_par_decode
        );
    }
}

fn main() {
    bench_one("y (GaussianConditional)", "data.bin", &[2, 4, 8, 16, 24, 32]);
    bench_one("z (EntropyBottleneck)", "data_z.bin", &[2, 4, 8, 16, 24]);
}
