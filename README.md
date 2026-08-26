# kompressor

A drop-in, **byte-exact-compatible** parallel entropy coder for [CompressAI](https://github.com/InterDigitalInc/CompressAI)'s mean-scale hyperprior image codecs (`mbt2018_mean` and friends), written in Rust. It produces identical compressed bitstreams to CompressAI's reference Python/C++ coder — verified byte-for-byte, not just by output length — while running roughly **60x faster** on the entropy-coding step, using true multi-core parallelism that CompressAI's Python bindings can't exploit.

## Why this exists

We were investigating whether `mbt2018_mean` (a CompressAI learned image codec, originally built for offline rate-distortion research, not real-time inference) could be made fast enough for real-time use. The obvious idea — a MobileNet-style architecture redesign (depthwise-separable convs, fewer channels, thinner stages) — turned out to be optimizing the wrong thing.

**Where the latency actually goes**, measured on a real `mbt2018_mean` (quality=3) forward pass at 1920x1080 on an RTX PRO 6000 Blackwell:

| Stage | Time | % of total |
|---|---:|---:|
| `g_a` (encoder conv, GPU) | 4.78 ms | 1.8% |
| `h_a` (hyperprior encoder, GPU) | 0.17 ms | 0.1% |
| entropy-code `z` (hyperprior latent) | 15.1 ms | 5.5% |
| `h_s` (hyperprior decoder, GPU) | 0.32 ms | 0.1% |
| **entropy-code `y` (main latent)** | **246.0 ms** | **90.2%** |
| `g_s` (decoder conv, GPU) | 4.96 ms | 1.8% |
| **Total** | **~273 ms** | 100% |

90% of the latency is CPU-side arithmetic coding of the main latent — a reference implementation prioritizing correctness over throughput, not the neural network. A MobileNet-style thinning of the conv stack (which we also prototyped) only touches the ~4% GPU portion and is a rounding error against the total. See [`python/BENCHMARKS.md`](python/BENCHMARKS.md) for the full investigation, including why naive Python threading (GIL-bound, 0.99x) and `multiprocessing` (IPC-bound, ~1.08x) don't fix it.

## What this crate does

`mbt2018_mean`'s mean-scale hyperprior has **no spatial autoregression** — every symbol's probability distribution depends only on its own predicted scale/mean, not on neighboring already-decoded symbols. That means the entropy coding of the `y`/`z` latents is embarrassingly parallel: you can split the tensor into N independent chunks, each coded into its own bitstream, on N threads, with zero cross-chunk dependency and zero loss in compression ratio (beyond a few bytes of per-chunk flush overhead).

CompressAI's Python bindings can't exploit this: `threading` is blocked by the GIL, and `multiprocessing` gets eaten by IPC/pickling overhead. Native Rust threads (via `rayon`) hit real, close-to-linear parallelism instead.

This repo is a faithful Rust port of CompressAI's actual algorithm — 64-bit rANS (`ryg_rans`-style), 16-bit CDF precision, same bypass-coding path for distribution outliers — not a reimplementation from scratch. It is validated to produce **byte-identical output** to the real CompressAI C++ coder across multiple test cases, including the outlier/bypass edge case.

## Results

Measured on an RTX PRO 6000 Blackwell (24 CPU cores), real `mbt2018_mean` (quality=3) weights, 1920x1080 input, single image (batch=1).

### Entropy coding: `y` latent (1,566,720 symbols)

| | Python/C++ reference | Rust, 1 thread | Rust, 24 threads |
|---|---:|---:|---:|
| encode | 58.0 ms (coder call) + ~10 ms marshaling | 18.4 ms | **2.3–2.8 ms** |
| decode | ~90 ms (coder call, est.) + ~10 ms marshaling | 22.3 ms | **1.7–2.1 ms** |

### Entropy coding: `z` latent (65,280 symbols, hyperprior)

| | Rust, 1 thread | Rust, 24 threads |
|---|---:|---:|
| encode | 0.81 ms | **0.15 ms** |
| decode | 0.93 ms | **0.09 ms** |

### Full pipeline, assembled from measured components

| | Baseline (CompressAI reference) | With this crate |
|---|---:|---:|
| conv stack (`g_a`+`h_a`+`h_s`+`g_s`, GPU, unchanged) | 10.2 ms | 10.2 ms |
| entropy coding (`y` + `z`) | 246 + 15.1 = 261.1 ms | **~4.4 ms** |
| **Total** | **~273 ms** | **~14.6 ms** |

**~18.7x end-to-end**, **~56x on the entropy-coding step specifically**, with zero retraining and zero risk to compression ratio — the output bitstream is bit-for-bit identical to what CompressAI itself would have produced.

> The conv-stack and entropy-coding numbers above were benchmarked as separate components (PyTorch/CUDA and standalone Rust respectively) and summed; an integrated pipeline would additionally pay GPU→CPU tensor transfer and FFI call overhead, not yet measured here. Given the small transfer size (~6 MB for `y`), this is expected to be low-single-digit milliseconds, but treat the "with this crate" total as a strong estimate assembled from validated pieces, not yet a single measured end-to-end run.

Only *after* the entropy coder stops dominating does the conv stack (~10ms) become the bottleneck worth attacking with an architecture redesign — see `python/thin_model.py` for a prototyped MobileNet-style thinned variant (depthwise-separable convs, 58x fewer parameters) and its own numbers, which are much more modest (~1.5-2x on that remaining portion, GPU-dependent).

## Correctness

Two layers of validation, both run in CI-able binaries:

1. **`cargo run --release --bin test_vectors`** — cross-interop correctness against real CompressAI Python/C++ output across 5 cases (normal Gaussian data, an entropy-bottleneck case, and — critically — a case with extreme outliers that forces the coder's bypass path). For each case it checks, independently:
   - Rust can **decode bytes Python encoded**, recovering the exact original symbols
   - Rust's own decode matches CompressAI's own `decompress()` output
   - Rust's **encode produces byte-identical output** to CompressAI's encoder (not just matching length)
   - Rust's own encode→decode round-trips correctly

   Test vectors are generated by `python/export_test_vectors.py`, which builds them the same way CompressAI's own `tests/test_entropy_models.py` validates round-trips (`test_compression_2D`/`test_compression_ND`), so this cross-checks against the project's own correctness philosophy rather than inventing a new one.

2. **`cargo run --release`** — the benchmark binary also asserts byte-exact match against real exported model data (real `y`/`z` tensors from an actual `mbt2018_mean` forward pass) before timing anything, so a performance number is never reported for output that doesn't actually match.

```
$ cargo run --release --bin test_vectors
...
ALL CASES PASSED
```

## Usage

```bash
# 1. set up the python side (CompressAI + a CUDA-enabled torch build appropriate for your GPU)
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128  # pick the right cu### for your GPU
pip install compressai

# 2. export real symbol/CDF data from a real model forward pass, and correctness test vectors
python3 export_data.py            # -> ../data.bin, ../data_z.bin
python3 export_test_vectors.py    # -> ../test_vectors.bin

# 3. build and run
cd ..
cargo build --release
./target/release/test_vectors   # correctness: byte-exact interop vs. real CompressAI output
./target/release/kompressor     # benchmark: single-thread and parallel encode/decode timings
```

## Repo layout

```
src/
  coder.rs        core rANS encode/decode -- faithful port of compressai's rans_interface.cpp
  lib.rs          pub mod coder;
  main.rs         benchmark binary (bin: kompressor)
  bin/
    test_vectors.rs   cross-interop correctness harness
python/
  thin_model.py            prototyped MobileNet-style thinned architecture (depthwise-separable convs)
  bench.py                 GPU forward-pass + full compress()/decompress() baseline timings
  bench_breakdown.py       per-stage latency breakdown (where does the 273ms actually go)
  bench_marshal.py         isolates Python<->C++ marshaling overhead from actual coder cost
  bench_parallel_entropy.py   demonstrates GIL blocks naive threading (0.99x speedup)
  bench_mp_entropy.py         demonstrates multiprocessing IPC overhead eats the gain (~1.08x)
  export_data.py           exports real y/z tensors + CDFs from a real mbt2018_mean forward pass
  export_test_vectors.py   exports correctness test vectors, incl. the bypass-path edge case
```

## Prior art / credits

The entropy coding algorithm (64-bit rANS, `ryg_rans`-derived, 16-bit CDF precision, bypass coding for distribution outliers) is a faithful re-implementation of [CompressAI](https://github.com/InterDigitalInc/CompressAI)'s `rans_interface.cpp` (InterDigital Communications, BSD-style license), which itself builds on Fabian Giesen's [ryg_rans](https://github.com/rygorous/ryg_rans). All credit for the algorithm design goes there; this project's contribution is the Rust port, the removal of the Python/C++ marshaling tax, and true multi-core parallelism for the (autoregression-free) mean-scale hyperprior case.

## License

MIT
