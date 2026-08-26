# kompressor

A drop-in entropy coder for CompressAI's mean-scale hyperprior image codecs (`mbt2018_mean` and similar) that makes them fast enough to run in real time.

If you're running `mbt2018_mean` in a latency-sensitive pipeline, the reference Python/C++ coder is almost certainly your bottleneck, not the neural network — at 1080p it's **~90% of total latency**. This is a byte-exact Rust replacement for that coder: same algorithm, identical compressed output, but **~56x faster** by removing Python/C++ marshaling overhead and exploiting the fact that mean-scale hyperpriors have no spatial autoregression, so encoding parallelizes cleanly across CPU cores.

- **~56x faster entropy coding** (246ms → ~4.4ms for a 1080p `y` latent, 24 cores)
- **~18.7x faster end to end** (~273ms → ~14.6ms total, conv stack unchanged)
- **Zero retraining, zero compression-ratio risk** — output is byte-identical to CompressAI's own coder, verified, not approximated
- **Zero architecture changes** — this is not a model redesign, it's a coder swap

## Why the neural network isn't the problem

It's tempting to assume a slow learned codec needs a lighter architecture — fewer channels, depthwise-separable convs, MobileNet-style thinning. We prototyped that (`python/thin_model.py`) and it barely moves the needle: the conv stack is only ~5% of total latency. The other ~90% is CPU-side arithmetic coding, a correctness-first reference implementation that was never built for throughput. Full breakdown, including why naive Python threading and multiprocessing don't fix it, in [`python/BENCHMARKS.md`](python/BENCHMARKS.md).

## How it works

`mbt2018_mean`'s mean-scale hyperprior has **no spatial autoregression** — each symbol's probability distribution depends only on its own predicted scale/mean, never on neighboring already-decoded symbols. That means the entropy coding step is embarrassingly parallel: split the latent into N independent chunks, code each into its own bitstream on its own thread, with zero cross-chunk dependency and no loss in compression ratio beyond a few bytes of per-chunk overhead.

CompressAI's Python bindings can't exploit this — `threading` is blocked by the GIL, `multiprocessing` is eaten by IPC overhead. This crate is a faithful Rust port of CompressAI's actual algorithm (64-bit rANS, `ryg_rans`-style, same bypass-coding path for outliers) using native threads (`rayon`) for real parallelism, plus a couple of overhead fixes (no Python-list marshaling, binary search instead of linear scan on CDF lookup) that help even single-threaded.

## Results

Measured on an RTX PRO 6000 Blackwell (24 cores), real `mbt2018_mean` (quality=3) weights, 1920x1080 input.

**Entropy coding, `y` latent (1,566,720 symbols):**

| | Python/C++ reference | Rust, 1 thread | Rust, 24 threads |
|---|---:|---:|---:|
| encode | ~68 ms | 18.4 ms | **2.3–2.8 ms** |
| decode | ~100 ms | 22.3 ms | **1.7–2.1 ms** |

**Entropy coding, `z` latent (65,280 symbols, hyperprior):**

| | Rust, 1 thread | Rust, 24 threads |
|---|---:|---:|
| encode | 0.81 ms | **0.15 ms** |
| decode | 0.93 ms | **0.09 ms** |

**Full pipeline** (conv stack measured on GPU, unchanged; entropy coding measured standalone in Rust):

| | Baseline | With kompressor |
|---|---:|---:|
| conv stack (`g_a`+`h_a`+`h_s`+`g_s`) | 10.2 ms | 10.2 ms |
| entropy coding (`y` + `z`) | 261.1 ms | **~4.4 ms** |
| **Total** | **~273 ms** | **~14.6 ms** |

> These two halves were benchmarked separately (PyTorch/CUDA and standalone Rust) and summed. An integrated pipeline would additionally pay GPU→CPU transfer and FFI call overhead — expected to be low-single-digit ms given the small transfer size, but not yet measured as one run. Full methodology in `python/BENCHMARKS.md`.

## Correctness

Two validation layers, both runnable directly:

```
$ cargo run --release --bin test_vectors
```

Checks byte-exact interop against real CompressAI output across 5 cases — including one with extreme outliers that forces the coder's less-common bypass path. For each case: Rust decodes Python-encoded bytes correctly, Rust's decode matches CompressAI's own `decompress()`, Rust's encode produces byte-identical output to CompressAI's encoder (not just matching length), and Rust's own round trip is self-consistent. Test vectors are built the same way CompressAI's own test suite (`test_compression_2D`/`test_compression_ND` in `tests/test_entropy_models.py`) validates round-trips.

```
$ cargo run --release
```

The benchmark binary also asserts byte-exact match against real exported model data before timing anything — a performance number is never reported for output that doesn't actually match.

## Usage

```bash
# 1. python side: CompressAI + a torch build matching your GPU's CUDA capability
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # pick the right cu### for your GPU
pip install compressai

# 2. export real symbol/CDF data from a real model forward pass, plus correctness test vectors
python3 export_data.py            # -> ../data.bin, ../data_z.bin
python3 export_test_vectors.py    # -> ../test_vectors.bin

# 3. build and run
cd ..
cargo build --release
./target/release/test_vectors   # correctness
./target/release/kompressor     # benchmark
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
  thin_model.py            prototyped MobileNet-style thinned architecture (for comparison -- see BENCHMARKS.md)
  bench.py                 GPU forward-pass + full compress()/decompress() baseline timings
  bench_breakdown.py       per-stage latency breakdown
  bench_marshal.py         isolates Python<->C++ marshaling overhead from actual coder cost
  bench_parallel_entropy.py   why naive Python threading doesn't work (GIL)
  bench_mp_entropy.py         why naive multiprocessing doesn't work (IPC overhead)
  export_data.py           exports real y/z tensors + CDFs from a real mbt2018_mean forward pass
  export_test_vectors.py   exports correctness test vectors, incl. the bypass-path edge case
  BENCHMARKS.md            full investigation writeup and methodology
```

## Prior art / credits

The entropy coding algorithm (64-bit rANS, `ryg_rans`-derived, 16-bit CDF precision, bypass coding for outliers) is a faithful re-implementation of [CompressAI](https://github.com/InterDigitalInc/CompressAI)'s `rans_interface.cpp` (InterDigital Communications, BSD-style license), itself built on Fabian Giesen's [ryg_rans](https://github.com/rygorous/ryg_rans). This project's contribution is the Rust port, the removal of the Python/C++ marshaling tax, and multi-core parallelism for the (autoregression-free) mean-scale hyperprior case.

## License

MIT
