# kompressor

Byte-exact Rust replacement for CompressAI's entropy coder (`mbt2018_mean` and other mean-scale hyperprior codecs). Same algorithm, identical compressed output, ~56x faster.

At 1080p, CompressAI's reference Python/C++ entropy coder is ~90% of total codec latency. The neural net is ~10%. kompressor replaces just the coder. No retraining, no compression-ratio change, no architecture work.

## Numbers

RTX PRO 6000 Blackwell, 24 cores, real `mbt2018_mean` (quality=3) weights, 1920x1080.

| | reference (Python/C++) | kompressor, 1 thread | kompressor, 24 threads |
|---|---:|---:|---:|
| encode `y` (1.57M symbols) | ~68 ms | 18.4 ms | 2.3-2.8 ms |
| decode `y` | ~100 ms | 22.3 ms | 1.7-2.1 ms |
| encode `z` (65K symbols) | n/a | 0.81 ms | 0.15 ms |
| decode `z` | n/a | 0.93 ms | 0.09 ms |

Full pipeline: conv stack unchanged at 10.2ms, entropy coding swapped, total ~273ms down to ~14.6ms. Conv and entropy were benchmarked separately and summed, not yet measured as one integrated run. Full methodology in [`python/BENCHMARKS.md`](python/BENCHMARKS.md). That file also covers a MobileNet-style architecture thinning we prototyped (`python/thin_model.py`); it barely helps, since it only touches the 10%.

## How it works

Mean-scale hyperpriors have no spatial autoregression. Each symbol's distribution depends only on its own predicted scale/mean, not on neighbors. So the coder parallelizes across cores with zero cross-chunk dependency and no compression-ratio loss.

CompressAI's Python bindings can't use this: `threading` is GIL-blocked, `multiprocessing` is IPC-bound. Native threads via `rayon` can. This is a faithful port of CompressAI's actual coder (64-bit rANS, `ryg_rans`-style), not a reimplementation. It also fixes two overhead sources that help even single-threaded: no Python-list marshaling, and a binary search instead of a linear scan on CDF lookup.

## Correctness

```
cargo run --release --bin test_vectors
```

Checks byte-exact interop against real CompressAI output across 5 cases, including an outlier case that forces the coder's bypass path. For each case: Rust decodes Python-encoded bytes correctly, matches Python's own decode, Rust's encode is byte-identical to Python's (not just same length), and self round-trips. `cargo run --release`, the benchmark binary, also asserts byte-exact match before timing anything.

## Usage

```bash
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # match your GPU
pip install compressai

python3 export_data.py            # -> ../data.bin, ../data_z.bin
python3 export_test_vectors.py    # -> ../test_vectors.bin

cd ..
cargo build --release
./target/release/test_vectors   # correctness
./target/release/kompressor     # benchmark
```

## Layout

```
src/coder.rs                    core rANS encode/decode
src/bin/test_vectors.rs         correctness harness
python/thin_model.py            MobileNet-style thinned model, for comparison
python/bench*.py                latency breakdown, marshaling cost, GIL/IPC dead ends
python/export_*.py              generates data.bin / test_vectors.bin from real model runs
python/BENCHMARKS.md            full investigation writeup
```

## Credits

Algorithm is a faithful port of [CompressAI](https://github.com/InterDigitalInc/CompressAI)'s `rans_interface.cpp` (InterDigital, BSD-style), built on Fabian Giesen's [ryg_rans](https://github.com/rygorous/ryg_rans). This project's contribution: the Rust port, removing the marshaling overhead, and multi-core parallelism for the autoregression-free case.

MIT
