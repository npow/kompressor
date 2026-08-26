<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <b>kompressor</b>
</p>

<p align="center">
  <i>Byte-exact Rust replacement for CompressAI's entropy coder. ~56x faster.</i>
</p>

## Highlights

- A byte-exact drop-in for CompressAI's rANS entropy coder — same algorithm, identical compressed output, not an approximation.
- ~56x faster entropy coding, ~18.7x faster end-to-end on `mbt2018_mean` at 1080p.
- Real multi-core parallelism via native threads. CompressAI's Python coder can't do this: `threading` is GIL-blocked, `multiprocessing` is IPC-bound.
- No retraining, no architecture changes, no compression-ratio risk.
- Ships a correctness harness that cross-validates against real CompressAI output, including the coder's less-common bypass-path edge case.

## Why

CompressAI's reference Python/C++ entropy coder is a correctness-first research implementation. At 1080p it's ~90% of total `mbt2018_mean` latency — the neural net is ~10%.

| Component | % of total latency |
|---|---:|
| entropy coding (`y` + `z`) | ~90% |
| conv stack (`g_a`/`h_a`/`h_s`/`g_s`) | ~10% |

Mean-scale hyperpriors have no spatial autoregression: each symbol's distribution depends only on its own predicted scale/mean, not on neighbors. That makes the coder embarrassingly parallel across cores, with zero cross-chunk dependency and no compression-ratio loss. kompressor is a faithful port of CompressAI's actual coder (64-bit rANS, `ryg_rans`-style) that exploits this with `rayon`, plus two overhead fixes that help even single-threaded: no Python-list marshaling, and a binary search instead of a linear scan on CDF lookup.

Full investigation, including why a MobileNet-style architecture thinning barely helps (it only touches the 10%), is in [`python/BENCHMARKS.md`](python/BENCHMARKS.md).

## Getting Started

Export real symbol/CDF data from a `mbt2018_mean` forward pass, then build and run:

```console
$ cd python && python3 -m venv .venv && source .venv/bin/activate
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # match your GPU
$ pip install compressai
$ python3 export_data.py            # -> ../data.bin, ../data_z.bin
$ python3 export_test_vectors.py    # -> ../test_vectors.bin

$ cd .. && cargo build --release
$ ./target/release/kompressor
### y (GaussianConditional) — 1566720 symbols, 64 cdf rows, 345192 ref bytes
  correctness: byte-exact match vs python AND rust decodes python's own bytes correctly
  single-thread encode: 18.601 ms  (11.87 ns/symbol)
  single-thread decode: 22.349 ms  (14.26 ns/symbol)
  chunks=24  parallel encode:   2.496 ms (speedup 7.45x)   parallel decode:   2.114 ms (speedup 10.57x)
```

## Benchmarks

RTX PRO 6000 Blackwell, 24 cores, real `mbt2018_mean` (quality=3) weights, 1920x1080.

Full pipeline (conv stack unchanged, entropy coder swapped):

```
reference   ████████████████████████████████████████  273ms
kompressor  ██                                          15ms
```

Entropy coding of the `y` latent alone (encode + decode, 1.57M symbols):

```
reference   ████████████████████████████████████████  168ms
kompressor  █                                            4ms
```

| | reference (Python/C++) | kompressor, 1 thread | kompressor, 24 threads |
|---|---:|---:|---:|
| encode `y` (1.57M symbols) | ~68 ms | 18.4 ms | 2.3-2.8 ms |
| decode `y` | ~100 ms | 22.3 ms | 1.7-2.1 ms |
| encode `z` (65K symbols) | n/a | 0.81 ms | 0.15 ms |
| decode `z` | n/a | 0.93 ms | 0.09 ms |

Conv and entropy stages were benchmarked separately (PyTorch/CUDA and standalone Rust) and summed, not yet measured as one integrated run. Full methodology in [`python/BENCHMARKS.md`](python/BENCHMARKS.md).

## Correctness

```console
$ cargo run --release --bin test_vectors
```

Checks byte-exact interop against real CompressAI output across 5 cases, including an outlier case that forces the coder's bypass path. For each case: Rust decodes Python-encoded bytes correctly, matches Python's own decode, Rust's encode is byte-identical to Python's (not just same length), and self round-trips. The benchmark binary (`cargo run --release`) also asserts byte-exact match before timing anything.

## Layout

```
src/coder.rs                    core rANS encode/decode
src/bin/test_vectors.rs         correctness harness
python/thin_model.py            MobileNet-style thinned model, for comparison
python/bench*.py                latency breakdown, marshaling cost, GIL/IPC dead ends
python/export_*.py              generates data.bin / test_vectors.bin from real model runs
python/BENCHMARKS.md            full investigation writeup
```

## Contributing

The highest-impact contributions right now:

1. **PyO3 bindings** — call kompressor directly from Python instead of exporting to intermediate files
2. **Integrated end-to-end benchmark** — one measured run instead of summed GPU + Rust components
3. **Masked/sparse entropy coding** — skip discarded regions entirely instead of coding the full tensor
4. **Broader CompressAI coverage** — the coder is generic to `EntropyBottleneck`/`GaussianConditional`; validate against other zoo models beyond `mbt2018_mean`

## Credits

Algorithm is a faithful port of [CompressAI](https://github.com/InterDigitalInc/CompressAI)'s `rans_interface.cpp` (InterDigital, BSD-style), built on Fabian Giesen's [ryg_rans](https://github.com/rygorous/ryg_rans). This project's contribution: the Rust port, removing the marshaling overhead, and multi-core parallelism for the autoregression-free case.

## License

MIT
