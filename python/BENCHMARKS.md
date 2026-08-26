# Benchmark notes: why the entropy coder, not the architecture

This documents the investigation that led to this repo. All numbers measured on an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (24 CPU cores), real `mbt2018_mean` (quality=3) pretrained weights from CompressAI's model zoo, 1920x1080 input padded to 1920x1088 (batch=1).

## 1. The starting question

`mbt2018_mean` was designed by CompressAI for offline rate-distortion research, without a latency budget as a design constraint. The obvious fix looks architectural: a MobileNet-style redesign — depthwise-separable convolutions instead of full convs, fewer channels, fewer/thinner stages — the way MobileNet was built around a mobile-inference latency budget instead of raw ImageNet accuracy.

We prototyped exactly that (`thin_model.py`: depthwise-separable convs, N=192->64, M=320->96, 58x fewer parameters, GDN replaced with LeakyReLU) and benchmarked it against the real pretrained baseline before drawing any conclusions.

## 2. The architecture redesign barely moves the needle

`bench.py` compares baseline vs. thinned forward-pass-only latency (`g_a`+`g_s`, no entropy coding):

| resolution | baseline | thinned | speedup |
|---|---:|---:|---:|
| 512x512 | 1.55 ms | 0.98 ms | 1.58x |
| 1920x1080 | 13.85 ms | 7.49 ms | 1.85x |

Real, but note the decoder side: at 512x512 the thinned `g_s` (1.48ms) was **slower** than baseline's standard-conv `g_s` (0.68ms) — depthwise-separable transposed convs don't automatically win on GPU at small batch size; kernel-launch overhead and low arithmetic intensity can eat the FLOP savings. This is a real risk of assuming a mobile-CPU/NPU-oriented technique transfers to GPU serving.

More importantly: on the **full compress()/decompress() round trip including real entropy coding**, thinning only moved the total from 273ms to ~264ms. That's because the conv stack is a small fraction of the actual cost.

## 3. Where the latency actually goes

`bench_breakdown.py` instruments every stage of `compress()`/`decompress()` individually:

```
--- baseline mbt2018_mean (1920x1088) ---
  g_a (GPU conv)                                                       4.776 ms   (  1.8%)
  h_a (GPU conv)                                                       0.170 ms   (  0.1%)
  entropy_bottleneck.compress [z] (CPU arith coding)                   6.165 ms   (  2.3%)
  entropy_bottleneck.decompress [z] (CPU arith coding)                 8.958 ms   (  3.3%)
  h_s (GPU conv)                                                       0.315 ms   (  0.1%)
  build_indexes (GPU)                                                  1.391 ms   (  0.5%)
  gaussian_conditional.compress [y] (CPU arith coding, THE BIG ONE)  106.121 ms   ( 38.9%)
  gaussian_conditional.decompress [y] (CPU arith coding, THE BIG ONE)  139.857 ms   ( 51.3%)
  g_s (GPU conv)                                                       4.961 ms   (  1.8%)
  TOTAL                                                              272.715 ms
```

**~90% of total latency is CPU-side entropy coding of the `y` latent.** The conv stack (the thing architecture redesigns touch) is ~5%.

## 4. Why the reference coder is slow: not just algorithm, partly marshaling

`bench_marshal.py` isolates CompressAI's Python/C++ boundary cost. `GaussianConditional.compress()` does `symbols[i].reshape(-1).int().tolist()` — converting a 1.57M-element tensor to a Python list, element by element, before the C++ coder even runs:

```
symbols.tolist(): 10.11 ms
indexes.tolist(): 10.00 ms
cdf.tolist() (200512 elements): 1.56 ms

encode_with_indexes (C++ coder only, pre-marshaled): 58.00 ms

full compress() call (marshal + encode): 86.88 ms
  of which symbols+indexes .tolist(): 20.10 ms (23.1%)
  of which cdf/len/offset .tolist() (redone EVERY call): 1.56 ms (1.8%)
  of which actual C++ coding: 58.00 ms (66.8%)
```

~25% of the cost is marshaling overhead that's unrelated to the entropy-coding algorithm itself. The other ~67% (58ms encode / ~87ms decode observed in isolation) is the actual per-symbol C++ range coding — real algorithmic cost, running single-threaded.

The decoder's C++ implementation (`rans_interface.cpp`) also does a **linear scan** (`std::find_if`) through the CDF table for every decoded symbol, an O(table size) operation per symbol where a binary search would be O(log table size) — this is why decode (~140ms) is slower than encode (~90ms) despite being conceptually symmetric.

## 5. The naive parallelization fixes don't work

Since `mbt2018_mean`'s mean-scale hyperprior has no spatial autoregression, each symbol's distribution depends only on its own scale/mean — the `y` tensor should be trivially shardable across cores. Two obvious approaches, both tested and both dead ends in Python:

**`bench_parallel_entropy.py`** — `ThreadPoolExecutor` across chunks of `y`:
```
single-shot gaussian_conditional.compress: 90.31 ms
  sharded x 2 (ThreadPoolExecutor): 91.61 ms   speedup vs single-shot: 0.99x
  sharded x 4 (ThreadPoolExecutor): 98.41 ms   speedup vs single-shot: 0.92x
```
No speedup at all — CompressAI's pybind11 binding doesn't release the GIL during the C++ call, so threads just serialize.

**`bench_mp_entropy.py`** — `multiprocessing.Pool` across the same chunks:
```
single-shot (CPU model, 1 thread): 91.21 ms
  sharded x4 (multiprocessing.Pool, includes pool startup): 451.36 ms   speedup: 0.20x
  sharded x4 (warm pool, steady-state): 84.83 ms   speedup: 1.08x
```
Real parallelism, but IPC/pickling overhead for the tensor arguments swallows nearly all of the gain — even on a warm, pre-started pool.

## 6. What actually works: a native Rust port

See the top-level `README.md` and `src/` for the Rust implementation and its results. Summary: a faithful port of the same rANS algorithm, with the marshaling tax removed (operates directly on exported buffers, no `.tolist()`) and a binary-search CDF lookup on decode, gets single-threaded numbers already ~3x faster than the raw (post-marshal) C++ call. Layered on top, `rayon`-based sharding (same chunking strategy that failed in Python) gets real, close-to-linear multi-core scaling since it uses actual OS threads with shared memory instead of the GIL or IPC:

- `y` entropy coding: 230ms (Python, encode+decode combined, post-marshal) -> ~4ms (Rust, 24 threads)
- Full round trip: ~273ms -> ~14.6ms (conv stack unchanged, entropy coding replaced)

All of this is validated byte-exact against real CompressAI output — see the top-level README's Correctness section and `export_test_vectors.py`.

## Takeaway

For this codec, an architecture redesign is the wrong first lever. The entropy coder — a research-grade reference implementation never optimized for throughput — is ~90% of the cost, and fixing it requires no retraining and carries no risk to the compression ratio (bit-exact output). Only after that fix does the conv stack become the dominant remaining cost, which is the point where a MobileNet-style redesign starts to be worth the retraining risk.
