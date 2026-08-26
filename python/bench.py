import time
import math
import sys
import torch
import torch.nn.functional as F
from compressai.zoo import mbt2018_mean

from thin_model import ThinMeanScaleHyperprior

DEVICE = "cuda"
torch.backends.cudnn.benchmark = True


def pad(x, factor=64):
    h, w = x.size(2), x.size(3)
    new_h = math.ceil(h / factor) * factor
    new_w = math.ceil(w / factor) * factor
    pad_h = new_h - h
    pad_w = new_w - w
    return F.pad(x, (0, pad_w, 0, pad_h))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def bench_forward(model, x, n_warmup=10, n_iters=50):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.g_s(model.g_a(x))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            y = model.g_a(x)
            x_hat = model.g_s(y)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1000.0  # ms


def bench_ga(model, x, n_warmup=10, n_iters=50):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.g_a(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            y = model.g_a(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1000.0


def bench_gs(model, y, n_warmup=10, n_iters=50):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.g_s(y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            x_hat = model.g_s(y)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1000.0


def bench_full_codec(model, x, n_warmup=3, n_iters=10):
    """Real entropy coding round trip (CPU-bound arithmetic coding included)."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            out = model.compress(x)
            _ = model.decompress(out["strings"], out["shape"])
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = model.compress(x)
        torch.cuda.synchronize()
        t_compress = (time.perf_counter() - t0) / n_iters * 1000.0

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model.decompress(out["strings"], out["shape"])
        torch.cuda.synchronize()
        t_decompress = (time.perf_counter() - t0) / n_iters * 1000.0

    bpp = sum(len(s[0]) for s in out["strings"][:1]) # y strings, list-of-list
    total_bytes = sum(len(s) for slist in out["strings"] for s in slist)
    return t_compress, t_decompress, total_bytes


def run_for_resolution(h, w, quality):
    print(f"\n=== Resolution {w}x{h}, quality={quality} ===")
    x = torch.rand(1, 3, h, w, device=DEVICE)
    xp = pad(x)
    print(f"padded to {xp.shape[3]}x{xp.shape[2]}")

    baseline = mbt2018_mean(quality=quality, pretrained=True).to(DEVICE)
    baseline.update(force=True)
    n_params_base = count_params(baseline)

    thin = ThinMeanScaleHyperprior(N=64, M=96).to(DEVICE)
    thin.update(force=True)
    n_params_thin = count_params(thin)

    print(f"baseline params: {n_params_base/1e6:.2f}M  |  thin params: {n_params_thin/1e6:.2f}M "
          f"({n_params_base/n_params_thin:.1f}x fewer)")

    # forward-only (no entropy coding) timings
    base_fwd = bench_forward(baseline, xp)
    thin_fwd = bench_forward(thin, xp)
    print(f"[forward g_a+g_s, no entropy coding]  baseline: {base_fwd:.2f} ms   thin: {thin_fwd:.2f} ms   "
          f"speedup: {base_fwd/thin_fwd:.2f}x")

    base_ga = bench_ga(baseline, xp)
    thin_ga = bench_ga(thin, xp)
    print(f"[g_a only]  baseline: {base_ga:.2f} ms   thin: {thin_ga:.2f} ms")

    y_base = baseline.g_a(xp)
    y_thin = thin.g_a(xp)
    base_gs = bench_gs(baseline, y_base)
    thin_gs = bench_gs(thin, y_thin)
    print(f"[g_s only]  baseline: {base_gs:.2f} ms   thin: {thin_gs:.2f} ms")

    # full real codec round trip incl. actual arithmetic coding (CPU-bound part included)
    base_c, base_d, base_bytes = bench_full_codec(baseline, xp)
    thin_c, thin_d, thin_bytes = bench_full_codec(thin, xp)
    print(f"[full compress()] baseline: {base_c:.2f} ms   thin: {thin_c:.2f} ms   speedup: {base_c/thin_c:.2f}x")
    print(f"[full decompress()] baseline: {base_d:.2f} ms   thin: {thin_d:.2f} ms   speedup: {base_d/thin_d:.2f}x")
    print(f"[full round trip] baseline: {base_c+base_d:.2f} ms   thin(untrained,rate meaningless): {thin_c+thin_d:.2f} ms")
    print(f"bytes (informational, thin is untrained so its rate is not meaningful): "
          f"baseline={base_bytes}  thin={thin_bytes}")


if __name__ == "__main__":
    torch.set_num_threads(24)
    for (h, w) in [(512, 512), (1080, 1920)]:
        run_for_resolution(h, w, quality=3)
