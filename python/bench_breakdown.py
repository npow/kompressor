import time
import torch
from compressai.zoo import mbt2018_mean
from bench import pad

DEVICE = "cuda"


def timed(fn, n=10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0, out


def breakdown(model, x, label, n=10):
    model.eval()
    print(f"\n--- {label} ({x.shape[3]}x{x.shape[2]}) ---")
    with torch.no_grad():
        # warmup
        for _ in range(3):
            out = model.compress(x)
            _ = model.decompress(out["strings"], out["shape"])

        t_ga, y = timed(lambda: model.g_a(x), n)
        t_ha, z = timed(lambda: model.h_a(y), n)
        t_ebc, z_strings = timed(lambda: model.entropy_bottleneck.compress(z), n)
        t_ebd, z_hat = timed(lambda: model.entropy_bottleneck.decompress(z_strings, z.size()[-2:]), n)
        t_hs, gaussian_params = timed(lambda: model.h_s(z_hat), n)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        t_idx, indexes = timed(lambda: model.gaussian_conditional.build_indexes(scales_hat), n)
        t_gcc, y_strings = timed(lambda: model.gaussian_conditional.compress(y, indexes, means=means_hat), n)

        t_gcd, y_hat = timed(lambda: model.gaussian_conditional.decompress(y_strings, indexes, means=means_hat), n)
        t_gs, x_hat = timed(lambda: model.g_s(y_hat), n)

        total = t_ga + t_ha + t_ebc + t_ebd + t_hs + t_idx + t_gcc + t_gcd + t_gs
        rows = [
            ("g_a (GPU conv)", t_ga),
            ("h_a (GPU conv)", t_ha),
            ("entropy_bottleneck.compress [z] (CPU arith coding)", t_ebc),
            ("entropy_bottleneck.decompress [z] (CPU arith coding)", t_ebd),
            ("h_s (GPU conv)", t_hs),
            ("build_indexes (GPU)", t_idx),
            ("gaussian_conditional.compress [y] (CPU arith coding, THE BIG ONE)", t_gcc),
            ("gaussian_conditional.decompress [y] (CPU arith coding, THE BIG ONE)", t_gcd),
            ("g_s (GPU conv)", t_gs),
        ]
        for name, t in rows:
            print(f"  {name:65s} {t:8.3f} ms   ({t/total*100:5.1f}%)")
        print(f"  {'TOTAL':65s} {total:8.3f} ms")


if __name__ == "__main__":
    torch.set_num_threads(24)
    baseline = mbt2018_mean(quality=3, pretrained=True).to(DEVICE)
    baseline.update(force=True)

    for (h, w) in [(512, 512), (1080, 1920)]:
        x = pad(torch.rand(1, 3, h, w, device=DEVICE))
        breakdown(baseline, x, "baseline mbt2018_mean")
