import time
import torch
from concurrent.futures import ThreadPoolExecutor
from compressai.zoo import mbt2018_mean
from bench import pad

DEVICE = "cuda"


def timed(fn, n=5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0, out


def main():
    torch.set_num_threads(1)  # don't let libtorch/mkl steal threads from the pool test
    model = mbt2018_mean(quality=3, pretrained=True).to(DEVICE)
    model.update(force=True)
    model.eval()

    h, w = 1080, 1920
    x = pad(torch.rand(1, 3, h, w, device=DEVICE))

    with torch.no_grad():
        y = model.g_a(x)
        z = model.h_a(y)
        z_strings = model.entropy_bottleneck.compress(z)
        z_hat = model.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = model.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = model.gaussian_conditional.build_indexes(scales_hat)

        print(f"y shape: {y.shape}")

        # baseline: single-shot compress of the whole y tensor
        t_single, y_strings_ref = timed(lambda: model.gaussian_conditional.compress(y, indexes, means=means_hat), n=5)
        print(f"single-shot gaussian_conditional.compress: {t_single:.2f} ms")

        # sharded: split along height (dim=2) into N independent chunks, compress each
        # independently (valid because mean-scale hyperprior has NO spatial autoregression --
        # each symbol's distribution depends only on its own scale/mean, not on neighboring
        # already-decoded symbols).
        for n_shards in [2, 4, 8, 16, 24]:
            H = y.shape[2]
            if H % n_shards != 0:
                continue
            y_chunks = list(torch.chunk(y, n_shards, dim=2))
            idx_chunks = list(torch.chunk(indexes, n_shards, dim=2))
            mean_chunks = list(torch.chunk(means_hat, n_shards, dim=2))

            def compress_chunk(i):
                return model.gaussian_conditional.compress(y_chunks[i], idx_chunks[i], means=mean_chunks[i])

            def run_sharded():
                with ThreadPoolExecutor(max_workers=n_shards) as ex:
                    return list(ex.map(compress_chunk, range(n_shards)))

            t_sharded, _ = timed(run_sharded, n=5)
            print(f"  sharded x{n_shards:2d} (ThreadPoolExecutor): {t_sharded:.2f} ms   "
                  f"speedup vs single-shot: {t_single/t_sharded:.2f}x")


if __name__ == "__main__":
    main()
