import time
import torch
import torch.multiprocessing as mp
from compressai.zoo import mbt2018_mean
from bench import pad

DEVICE = "cuda"


def worker_compress(args):
    model, y_chunk, idx_chunk, mean_chunk = args
    return model.gaussian_conditional.compress(y_chunk, idx_chunk, means=mean_chunk)


def main():
    torch.set_num_threads(1)
    model = mbt2018_mean(quality=3, pretrained=True)
    model.update(force=True)
    model.eval()
    model.share_memory()

    h, w = 1080, 1920
    x = pad(torch.rand(1, 3, h, w))

    with torch.no_grad():
        y = model.g_a(x)
        z = model.h_a(y)
        z_strings = model.entropy_bottleneck.compress(z)
        z_hat = model.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = model.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = model.gaussian_conditional.build_indexes(scales_hat)

        t0 = time.perf_counter()
        y_strings_ref = model.gaussian_conditional.compress(y, indexes, means=means_hat)
        t_single = (time.perf_counter() - t0) * 1000
        print(f"single-shot (CPU model, 1 thread): {t_single:.2f} ms")

        for n_shards in [4, 8]:
            H = y.shape[2]
            if H % n_shards != 0:
                continue
            y_chunks = list(torch.chunk(y, n_shards, dim=2))
            idx_chunks = list(torch.chunk(indexes, n_shards, dim=2))
            mean_chunks = list(torch.chunk(means_hat, n_shards, dim=2))
            args = [(model, y_chunks[i], idx_chunks[i], mean_chunks[i]) for i in range(n_shards)]

            t0 = time.perf_counter()
            with mp.Pool(processes=n_shards) as pool:
                results = pool.map(worker_compress, args)
            t_sharded = (time.perf_counter() - t0) * 1000
            print(f"  sharded x{n_shards} (multiprocessing.Pool, includes pool startup): "
                  f"{t_sharded:.2f} ms   speedup: {t_single/t_sharded:.2f}x")

            # second call reusing a warm pool to separate startup cost from steady-state
            t0 = time.perf_counter()
            with mp.Pool(processes=n_shards) as pool:
                pool.map(worker_compress, args)  # warm
                t0 = time.perf_counter()
                results = pool.map(worker_compress, args)
                t_sharded_warm = (time.perf_counter() - t0) * 1000
            print(f"  sharded x{n_shards} (warm pool, steady-state): {t_sharded_warm:.2f} ms   "
                  f"speedup: {t_single/t_sharded_warm:.2f}x")


if __name__ == "__main__":
    main()
