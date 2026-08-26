import time
import torch
from compressai.zoo import mbt2018_mean
from bench import pad

DEVICE = "cuda"


def timed(fn, n=10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0, out


def main():
    torch.set_num_threads(24)
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
        symbols = model.gaussian_conditional.quantize(y, "symbols", means_hat)

        print(f"num symbols: {symbols.numel()}")

        t_tolist_sym, sym_list = timed(lambda: symbols[0].reshape(-1).int().tolist(), n=10)
        t_tolist_idx, idx_list = timed(lambda: indexes[0].reshape(-1).int().tolist(), n=10)
        print(f"symbols.tolist(): {t_tolist_sym:.2f} ms")
        print(f"indexes.tolist(): {t_tolist_idx:.2f} ms")

        cdf = model.gaussian_conditional._quantized_cdf
        cdf_len = model.gaussian_conditional._cdf_length.reshape(-1).int()
        offset = model.gaussian_conditional._offset.reshape(-1).int()
        t_cdf_tolist, cdf_list = timed(lambda: cdf.tolist(), n=10)
        cdf_len_list = cdf_len.tolist()
        offset_list = offset.tolist()
        print(f"cdf.tolist() ({cdf.numel()} elements): {t_cdf_tolist:.2f} ms")

        # now the actual C++ coder call, with pre-marshaled python lists (marshaling excluded)
        t_encode, rv = timed(
            lambda: model.gaussian_conditional.entropy_coder.encode_with_indexes(
                sym_list, idx_list, cdf_list, cdf_len_list, offset_list
            ),
            n=10,
        )
        print(f"encode_with_indexes (C++ coder only, pre-marshaled): {t_encode:.2f} ms")

        t_full, _ = timed(lambda: model.gaussian_conditional.compress(y, indexes, means=means_hat), n=10)
        print(f"\nfull compress() call (marshal + encode, as measured before): {t_full:.2f} ms")
        marshal_total = t_tolist_sym + t_tolist_idx  # cdf/len/offset are cached across calls in real code, but compress() re-tolists cdf every call too:
        print(f"  of which symbols+indexes .tolist(): {marshal_total:.2f} ms ({marshal_total/t_full*100:.1f}%)")
        print(f"  of which cdf/len/offset .tolist() (redone EVERY call in compress()): {t_cdf_tolist:.2f} ms ({t_cdf_tolist/t_full*100:.1f}%)")
        print(f"  of which actual C++ coding: {t_encode:.2f} ms ({t_encode/t_full*100:.1f}%)")


if __name__ == "__main__":
    main()
