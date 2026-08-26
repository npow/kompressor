"""Export several test vectors covering normal + edge cases (incl. the bypass path)
straight from CompressAI's real GaussianConditional / EntropyBottleneck, the same
way compressai's own test_entropy_models.py validates round-trips -- so we can check
the Rust port against real Python-encoded bytes, not just against itself.
"""
import struct
import torch
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


def write_i32_vec(f, vec):
    f.write(struct.pack("<I", len(vec)))
    f.write(struct.pack(f"<{len(vec)}i", *vec))


def dump_case(f, name, symbols, indexes, cdf, cdf_length, offset, py_bytes, py_decoded_symbols):
    name_b = name.encode("utf-8")
    f.write(struct.pack("<I", len(name_b)))
    f.write(name_b)
    write_i32_vec(f, symbols)
    write_i32_vec(f, indexes)
    f.write(struct.pack("<I", len(cdf)))
    for row in cdf:
        write_i32_vec(f, row)
    write_i32_vec(f, cdf_length)
    write_i32_vec(f, offset)
    f.write(struct.pack("<I", len(py_bytes)))
    f.write(py_bytes)
    write_i32_vec(f, py_decoded_symbols)


def gaussian_case(name, scales, means, x):
    gc = GaussianConditional(None)
    gc.update_scale_table(torch.exp(torch.linspace(torch.log(torch.tensor(0.11)), torch.log(torch.tensor(64.0)), 64)))
    gc.update()
    indexes = gc.build_indexes(scales)
    symbols = gc.quantize(x, "symbols", means)
    strings = gc.compress(x, indexes, means=means)
    py_bytes = strings[0]
    decoded = gc.decompress(strings, indexes, means=means)
    py_decoded_symbols = gc.quantize(decoded, "symbols", means).reshape(-1).int().tolist()
    # sanity per compressai's own test philosophy: decompressed should equal round(x-mean)+mean
    assert torch.allclose(torch.round(x - means) + means, decoded), f"{name}: compressai self round-trip failed!"

    cdf = gc._quantized_cdf.int().tolist()
    cdf_length = gc._cdf_length.reshape(-1).int().tolist()
    offset = gc._offset.reshape(-1).int().tolist()
    sym_list = symbols.reshape(-1).int().tolist()
    idx_list = indexes.reshape(-1).int().tolist()
    return name, sym_list, idx_list, cdf, cdf_length, offset, py_bytes, py_decoded_symbols


def eb_case(name, x):
    eb = EntropyBottleneck(x.size(1))
    eb.update(force=True)
    indexes = eb._build_indexes(x.size())
    medians = eb._get_medians().detach()
    medians = eb._extend_ndims(medians, len(x.size()) - 2)
    medians_exp = medians.expand(x.size(0), *([-1] * (len(x.size()) - 1)))
    symbols = eb.quantize(x, "symbols", medians_exp)

    strings = eb.compress(x)
    py_bytes = strings[0]
    decoded = eb.decompress(strings, x.size()[2:])
    assert torch.allclose(torch.round(x - medians_exp) + medians_exp, decoded), f"{name}: compressai self round-trip failed!"
    py_decoded_symbols = eb.quantize(decoded, "symbols", medians_exp).reshape(-1).int().tolist()

    cdf = eb._quantized_cdf.int().tolist()
    cdf_length = eb._cdf_length.reshape(-1).int().tolist()
    offset = eb._offset.reshape(-1).int().tolist()
    sym_list = symbols.reshape(-1).int().tolist()
    idx_list = indexes.reshape(-1).int().tolist()
    return name, sym_list, idx_list, cdf, cdf_length, offset, py_bytes, py_decoded_symbols


def main():
    torch.manual_seed(0)
    cases = []

    # 1. normal small case, moderate scales
    scales = torch.rand(1, 8, 16, 16) * 5 + 0.5
    means = torch.randn(1, 8, 16, 16) * 2
    x = means + torch.randn_like(means) * scales
    cases.append(gaussian_case("gaussian_normal", scales, means, x))

    # 2. edge case: huge outliers relative to tiny scales -> forces bypass path
    scales = torch.full((1, 4, 8, 8), 0.11)  # near the minimum allowed scale
    means = torch.zeros(1, 4, 8, 8)
    x = torch.zeros(1, 4, 8, 8)
    x[0, 0, 0, 0] = 500.0    # huge positive outlier
    x[0, 0, 0, 1] = -500.0   # huge negative outlier
    x[0, 1, 2, 3] = 1e4
    x[0, 2, 3, 4] = -1e4
    cases.append(gaussian_case("gaussian_bypass_outliers", scales, means, x))

    # 3. large-ish random case with wide scale range (like real model output)
    scales = torch.exp(torch.randn(1, 32, 40, 60) * 1.5)
    means = torch.randn(1, 32, 40, 60) * 3
    x = means + torch.randn_like(means) * scales
    cases.append(gaussian_case("gaussian_wide_scale_range", scales, means, x))

    # 4. entropy bottleneck, moderate size
    x = torch.randn(1, 16, 24, 24) * 3
    cases.append(eb_case("entropy_bottleneck_normal", x))

    # 5. entropy bottleneck with outliers
    x = torch.randn(1, 8, 12, 12) * 2
    x[0, 0, 0, 0] = 200.0
    x[0, 1, 1, 1] = -200.0
    cases.append(eb_case("entropy_bottleneck_outliers", x))

    with open("../test_vectors.bin", "wb") as f:
        f.write(struct.pack("<I", len(cases)))
        for c in cases:
            dump_case(f, *c)
            n_bypass_like = "n/a"
            print(f"case '{c[0]}': {len(c[1])} symbols, {len(c[3])} cdf rows, {len(c[6])} bytes")

    print("wrote test_vectors.bin")


if __name__ == "__main__":
    main()
