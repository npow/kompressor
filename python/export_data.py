import struct
import torch
from compressai.zoo import mbt2018_mean
from bench import pad

DEVICE = "cuda"


def write_i32_vec(f, vec):
    f.write(struct.pack("<I", len(vec)))
    f.write(struct.pack(f"<{len(vec)}i", *vec))


def write_bundle(path, symbols, indexes, cdf, cdf_length, offset, ref_bytes):
    with open(path, "wb") as f:
        write_i32_vec(f, symbols)
        write_i32_vec(f, indexes)
        f.write(struct.pack("<I", len(cdf)))
        for row in cdf:
            write_i32_vec(f, row)
        write_i32_vec(f, cdf_length)
        write_i32_vec(f, offset)
        f.write(struct.pack("<I", len(ref_bytes)))
        f.write(ref_bytes)


def main():
    model = mbt2018_mean(quality=3, pretrained=True).to(DEVICE)
    model.update(force=True)
    model.eval()

    h, w = 1080, 1920
    x = pad(torch.rand(1, 3, h, w, device=DEVICE))

    with torch.no_grad():
        y = model.g_a(x)
        z = model.h_a(y)

        eb = model.entropy_bottleneck
        z_indexes = eb._build_indexes(z.size())
        medians = eb._get_medians().detach()
        medians = eb._extend_ndims(medians, len(z.size()) - 2)
        medians_exp = medians.expand(z.size(0), *([-1] * (len(z.size()) - 1)))
        z_symbols = eb.quantize(z, "symbols", medians_exp)

        z_cdf = eb._quantized_cdf.cpu().int().tolist()
        z_cdf_length = eb._cdf_length.reshape(-1).int().cpu().tolist()
        z_offset = eb._offset.reshape(-1).int().cpu().tolist()
        z_sym_list = z_symbols[0].reshape(-1).int().cpu().tolist()
        z_idx_list = z_indexes[0].reshape(-1).int().cpu().tolist()

        z_strings = eb.compress(z)
        z_ref_bytes = z_strings[0]

        print(f"z shape: {z.shape}, num z symbols: {len(z_sym_list)}, z cdf rows: {len(z_cdf)}, z ref bytes: {len(z_ref_bytes)}")

        write_bundle("../data_z.bin", z_sym_list, z_idx_list, z_cdf, z_cdf_length, z_offset, z_ref_bytes)

        # also re-derive full pipeline timing reference (for sanity on end-to-end ms later)
        z_hat = eb.decompress(z_strings, z.size()[-2:])
        gaussian_params = model.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        gc = model.gaussian_conditional
        indexes = gc.build_indexes(scales_hat)
        symbols = gc.quantize(y, "symbols", means_hat)
        cdf = gc._quantized_cdf.cpu().int().tolist()
        cdf_length = gc._cdf_length.reshape(-1).int().cpu().tolist()
        offset = gc._offset.reshape(-1).int().cpu().tolist()
        sym_list = symbols[0].reshape(-1).int().cpu().tolist()
        idx_list = indexes[0].reshape(-1).int().cpu().tolist()
        strings = gc.compress(y, indexes, means=means_hat)
        ref_bytes = strings[0]
        write_bundle("../data.bin", sym_list, idx_list, cdf, cdf_length, offset, ref_bytes)
        print(f"y shape: {y.shape}, num y symbols: {len(sym_list)}, y ref bytes: {len(ref_bytes)}")

    print("wrote data.bin and data_z.bin")


if __name__ == "__main__":
    main()
