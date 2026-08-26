"""MobileNet-style thinned variant of CompressAI's MeanScaleHyperprior (mbt2018_mean).

Same topology (4x stride-2 stages in g_a/g_s, 2x stride-2 in h_a/h_s) but:
  - standard convs replaced with depthwise-separable convs (depthwise k x k + pointwise 1x1)
  - channel counts reduced (N, M configurable, default much thinner than N=192, M=320)
  - GDN/IGDN replaced with LeakyReLU (GDN has poor kernel support on most inference
    runtimes; this isolates how much of GDN's cost is activation-function overhead)

This is UNTRAINED (random init). It is only valid for latency benchmarking, not for any
rate-distortion / bpp claim.
"""
import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


def dw_sep_conv(in_ch, out_ch, kernel_size=5, stride=2):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding,
                   groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
    )


def dw_sep_deconv(in_ch, out_ch, kernel_size=5, stride=2):
    output_padding = stride - 1
    padding = kernel_size // 2
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding,
                            output_padding=output_padding, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
    )


class ThinMeanScaleHyperprior(CompressionModel):
    def __init__(self, N=64, M=96):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.N = N
        self.M = M

        act = lambda: nn.LeakyReLU(inplace=True)

        self.g_a = nn.Sequential(
            dw_sep_conv(3, N), act(),
            dw_sep_conv(N, N), act(),
            dw_sep_conv(N, N), act(),
            dw_sep_conv(N, M),
        )
        self.g_s = nn.Sequential(
            dw_sep_deconv(M, N), act(),
            dw_sep_deconv(N, N), act(),
            dw_sep_deconv(N, N), act(),
            dw_sep_deconv(N, 3),
        )
        self.h_a = nn.Sequential(
            dw_sep_conv(M, N, stride=1), act(),
            dw_sep_conv(N, N), act(),
            dw_sep_conv(N, N),
        )
        self.h_s = nn.Sequential(
            dw_sep_deconv(N, M), act(),
            dw_sep_deconv(M, M * 3 // 2), act(),
            dw_sep_conv(M * 3 // 2, M * 2, stride=1),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
