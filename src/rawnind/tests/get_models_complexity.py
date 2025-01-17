import sys
import torch
import ptflops

sys.path.append("..")
from rawnind.libs import abstract_trainer
from rawnind.models import raw_denoiser
from rawnind.models import manynets_compression
from rawnind.models import denoise_then_compress
from rawnind.models import compression_autoencoders
from nind_denoise.networks import UtNet

if __name__ == "__main__":
    all_megapix_dims = ((4, 512, 512), (3, 1024, 1024))
    models_to_test = {
        4: {
            "JDDC (Bayer input)": manynets_compression.ManyPriors_RawImageCompressor(
                in_channels=4,
                encoder_cls=compression_autoencoders.BalleEncoder,
                decoder_cls=compression_autoencoders.BayerPSDecoder,
                device=torch.device("cpu"),
            ),
            "DenoiseThenCompress (Bayer input)": denoise_then_compress.DenoiseThenCompress(
                in_channels=4,
                device=torch.device("cpu"),
                encoder_cls=compression_autoencoders.BalleEncoder,
                decoder_cls=compression_autoencoders.BalleDecoder,
            ),
            "JDDC (Pre-upsample)": manynets_compression.ManyPriors_RawImageCompressor(
                in_channels=4,
                device=torch.device("cpu"),
                encoder_cls=compression_autoencoders.BalleEncoder,
                decoder_cls=compression_autoencoders.BalleDecoder,
                preupsample=True,
            ),
            "U-Net (Bayer input)": raw_denoiser.UtNet2(in_channels=4, funit=32),
        },
        3: {
            "Compression or JDC (RGB input)": manynets_compression.ManyPriors_RawImageCompressor(
                in_channels=3,
                device=torch.device("cpu"),
                encoder_cls=compression_autoencoders.BalleEncoder,
                decoder_cls=compression_autoencoders.BalleDecoder,
            ),
            "DenoiseThenCompress (RGB input)": denoise_then_compress.DenoiseThenCompress(
                in_channels=3,
                device=torch.device("cpu"),
                encoder_cls=compression_autoencoders.BalleEncoder,
                decoder_cls=compression_autoencoders.BalleDecoder,
            ),
            "U-Net (RGB input)": raw_denoiser.UtNet2(in_channels=3, funit=32),
            "U-Net (RGB input, full channels)": raw_denoiser.UtNet2(
                in_channels=3, funit=64
            ),
            # "oldunet": UtNet.UtNet(),
        },
    }
    for megapix_dims in all_megapix_dims:
        for model_name, model in models_to_test[megapix_dims[0]].items():
            print(f"Model: {model_name}")
            macs, params = ptflops.get_model_complexity_info(
                model,
                megapix_dims,
            )
            print(f"{model_name} macs: {macs}, params: {params}")

    # print("Bayer complexity")
    # megapixel_dims = 4, 512, 512
    # models = {
    #     "f32model": raw_denoiser.UtNet2(in_channels=4, funit=32).eval(),
    #     "f48model": raw_denoiser.UtNet2(in_channels=4, funit=48).eval(),
    #     "f64model": raw_denoiser.UtNet2(in_channels=4, funit=64).eval(),
    #     "fUtNet3model": raw_denoiser.UtNet3(in_channels=4).eval(),
    # }
    # for model_name, model in models.items():
    #     macs, params = ptflops.get_model_complexity_info(
    #         model,
    #         megapixel_dims,
    #     )
    #     print(f"{model_name} macs: {macs}, params: {params}")
    # print("RGB complexity")
    # megapixel_dims = 3, 1024, 1024
    # model = raw_denoiser.UtNet2(in_channels=3, funit=32).eval()
    # macs, params = ptflops.get_model_complexity_info(model, megapixel_dims)
    # print(f"RGB macs: {macs}, params: {params}")
