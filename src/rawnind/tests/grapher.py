"""Grapher for RawNIND. Also save results to csv.
Don't forget to run python tools/list_strictly_worse_plotted_models.py
"""

print(__doc__)


from io import BytesIO
import os
import re
import sys
import configargparse
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import csv
import subprocess

sys.path.append("..")
from common.libs import json_saver
from common.libs import utilities
from rawnind.tools.test_all_known import (
    TESTS,
    MODEL_TYPES,
    MODEL_INPUTS,
    MODELS_ROOT_DIR,
)

ET.register_namespace("", "http://www.w3.org/2000/svg")


TRAINED_MODELS_YAMLFPATHS = {
    "denoise": os.path.join("config", "trained_denoise_models.yaml"),
    "dc": os.path.join("config", "trained_dc_models.yaml"),
}
MODELS_DEFINITIONS_YAMLFPATHS = {
    "denoise": os.path.join("config", "graph_denoise_models_definitions.yaml"),
    "dc": os.path.join("config", "graph_dc_models_definitions.yaml"),
}
METRICS = ("mse", "msssim")
METRICS_LOSS = {"mse": "mse", "msssim": "msssim_loss", "psnr": "mse"}
TESTS = [
    "val",
    "test",
    # "ext_raw_denoise_test",
    # "ext_raw_nik_denoise_test",
    "manproc",
    "manproc_bostitch",
    "manproc_hq",
    "manproc_gt",
    "manproc_q99",
    "manproc_q995",
    "playraw",
    "manproc_playraw",
]

MARKERS = {
    "bayer": "s",
    "JDDC (bayer input) + dev.": "s",
    "Linear RGB": "^",
    "JDC (Linear RGB input) + dev.": "^",
    "passthrough": "_",
    "BM3D (sRGB)": "$3$",
    "LinRGB (extra pairs)": "+",
    "JDC LinRGB (extra pairs)": "+",
    "bayer (extra pairs)": "+",
    "JDC bayer (extra pairs)": "+",
    "bayer (no unpaired data)": "$D$",
    "JDDC (no clean data, bayer input) + dev.": "$D$",
    "LinRGB (no unpaired data)": "$D$",
    "JDC (no clean data, LinRGB input) + dev.": "$D$",
    "bayer (pre-upsampled)": "D",
    "JDDC (upsampled bayer input) + dev.": "D",
    "bayer (more channels)": "h",
    "JDDC (more channels, bayer input) + dev.": "h",
    "sRGB": "$s$",
    "JDC (developed input) [COMPDENOISE]": "$s$",
    "Compression AE (bayer input) + dev.": "$c$",
    "Compression AE (Linear RGB input) + dev.": "$c$",
    "Denoise then compress (LinRGB input) + dev.": "$2$",
    "LinRGB (w/gamma)": "*",
    "JDC LinRGB (w/gamma)": "*",
    "bayer (w/gamma)": "*",
    "JDC bayer (w/gamma)": "*",
    "JPEG XL (developed input) [JPEGXL]": ".",
    "JPEG XL (Linear RGB input) [JPEGXL] + dev.": ".",
}
COLORS = {
    "bayer": "green",
    "JDDC (bayer input) + dev.": "green",
    "Linear RGB": "red",
    "JDC (Linear RGB input) + dev.": "red",
    "passthrough": "tab:gray",
    "BM3D (sRGB)": "black",
    "LinRGB (extra pairs)": "maroon",
    "JDC LinRGB (extra pairs)": "maroon",
    "bayer (extra pairs)": "darkgreen",
    "JDC bayer (extra pairs)": "darkgreen",
    "bayer (no unpaired data)": "springgreen",
    "JDDC (no clean data, bayer input) + dev.": "springgreen",
    "LinRGB (no unpaired data)": "lightcoral",
    "JDC (no clean data, LinRGB input) + dev.": "lightcoral",
    "bayer (pre-upsampled)": "greenyellow",
    "JDDC (upsampled bayer input) + dev.": "greenyellow",
    "bayer (more channels)": "mediumturquoise",
    "JDDC (more channels, bayer input) + dev.": "mediumturquoise",
    "sRGB": "orange",
    "JDC (developed input) [COMPDENOISE]": "orange",
    "Compression AE (bayer input) + dev.": "palegreen",
    "Compression AE (Linear RGB input) + dev.": "lightpink",
    "Denoise then compress (LinRGB input) + dev.": "peru",
    "LinRGB (w/gamma)": "tab:purple",
    "JDC LinRGB (w/gamma)": "tab:purple",
    "bayer (w/gamma)": "darkolivegreen",
    "JDC bayer (w/gamma)": "darkolivegreen",
    "JPEG XL (developed input) [JPEGXL]": "slateblue",
    "JPEG XL (Linear RGB input) [JPEGXL] + dev.": "purple",
}
LINEWIDTH = 0.15
# MINMAX_TO_MEAN_MSSSIM = {  # TODO check this since we are working w/ manproc
#     "ge": {
#         0.0: 0.9471623671631659,
#         0.1: 0.9471623671631659,
#         0.2: 0.9471623671631659,
#         0.3: 0.9471623671631659,
#         0.4: 0.9471623671631659,
#         0.5: 0.9471623671631659,
#         0.55: 0.9528887702900672,
#         0.6: 0.9528887702900672,
#         0.65: 0.9577384954803928,
#         0.7: 0.9632890114317769,
#         0.75: 0.9668514581725878,
#         0.8: 0.9696365957384678,
#         0.85: 0.9752269836584317,
#         0.9: 0.9857954731090464,
#         0.95: 0.9908681560998485,
#         0.96: 0.991699353873151,
#         0.97: 0.9930015209362592,
#         0.98: 0.9955494141904637,
#         0.99: 0.9972834067112332,
#         1.0: 1.0,
#     },
#     "le": {
#         0.55: 0.5386789441108704,
#         0.6: 0.5386789441108704,
#         0.65: 0.5752351880073547,
#         0.7: 0.6133408308029175,
#         0.75: 0.638195092861469,
#         0.8: 0.664829870685935,
#         0.85: 0.7104434280291848,
#         0.9: 0.7919600051262475,
#         0.95: 0.8307980092768931,
#         0.96: 0.8370669360160827,
#         0.97: 0.8493611401599237,
#         0.98: 0.8775720073935691,
#         0.99: 0.8977294883411591,
#         1.0: 0.9471623671631659,
#     },
# }


PROGRESSIVE_DENOISING_TESTNAMES = [
    # "test",
    "test_manproc_rawnind",
    "test_manproc_bostitch_rawnind",
]
DENOISING_TESTNAMES = ["manproc", "manproc_bostitch"]
# for testname in PROGRESSIVE_TESTNAMES:
#     for operator in MINMAX_TO_MEAN_MSSSIM.keys():
#         TESTS.extend(
#             [
#                 f"progressive_{testname}_msssim_{operator}_{n}"
#                 for n in MINMAX_TO_MEAN_MSSSIM[operator]
#             ]
#         )
MAX_BITRATE = 2


"""
different graphs:
    - denoise @ progressive noise: y-axis: msssim, x-axis: noise level
    - rate-distortion (all)
"""
# also in mk_megafig.py
LITERATURE = {
    "jddc": {
        "[BM3D]": "[3]",
        "[NIND]": "[10]",
        "[OURS]": "",
        "[JPEGXL]": "[17]",
        "[COMPDENOISE]": "[24]",
        "[MANYPRIORS]": "[27]",
    },
    "thesis": {
        "[BM3D]": "",
        "[NIND]": "",
        "[OURS]": "(Ch. 5)",
        "[JPEGXL]": "",
        "[COMPDENOISE]": "(Ch. 4)",
        "[MANYPRIORS]": "(Ch. 3)",
    },
}


def add_citation(model_name: str, paper: str):
    assert paper in LITERATURE
    for citation_name, citation_val in LITERATURE[paper].items():
        if citation_name in model_name:
            return model_name.replace(citation_name, citation_val)
    return model_name


def add_default_values_to_models_list(models_list):
    for model_attrs in models_list.values():
        model_attrs.setdefault("processed_input", False)
        model_attrs.setdefault("preupsample", False)
        model_attrs.setdefault("data_pairing", "x_y")
        model_attrs.setdefault("transfer_fun", "None")
        model_attrs.setdefault("morechans", False)
        # model_attrs.setdefault("lambda", None)
        model_attrs.setdefault("extrapairs_B", False)
        model_attrs.setdefault("incl_cc", True)
        model_attrs.setdefault("passthrough", False)
        model_attrs.setdefault("denoise_then_compress", False)
        model_attrs.setdefault("bm3d", False)
        model_attrs.setdefault("noownpics", False)
        model_attrs.setdefault("JPEGXL", "False")


def load_models_results(
    model_types: list[str] = MODEL_TYPES, strictly_worse_models: list[str] = []
):
    def get_load_metric_key(model_attrs: dict) -> str:
        if "val_key" in model_attrs:
            return model_attrs["val_key"]
        if "lambda" not in model_attrs:
            res = f"val_{model_attrs.get('loss', 'msssim_loss')}"
            # breakpoint()
        else:
            res = "val_combined"
        if model_attrs["processed_input"]:
            res += ".arbitraryproc"
        else:
            res += "." + str(model_attrs.get("transfer_fun"))
        return res

    def get_model_results(model_attrs):
        model_results_fpath = os.path.join(
            MODELS_ROOT_DIR, "rawnind_" + model_type, model_name, "trainres.yaml"
        )
        assert os.path.isfile(model_results_fpath), model_results_fpath
        model_results_handler = json_saver.YAMLSaver(model_results_fpath)
        load_metric_key = get_load_metric_key(model_attrs)
        # if model_attrs["passthrough"]:
        #     breakpoint()
        try:
            model_attrs["results"] = model_results_handler.get_best_step_results(
                load_metric_key
            )
        except KeyError:
            print(f"KeyError: {load_metric_key} not found in {model_results_fpath}.")
            if re.search(r"\.[a-zA-Z]", load_metric_key):
                load_metric_key = load_metric_key.rpartition(".")[0]
            else:
                load_metric_key = load_metric_key + ".None"
            try:
                model_attrs["results"] = model_results_handler.get_best_step_results(
                    load_metric_key
                )
                print(f"Found {load_metric_key} instead.")
            except KeyError:
                print("Still unable to find the right key with/without .None")
                print(f"{model_attrs=}, {load_metric_key=}")
                breakpoint()

    def losses_to_metrics(model_results):
        new_keys = {}
        for reskey, resval in model_results.items():
            if "msssim_loss" in reskey:
                new_keys[reskey.replace("msssim_loss", "msssim")] = 1 - resval
        model_results.update(new_keys)

    def sanitize_loss_key(model_results):
        for reskey in list(model_results.keys()):
            # if "manproc" in reskey and reskey.endswith(".arbitraryproc"):
            #     model_results[reskey.replace(".arbitraryproc", "")] = model_results.pop(
            #         reskey
            #     )
            if re.search(r"\.[a-zA-Z]", reskey):
                # if reskey.endswith(".None"):
                model_results[reskey.rpartition(".")[0]] = model_results.pop(reskey)

    models = {}
    for model_type in model_types:
        models[model_type] = utilities.load_yaml(
            TRAINED_MODELS_YAMLFPATHS[model_type], error_on_404=True
        )
        add_default_values_to_models_list(models[model_type])
        for model_name, model_attrs in models[model_type].items():
            get_model_results(model_attrs)
            losses_to_metrics(model_attrs["results"])
            sanitize_loss_key(model_attrs["results"])
            # model_attrs['results'] = utilities.sort_dictionary(model_attrs['results'])
            if model_name in strictly_worse_models:
                model_attrs["results"]["strictly_worse"] = True
    return models


def get_models_definitions(model_types: list[str] = MODEL_TYPES):
    models_definitions = {}
    for model_type in model_types:
        models_definitions[model_type] = utilities.load_yaml(
            MODELS_DEFINITIONS_YAMLFPATHS[model_type], error_on_404=True
        )
        add_default_values_to_models_list(models_definitions[model_type])
    return models_definitions


def group_relevant_models(
    models_definitions, models_results, metric: str, reduce: bool = False
) -> dict[str, list[dict[str, dict[str, float]]]]:
    """Group models by definition"""
    models_per_definition = {}
    for def_key, model_definition in models_definitions.items():
        assert def_key not in models_per_definition
        models_per_definition[def_key] = []
        best_model_attrs = best_model_name = None
        for model_name, model_attrs in models_results.items():
            # if any of the attributes differ, skip
            # print(model_attrs["passthrough"])

            if any(
                [
                    model_attrs[attr] != model_definition[attr]
                    for attr in model_definition
                ]
            ) or (
                not model_attrs["passthrough"]
                and METRICS_LOSS[metric] not in model_attrs.get("loss")
            ):
                # if (
                #     model_name
                #     == "DenoiseThenCompressDCTrainingProfiledRGBToProfiledRGB_3ch_L128.0_Balle_Balle_2024-06-14-dc_msssim_mgout_prgb_x_x_notrans_valeither_128from128_-3"
                #     and model_definition["denoise_then_compress"]
                #     and metric == "msssim"
                # ):
                #     breakpoint()
                continue

            try:
                if reduce and (
                    best_model_name is None
                    or model_attrs["results"][f"val_{metric}"]
                    > best_model_attrs["results"][f"val_{metric}"]
                ):
                    best_model_attrs = model_attrs
                    best_model_name = model_name
                    continue
                else:
                    models_per_definition[def_key].append(
                        {model_name: model_attrs["results"]}
                    )
            except KeyError as e:
                print(e)
                breakpoint()
        if reduce and best_model_name is not None:
            models_per_definition[def_key].append(
                {best_model_name: best_model_attrs["results"]}
            )
    return models_per_definition


def plot_tooltip_points(tooltips_coordinates_to_name: dict[tuple[float, float], str]):
    return
    for (x, y), name in tooltips_coordinates_to_name.items():
        plt.annotate(
            name, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

        # plt.gca().text(x, y, name, fontsize=6)


def plot_rd_curves(
    models_results,
    models_definitions,
    metrics=METRICS,
    tests=TESTS,
    visualize=False,
    max_bitrate=MAX_BITRATE,
    paper: str = "jddc",
):
    AXIS_LIMITS = [
        {
            "metric": "msssim",
            "test": "manproc",
            "max_bitrate": 0.32,
            "min_bitrate": 0.027,
            "min_metric": 0.903,
            "max_metric": 0.95,
            "min_metric_sub": 0.73,
            "max_metric_sub": 0.93,
            "mean_input": 0.763277364,
        },
        {
            "metric": "mse",
            "test": "manproc",
            "max_bitrate": 0.5,
            "min_bitrate": 0.05,
        },
        {
            "metric": "msssim",
            "test": "playraw",
            "max_bitrate": 0.23,
            "min_bitrate": 0.03,
            "min_metric": 0.992,
            "max_metric": 0.998,
        },
        {
            "metric": "msssim",
            "test": "manproc_playraw",
            "max_bitrate": 0.36,
            "min_bitrate": 0.03,
            "min_metric": 0.908,
            "max_metric": 0.971,
        },
        {
            "metric": "msssim",
            "test": "manproc_hq",
            "max_bitrate": 0.36,
            "min_bitrate": 0.03,
            "min_metric": 0.90,
            "max_metric": 0.96,
        },
        {
            "metric": "msssim",
            "test": "manproc_gt",
            "max_bitrate": 0.27,
            "min_bitrate": 0.03,
            "min_metric": 0.91,
            "max_metric": 0.975,
            "min_metric_sub": 0.75,
            "max_metric_sub": 0.91,
            "mean_input": 1.00,
        },
        {
            "metric": "msssim",
            "test": "manproc_q99",
            "max_bitrate": 0.281,
            "min_bitrate": 0.027,
            "min_metric": 0.91,
            "max_metric": 0.966,
            "min_metric_sub": 0.76,
            "max_metric_sub": 0.90,
            "mean_input": 0.933550062,
        },
        {
            "metric": "msssim",
            "test": "manproc_q995",
            "max_bitrate": 0.281,
            "min_bitrate": 0.027,
            "min_metric": 0.91,
            "max_metric": 0.966,
            "min_metric_sub": 0.79,
            "max_metric_sub": 0.90,
            "mean_input": 0.96110374,
        },
        {
            "metric": "mse",
            "test": "playraw",
            "max_bitrate": 0.5,
            "min_bitrate": 0.05,
        },
        {
            "metric": "msssim",
            "test": "ext_raw_denoise_test",
            "max_bitrate": 0.5,
            "min_bitrate": 0.05,
        },
    ]

    # Define the models that should be plotted behind and in front
    BACKGROUND_MODELS = [
        "JPEG XL (developed input) [JPEGXL]",
        "JPEG XL (Linear RGB input) [JPEGXL] + dev.",
    ]
    FRONT_MODELS = [
        "JDDC (bayer input) + dev.",
        "JDC (Linear RGB input) + dev.",
        "JDC (developed input) [COMPDENOISE]",
    ]

    for metric in metrics:
        grouped_models = group_relevant_models(
            models_definitions["dc"], models_results["dc"], metric
        )
        for test in tests:
            figs2plot = None
            for bitrate_limits_dict in AXIS_LIMITS:
                if (
                    bitrate_limits_dict["metric"] == metric
                    and bitrate_limits_dict["test"] == test
                ):
                    if "min_metric_sub" in bitrate_limits_dict:
                        fig, (ax1, ax2) = plt.subplots(
                            2, 1, sharex=True, gridspec_kw={"height_ratios": [5, 1]}
                        )
                        ax1.set_ylim(
                            bitrate_limits_dict["min_metric"],
                            bitrate_limits_dict["max_metric"],
                        )
                        ax2.set_ylim(
                            bitrate_limits_dict["min_metric_sub"],
                            bitrate_limits_dict["max_metric_sub"],
                        )
                        figs2plot = [ax1, ax2]
                    else:
                        if (
                            "min_metric" in bitrate_limits_dict
                            or "max_metric" in bitrate_limits_dict
                        ):
                            plt.gca().set_ylim(
                                bitrate_limits_dict.get("min_metric", None),
                                bitrate_limits_dict.get("max_metric", None),
                            )
                        figs2plot = [plt]
                    break
            else:
                print(f"No {test=} with {metric=}. Using default values")
                bitrate_limits_dict = {"min_bitrate": None, "max_bitrate": None}
                figs2plot = [plt]
            # Check that figs2plot exists
            assert figs2plot is not None, figs2plot

            csv_results = []
            tooltips_coordinates_to_name = {}  # for tooltip
            # Models to ignore go here
            for grouped_model_name, individual_models in grouped_models.items():
                if (
                    (
                        grouped_model_name
                        == "Denoise then compress (LinRGB input) + dev."
                        and metric == "msssim"
                        and "playraw" in test
                    )
                    or ("w/gamma" in grouped_model_name and "manproc" in test)
                    or (
                        grouped_model_name == "Compression AE (bayer input) + dev."
                        and "manproc" in test
                    )
                    or (
                        grouped_model_name == "JDDC (upsampled bayer input) + dev."
                        and "manproc_" in test
                    )
                ):
                    print(f"Skipping {grouped_model_name=} for {test=}")
                    continue

                x_values = []
                y_values = []
                for individual_model in individual_models:
                    individual_model_results = next(iter(individual_model.values()))
                    if f"{test}_{metric}" not in individual_model_results:
                        continue
                    try:
                        bpp_val = individual_model_results[f"{test}_bpp"]
                    except KeyError as e:
                        print(
                            f"error {e} with {individual_model.keys()=} missing key {test}_bpp"
                        )
                        continue
                    metric_val = individual_model_results[f"{test}_{metric}"]

                    if individual_model_results.get("strictly_worse", False):
                        csv_results.append(
                            (
                                grouped_model_name,
                                next(iter(individual_model.keys())),
                                bpp_val,
                                metric_val,
                                True,
                            )
                        )
                    else:
                        x_values.append(bpp_val)
                        y_values.append(metric_val)
                        tooltips_coordinates_to_name[x_values[-1], y_values[-1]] = next(
                            iter(individual_model.keys())
                        )  # for tooltip and csv
                        csv_results.append(
                            (
                                grouped_model_name,
                                next(iter(individual_model.keys())),
                                x_values[-1],
                                y_values[-1],
                                False,
                            )
                        )
                if len(x_values) == 0:
                    print(f"No {test=} for {grouped_model_name=}")
                    continue
                # Sort x_values, y_values by x_values
                x_values, y_values = zip(*sorted(zip(x_values, y_values)))

                for afig in figs2plot:
                    # Determine zorder based on model type
                    if grouped_model_name in BACKGROUND_MODELS:
                        z_order = 3  # Lower zorder to plot behind
                    elif grouped_model_name in FRONT_MODELS:
                        z_order = 5  # Higher zorder to plot on top
                    else:
                        z_order = 4  # Default zorder

                    afig.plot(
                        x_values,
                        y_values,
                        label=add_citation(grouped_model_name, paper),
                        marker=MARKERS.get(grouped_model_name, "o"),
                        color=COLORS.get(grouped_model_name, "blue"),
                        linewidth=LINEWIDTH,
                        zorder=z_order,  # Set the zorder
                    )
                plot_tooltip_points(tooltips_coordinates_to_name)  # tooltip

            # Check if there is anything to plot
            if len(plt.gca().lines) == 0:
                print(f"No {test=} done with {metric=}")
                continue
            # Set x-y limits
            max_bitrate = bitrate_limits_dict["max_bitrate"]
            min_bitrate = bitrate_limits_dict["min_bitrate"]
            if "min_metric_sub" in bitrate_limits_dict:
                ax1.grid(True, which="both", zorder=0)
                ax2.grid(True, which="both", zorder=0)
                ax1.set_ylabel(
                    metric.replace(
                        "msssim", "Processed output mean quality (MS-SSIM) →"
                    )
                )
                # Hide spines between axes
                ax1.spines["bottom"].set_visible(False)
                ax2.spines["top"].set_visible(False)
                ax1.tick_params(bottom=False)
                fig.subplots_adjust(hspace=0.05)
                # Plot horizontal lines with low zorder
                ax1.axhline(
                    y=bitrate_limits_dict["min_metric"],
                    color="gray",
                    linestyle="--",
                    linewidth=0.5,
                    zorder=0,  # Ensure it's in the background
                )
                ax2.axhline(
                    y=bitrate_limits_dict["max_metric_sub"],
                    color="gray",
                    linestyle="--",
                    linewidth=0.5,
                    zorder=0,  # Ensure it's in the background
                )
                if "mean_input" in bitrate_limits_dict:
                    ax1.axhline(
                        y=bitrate_limits_dict["mean_input"],
                        color="black",
                        linestyle="dashdot",
                        linewidth=0.5,
                        label="Mean developed input MS-SSIM",
                        zorder=0,  # Ensure it's in the background
                    )
                    ax2.axhline(
                        y=bitrate_limits_dict["mean_input"],
                        color="black",
                        linestyle="dashdot",
                        linewidth=0.5,
                        label="Mean developed input MS-SSIM",
                        zorder=0,  # Ensure it's in the background
                    )
                ax1.legend(prop={"size": 7})
            else:
                if "mean_input" in bitrate_limits_dict:
                    plt.axhline(
                        y=bitrate_limits_dict["mean_input"],
                        color="black",
                        linestyle="dashdot",
                        linewidth=0.5,
                        label="Mean developed input MS-SSIM",
                        zorder=0,  # Ensure it's in the background
                    )
                plt.gca().set_ylabel(
                    metric.replace("msssim", "Mean output quality (MS-SSIM) →")
                )
                plt.legend()
                plt.grid(True, zorder=0)
            # plt.title(f"{test=} with {metric=}")
            # Show x-y labels
            plt.gca().set_xlim([min_bitrate, max_bitrate])
            plt.gca().set_xlabel("← bits per pixel (bpp)")
            # Save plot
            if visualize:
                plt.show()
            plt.savefig(f"plots/rd_{test}_{metric}_{paper}.svg")
            plt.savefig(f"plots/rd_{test}_{metric}_{paper}.pdf")
            # run pdfcrop on the pdf
            subprocess.run(["pdfcrop", f"plots/rd_{test}_{metric}_{paper}.pdf"])

            plt.clf()
            # Save CSV
            csv_header = (
                "model_grouped",
                "model_individual",
                "bpp",
                metric,
                "strictly_worse",
            )
            with open(f"plots/rd_{test}_{metric}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
                writer.writerows(csv_results)


def add_svg_tooltips(num_indices: int, output_fpath):
    f = BytesIO()
    plt.savefig(f, format="svg")
    # --- Add interactivity ---

    # Create XML tree from the SVG file.
    tree, xmlid = ET.XMLID(f.getvalue())
    tree.set("onload", "init(event)")

    for i in range(0, num_indices):
        try:
            # Get the index of the shape
            # Hide the tooltips
            tooltip = xmlid[f"mytooltip_{i:03d}"]
            tooltip.set("visibility", "hidden")
            # Assign onmouseover and onmouseout callbacks to patches.
            mypatch = xmlid[f"mypatch_{i:03d}"]
            mypatch.set("onmouseover", "ShowTooltip(this)")
            mypatch.set("onmouseout", "HideTooltip(this)")
        except KeyError as e:
            print(
                f"KeyError: {e}. {i=}, {num_indices=}; unable to save annotations for {output_fpath=}"
            )
            # breakpoint()
            continue

    # This is the script defining the ShowTooltip and HideTooltip functions.
    script = """
        <script type="text/ecmascript">
        <![CDATA[

        function init(event) {
            if ( window.svgDocument == null ) {
                svgDocument = event.target.ownerDocument;
                }
            }

        function ShowTooltip(obj) {
            var cur = obj.id.split("_")[1];
            var tip = svgDocument.getElementById('mytooltip_' + cur);
            tip.setAttribute('visibility', "visible")
            }

        function HideTooltip(obj) {
            var cur = obj.id.split("_")[1];
            var tip = svgDocument.getElementById('mytooltip_' + cur);
            tip.setAttribute('visibility', "hidden")
            }

        ]]>
        </script>
        """

    # Insert the script at the top of the file and save it.
    tree.insert(0, ET.XML(script))
    ET.ElementTree(tree).write(output_fpath)
    f.close()


# def reduce_grouped_models(grouped_models, metric):
#     """Reduce the grouped_models to only the best model for each definition"""
#     for grouped_model_name, individual_models in grouped_models.items():
#         best_model = None
#         best_metric = 0
#         for individual_model in individual_models:
#             individual_model_results = next(iter(individual_model.values()))
#             if f"val_{metric}" not in individual_model_results:
#                 continue
#             if individual_model_results[f"val_{metric}"] > best_metric:
#                 best_metric = individual_model_results[f"val_{metric}"]
#                 best_model = individual_model
#         grouped_models[grouped_model_name] = [best_model]
#     return grouped_models


# TODO only plot the best of a definition


#### Denoising


def plot_1d_denoising(
    models_results,
    models_definitions,
    generic_test_name,
    metrics=METRICS,
    visualize=False,
):
    MODELS_TO_IGNORE = {
        "manproc": ["bayer (extra pairs)", "LinRGB (extra pairs)", "BM3D (LinRGB)"],
        "manproc_bostitch": ["BM3D (LinRGB)", "bayer (more channels)"],
    }
    """Plot the noise level for denoising models (1D; no bitrate or input noise, just different tests and one bar shown per model)
    Not used in the paper
    """

    def remove_strictly_worse_models(
        models_data: list[tuple[str, float, str, bool]], metric: str
    ):
        """Remove strictly worse models from the models_data"""
        models_data = [model_data for model_data in models_data if not model_data[3]]
        new_models_data = []
        # check if any model has a worse score, if so don't keep it
        # minimize mse metric, maximize *ssim metric
        for model_name, yval, individual_model_name, strictly_worse in models_data:
            if any(
                [
                    (
                        (model_data[1] < yval and metric.lower() == "mse")
                        or (model_data[1] > yval and "ssim" in metric.lower())
                    )
                    for model_data in models_data
                    if model_data[0] == model_name
                ]
            ):
                continue
            new_models_data.append(
                (model_name, yval, individual_model_name, strictly_worse)
            )
        return new_models_data

    for metric in metrics:
        grouped_models = group_relevant_models(
            models_definitions["denoise"],
            models_results["denoise"],
            metric,
            reduce=True,
        )

        # for test in tests:
        # if test == 'manproc':
        #     breakpoint()
        minval = 1.0
        maxval = 0.0
        models_data = []  # List to store (model_name, y_value) tuples
        passthrough_val = None

        tooltips_coordinates_to_name = {}  # for tooltip
        csv_results = []
        for grouped_model_name, individual_models in grouped_models.items():
            if grouped_model_name in MODELS_TO_IGNORE[generic_test_name]:
                print(
                    f"plot_1d_denoising: skipping {grouped_model_name=} for {generic_test_name=}"
                )
                continue
            for individual_model in individual_models:
                # if 'ssim' in metric and 'manproc' in test:
                #     breakpoint()
                try:
                    individual_model_results = next(iter(individual_model.values()))
                except IndexError as e:
                    print(f"{e} with {grouped_model_name=}")
                    continue

                # if f"{test}_{metric}" not in individual_model_results:
                if f"{generic_test_name}_{metric}" not in individual_model_results:
                    print(f"No {generic_test_name=} for {grouped_model_name=}")
                    continue
                yval = individual_model_results[f"{generic_test_name}_{metric}"]
                if grouped_model_name == "passthrough":
                    passthrough_val = yval
                    continue

                models_data.append(
                    (
                        grouped_model_name,
                        yval,
                        next(iter(individual_model.keys())),
                        individual_model_results.get("strictly_worse", False),
                    )
                )

                if yval < minval:
                    minval = yval
                if yval > maxval:
                    maxval = yval
                # print(f"{grouped_model_name=}, {yval=}")
                # tooltips_coordinates_to_name[0, yval] = next(
                #     iter(individual_model[0].keys())
                # )  # for tooltip
        # Sort the model_data based on y-values
        models_data.sort(key=lambda x: x[1])
        models_data = remove_strictly_worse_models(models_data, metric)
        utilities.sort_dictionary(tooltips_coordinates_to_name)  # isn't this empty?
        bars_and_individual_names = []
        for model_name, yval, individual_model_name, strictly_worse in models_data:
            if not strictly_worse:
                abar = plt.bar(
                    model_name,
                    yval,
                    label=model_name,
                    color=COLORS.get(model_name),
                )[0]
                print(f"{model_name=}, {yval=}")
                # plot_tooltip_points(
                #     tooltips_coordinates_to_name
                # )  # tooltip  # FIXME need x-y + area
                bars_and_individual_names.append((abar, individual_model_name))
            csv_results.append(
                (model_name, individual_model_name, yval, strictly_worse)
            )

        # check if there is anything to plot
        if len(plt.gca().patches) == 0:
            print(f"No {generic_test_name} done with {metric=}")
            continue

        # scale plot to minimum values - 1
        plt.gca().set_ylim(
            [max(0, minval * 0.95 if minval > 0.5 else 0), min(maxval * 1.03, 1.0)]
        )
        plt.xticks([])
        # ax = plt.gca()
        # ax.set_xticks(
        #     [
        #         bar[0].get_x() + bar[0].get_width() / 2
        #         for bar in bars_and_individual_names
        #     ]
        # )
        # add individual model labels for future interactivity
        for i, (bar, individual_model_name) in enumerate(bars_and_individual_names):
            print(f"{individual_model_name=}, {bar.get_xy()}")
            annotate = plt.annotate(
                individual_model_name,
                xy=bar.get_xy(),
                xytext=(0, 0),  # [
                #    max(0, minval * 0.95 if minval > 0.5 else 0),
                #    min(maxval * 1.05, 1.0),
                # ],  # bar.get_xy(),  # (0, 0),
                textcoords="offset points",
                ha="center",
                color="w",
                fontsize=8,
                bbox=dict(
                    boxstyle="round, pad=.5",
                    fc=(0.1, 0.1, 0.1, 0.92),
                    ec=(1.0, 1.0, 1.0),
                    lw=1,
                    zorder=1,
                ),
            )
            bar.set_gid(f"mypatch_{i:03d}")
            annotate.set_gid(f"mytooltip_{i:03d}")
            # plt.text(
            #     bar.get_x() + bar.get_width() / 2,
            #     bar.get_height() / 2,
            #     individual_model_name,
            #     ha="center",
            #     va="center",
            #     rotation=90,
            #     color="black",
            # )

            # breakpoint()
            # print(f"{individual_model_name=}")

        # plt.title(f"{test=} with {metric=}")
        # if passthrough_val is not None:
        #     # plt.axhline(passthrough_val, label="passthrough", linestyle="--", color="gray")
        #     plt.title(f"{test=} with {metric=} (mean input: {passthrough_val:.2f})")
        # show y labels

        plt.gca().set_ylabel(
            metric.replace("msssim", "Mean output quality (MS-SSIM) →")
        )
        # legend without border or background
        plt.legend(ncols=2, frameon=False)
        plt.tight_layout(rect=(0, 0.01, 1, 1))
        # add a grid and make it a dashed line
        plt.grid(True, linestyle="--", linewidth=0.5, which="both")
        if visualize:
            plt.show()
        # breakpoint()
        add_svg_tooltips(  # add interactivity
            len(bars_and_individual_names),
            f"plots/1d_denoising_{generic_test_name}_{metric}.svg",
        )
        # plt.savefig(f"plots/1d_denoising_{test}_{metric}.svg")
        plt.clf()
        # save csv
        csv_header = "model_grouped", "model_individual", metric, "strictly_worse"
        with open(f"plots/1d_denoising_{generic_test_name}_{metric}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(csv_results)


def plot_progressive_denoising_curve(
    models_results,
    models_definitions,
    generic_test_name: str,
    metrics=METRICS,
    visualize=False,
):
    """Plot output noise as a function of input noise
    Used twice in the paper
    """
    use_subplots = "manproc" in generic_test_name
    print(f"plot_progressive_denoising_curve: {generic_test_name}")
    UNWANTED_MODELS = ["BM3D (LinRGB)"]  # ["bayer_extrapairs", "prgb_extrapairs"]
    if generic_test_name == "test_manproc_rawnind":
        UNWANTED_MODELS.append("LinRGB (extra pairs)")
        UNWANTED_MODELS.append("bayer (extra pairs)")
    elif generic_test_name == "test_manproc_bostitch_rawnind":
        UNWANTED_MODELS.append("bayer (more channels)")
    # X_LIMITS: tuple[float, float] = (0.4, 1.01)
    # Y_RANGE: tuple[float, float] = (0.993, 1)

    TESTS_WANTED = (
        # f"progressive_{generic_test_name}_msssim_le_0.85_msssim",
        # f"progressive_{generic_test_name}_msssim_le_0.9_msssim",
        f"progressive_{generic_test_name}_msssim_le_0.97_msssim",
        f"progressive_{generic_test_name}_msssim_le_0.99_msssim",
        f"progressive_{generic_test_name}_msssim_le_0.9975_msssim",
        # f"progressive_{generic_test_name}_msssim_ge_0.1_msssim",
        # f"progressive_{generic_test_name}_msssim_ge_0.3_msssim",
        # f"progressive_{generic_test_name}_msssim_ge_0.4_msssim",
        f"progressive_{generic_test_name}_msssim_ge_0.5_msssim",
        # f"progressive_{generic_test_name}_msssim_ge_0.8_msssim",
        f"progressive_{generic_test_name}_msssim_ge_0.9_msssim",
        f"progressive_{generic_test_name}_msssim_ge_0.99_msssim",
        # f"progressive_{generic_test_name}_msssim_ge_1.0_msssim",
    )
    for metric in metrics:
        if use_subplots:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [5, 1]}
            )
            if "bostitch" in generic_test_name:
                YLIM1 = 0.845, 0.928
                YLIM2 = 0.774, 0.842  # 4-points
                YLIM2 = 0.695, 0.842
                XLIM = 0.448, 0.9
                ax1.set_xlim(XLIM)
                ax2.set_xlim(XLIM)
            else:
                YLIM1 = 0.932, 0.965
                YLIM1 = 0.942, 0.965
                YLIM2 = 0.816, 0.976
                # YLIM1 = 0.917, 0.976
                # YLIM2 = 0.816, 0.917
                XLIM = None
                # ax1.set_xlim(XLIM)
                # ax2.set_xlim(XLIM)

            ax1.set_ylim(YLIM1)
            ax2.set_ylim(YLIM2)
            figs2plot = [ax1, ax2]
        else:
            figs2plot = [plt]
        csv_results = []
        grouped_models = group_relevant_models(
            models_definitions["denoise"], models_results["denoise"], metric
        )
        # first get baseline (input noise)
        passthrough_model_results = next(
            iter(grouped_models["passthrough"][0].values())
        )
        test_x_values = {}
        for test_name, test_result in passthrough_model_results.items():
            if not test_name.startswith(
                f"progressive_{generic_test_name}_msssim_"
            ) or not test_name.endswith(metric):
                continue
            test_x_values[test_name] = test_result
        # for test_name, test_results in
        tooltips_coordinates_to_name = {}  # for tooltip
        for grouped_model_name, individual_models in grouped_models.items():
            if grouped_model_name.startswith("passthrough"):
                continue
            x_values = []
            y_values = []
            if not individual_models:
                continue
            # breakpoint()
            for individual_model in individual_models:
                individual_model_results = next(iter(individual_model.values()))
                for test_name, test_result in individual_model_results.items():
                    if not test_name.startswith(
                        f"progressive_{generic_test_name}_msssim_"
                    ) or not test_name.endswith(metric):
                        # print(test_name)
                        # breakpoint()
                        continue
                    # if "_ge_" in test_name:
                    #     operator_str = "ge"
                    # elif "_le_" in test_name:
                    #     operator_str = "le"
                    # else:
                    #     raise ValueError(f"Unknown operator in {test_name=}")
                    # x_values.append(MINMAX_TO_MEAN_MSSSIM[operator_str][float(test_name.split("_")[4])])
                    try:
                        new_x_value = test_x_values[test_name]
                        if test_name in TESTS_WANTED:
                            x_values.append(new_x_value)
                            y_values.append(test_result)
                            tooltips_coordinates_to_name[new_x_value, test_result] = (
                                next(iter(individual_model.keys()))
                            )  # for tooltip

                            csv_results.append(
                                (
                                    grouped_model_name,
                                    next(iter(individual_model.keys())),
                                    new_x_value,
                                    test_result,
                                    test_name,
                                )
                            )
                        else:
                            print(
                                f"Skipping {test_name=}, not wanted with {generic_test_name}"
                            )

                    except KeyError as e:
                        print(
                            f"KeyError: {e}. {test_x_values=}, {individual_model.keys()=}"
                        )
                        if test_name in TESTS_WANTED:
                            breakpoint()

                if not y_values:
                    continue
            if len(x_values) == 0:
                print(f"No {metric=} for {grouped_model_name=}")
                continue
            x_values, y_values = zip(*sorted(zip(x_values, y_values)))
            # remove strictly worse models. a model i is strictly worse than j if y_value[i] < y_value[j] and x_value[i] >= x_value[j]
            if metric == "msssim":
                new_x_values = []
                new_y_values = []
                for i, (xval, yval) in enumerate(zip(x_values, y_values)):
                    if yval < 1.0 and any(
                        [
                            (x_values[j] == xval and y_values[j] > yval)
                            for j in range(len(x_values))
                        ]
                    ):
                        continue
                    new_x_values.append(xval)
                    new_y_values.append(yval)
                x_values = new_x_values
                y_values = new_y_values
            if grouped_model_name not in UNWANTED_MODELS:
                print(f"{grouped_model_name=}, {x_values=}")
                for fig2plot in figs2plot:
                    fig2plot.plot(
                        x_values,
                        y_values,
                        label=grouped_model_name,
                        marker=MARKERS.get(grouped_model_name, "o"),
                        color=COLORS.get(grouped_model_name, "dimgray"),
                        linewidth=LINEWIDTH,
                    )
            plot_tooltip_points(tooltips_coordinates_to_name)  # tooltip
        # add baseline
        # plt.plot(MINMAX_TO_MEAN_MSSSIM['ge'].values(), MINMAX_TO_MEAN_MSSSIM['ge'].values(), label="baseline", linestyle="--", color="gray")
        # plt.plot(MINMAX_TO_MEAN_MSSSIM['le'].values(), MINMAX_TO_MEAN_MSSSIM['le'].values(), label="baseline", linestyle="--", color="gray")
        # title = f"Progressive denoising curve ({metric=})"
        ylabel = f"Mean output {(metric.upper()).replace('MSSSIM', 'MS-SSIM')} →"
        if use_subplots:
            # fig.suptitle(title)
            ax1.grid(True, which="both")
            ax2.grid(True, which="both")
            ax1.legend(prop={"size": 7})
            ax1.set_ylabel(ylabel)
            # Hide spines between axes
            ax1.spines["bottom"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax1.tick_params(axis="x", bottom=False)  # don't put tick labels at the top
            fig.subplots_adjust(hspace=0.05)
            ax1.axhline(y=YLIM1[0], color="gray", linestyle="--", linewidth=0.5)
            ax2.axhline(y=YLIM2[1], color="gray", linestyle="--", linewidth=0.5)
            fig.tight_layout(rect=(0, 0.01, 1, 1))

        else:
            # plt.title(f"Progressive denoising curve ({metric=})")
            # show x-y labels

            # plt.gca().set_xlim(X_LIMITS)
            # plt.gca().set_ylim(Y_RANGE)
            # display gridline
            plt.gca().set_ylabel(ylabel)
            plt.grid(True)
            plt.legend()

            # don't show x ticks
            # plt.gca().axes.xaxis.set_ticks([])
            # save plot
        plt.gca().set_xlabel(
            f"Mean input {(metric.upper()).replace('MSSSIM', 'MS-SSIM')} →"
        )

        if visualize:
            plt.show()
        plt.savefig(
            f"plots/progressive_{generic_test_name}_denoising_curve_{metric}.svg"
        )
        plt.clf()
        # save csv
        csv_header = (
            "model_grouped",
            "model_individual",
            "mean_input",
            metric,
            test_name,
        )
        with open(
            f"plots/progressive_{generic_test_name}_denoising_curve_{metric}.csv", "w"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(csv_results)


if __name__ == "__main__":
    # get model arguments
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--tests", nargs="+", default=TESTS, choices=TESTS
    )  # ,  "validate"])
    parser.add_argument(
        "--model_types", nargs="+", default=MODEL_TYPES, choices=MODEL_TYPES
    )
    parser.add_argument(
        "--model_input", nargs="+", default=MODEL_INPUTS, choices=MODEL_INPUTS
    )
    parser.add_argument("--metrics", nargs="+", default=METRICS, choices=METRICS)
    parser.add_argument("--max_bitrate", type=float, default=MAX_BITRATE)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--strictly_worse_models_cfg", default="config/strictly_worse_models.yaml"
    )
    args = parser.parse_args()
    # get models and results
    strictly_worse_models: list[str] = utilities.load_yaml(
        args.strictly_worse_models_cfg, error_on_404=True
    )
    models_results = load_models_results(
        model_types=args.model_types, strictly_worse_models=strictly_worse_models
    )
    models_definitions = get_models_definitions(model_types=args.model_types)
    # plot
    os.makedirs("plots", exist_ok=True)

    for paper in LITERATURE.keys():
        plot_rd_curves(
            models_results,
            models_definitions,
            tests=args.tests,
            metrics=args.metrics,
            visualize=args.visualize,
            max_bitrate=args.max_bitrate,
            paper=paper,
        )
    for generic_test_name in DENOISING_TESTNAMES:
        plot_1d_denoising(
            models_results,
            models_definitions,
            generic_test_name=generic_test_name,
            visualize=args.visualize,
        )
    for generic_test_name in PROGRESSIVE_DENOISING_TESTNAMES:
        plot_progressive_denoising_curve(
            models_results,
            models_definitions,
            generic_test_name=generic_test_name,
            metrics=METRICS,
            visualize=args.visualize,
        )
