import logging
import os
import time
import statistics
import sys

sys.path.append("..")
from rawnind.libs import rawds
from rawnind.libs import rawproc
from rawnind.libs import abstract_trainer

LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")


def test_train_images_load_time(dataset_class: rawds.RawImageDataset, **kwargs):
    """Measure the load time for each image in the dataset."""
    dataset = dataset_class(**kwargs)
    start_time = time.time()
    timings = []
    for i, image in enumerate(dataset):
        fetch_time = time.time() - start_time
        timings.append(fetch_time)
        fpaths = {}
        if isinstance(dataset, rawds.CleanProfiledRGBCleanBayerImageDataset):
            fpaths["x"], fpaths["y"] = dataset.ds_xy_fpaths[i]
        elif isinstance(dataset, rawds.CleanProfiledRGBCleanProfiledRGBImageDataset):
            fpaths["x"] = dataset.ds_fpaths[i]
        elif isinstance(dataset, rawds.CleanProfiledRGBNoisyBayerImageDataset):
            fpaths["x"] = dataset.dataset[i]["gt_linrec2020_fpath"]
            fpaths["y"] = dataset.dataset[i]["f_bayer_fpath"]
        elif isinstance(dataset, rawds.CleanProfiledRGBNoisyProfiledRGBImageDataset):
            fpaths["x"] = dataset.dataset[i]["gt_linrec2020_fpath"]
            fpaths["y"] = dataset.dataset[i]["f_linrec2020_fpath"]

        logging.info(f"Image(s) {fpaths} fetch time: {fetch_time:.2f} seconds. ")
        start_time = time.time()
    return {"min": min(timings), "avg": statistics.mean(timings), "max": max(timings)}


if __name__ == "__main__":
    # the usual logging init
    logging.basicConfig(
        filename=LOG_FPATH,
        format="%(message)s",
        level=logging.INFO,
        filemode="w",
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"# python {' '.join(sys.argv)}")

    logging.info("Testing CleanProfiledRGBNoisyBayerImageDataset runtimes...")
    timings = test_train_images_load_time(
        rawds.CleanProfiledRGBNoisyBayerImageDataset,
        num_crops=16,
        content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
        crop_size=256,
        test_reserve=[],
    )
    logging.info(f"CleanProfiledRGBNoisyBayerImageDataset timing: {timings}")

    logging.info("Testing CleanProfiledRGBNoisyProfiledRGBImageDataset runtimes...")
    timings = test_train_images_load_time(
        rawds.CleanProfiledRGBNoisyProfiledRGBImageDataset,
        num_crops=8,
        content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
        crop_size=256,
        test_reserve=[],
    )
    logging.info(f"CleanProfiledRGBNoisyProfiledRGBImageDataset timing: {timings}")

    logging.info("Testing CleanProfiledRGBCleanBayerImageDataset runtimes...")
    timings = test_train_images_load_time(
        rawds.CleanProfiledRGBCleanBayerImageDataset,
        num_crops=16,
        data_dpath=abstract_trainer.EXTRARAW_DATA_DPATH,
        crop_size=256,
    )
    logging.info(f"CleanProfiledRGBCleanBayerImageDataset timing: {timings}")

    logging.info("Testing CleanProfiledRGBCleanProfiledRGBImageDataset runtimes...")
    timings = test_train_images_load_time(
        rawds.CleanProfiledRGBCleanProfiledRGBImageDataset,
        num_crops=8,
        data_dpath=abstract_trainer.EXTRARAW_DATA_DPATH,
        crop_size=256,
    )
    logging.info(f"CleanProfiledRGBCleanProfiledRGBImageDataset timing: {timings}")
