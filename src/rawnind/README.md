# RawNIND

## Requirements

### Arch:

* `python-pytorch-opt-rocm` and `python-torchvision-rocm` or `python-pytorch-opt-cuda` and `python-torchvision-cuda`
* `libraw python-rawpy python-openexr python-opencv python-colour-science python-pytorch-msssim-git python-configargparse python-rawpy python-pytorch-piqa-git python-tqdm python-colorspacious python-ptflops openimageio`

### Other distributions / pip:
pip3 install colour-science pytorch-msssim ConfigArgParse tqdm pypng opencv-python matplotlib piqa rawpy requests pyyaml ptflops
pip3 install https://download.pytorch.org/whl/cu111/torchvision-0.11.3%2Bcu111-cp38-cp38-linux_x86_64.whl https://download.pytorch.org/whl/cu111/torch-1.10.2%2Bcu111-cp38-cp38-linux_x86_64.whl  # latest version that matches Cuda 1.11.1, feel free to use newer versions of both; get relevant command on https://pytorch.org/
git clone https://github.com/sanguinariojoe/pip-openexr.git
cd pip-openexr
pip3 install .

Install https://github.com/jamesbowman/openexrpython (`openexr` on `pip`). If it fails you probably need to install OpenEXR manually and apply the patches used in https://aur.archlinux.org/cgit/aur.git/tree/?h=python-openexr , change setup.py to include the local include/lib directories, and run "pip install .". Finally in a local installation you will likely need to run `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib python [args]`

### Slurm:

module load GCC CUDA LibTIFF PyYAML PyTorch libpng libjpeg-turbo 

mkdir ~/tmp/cloned/python-openexr
cd ~/tmp/cloned/python-openexr
wget https://files.pythonhosted.org/packages/7c/c4/76bf884f59d3137847edf8b93aaf40f6257d8315d0064e8b1a606ad80b1b/OpenEXR-1.3.2.tar.gz
tar -xvf OpenEXR-1.3.2.tar.gz
cd OpenEXR-1.3.2/
patch -p1 <../fix1.patch 


The following fails
PYVER=3.11.1
mkdir -p tmp/cloned
wget "https://www.python.org/ftp/python/${PYVER}/Python-${PYVER}.tgz"
tar -xvf "Python-${PYVER}.tgz"
cd "Python-${PYVER}"
mkdir ~/.localpython
./configure --prefix /home/ucl/elen/brummer/.localpython --enable-optimizations
make
make install


git clone https://github.com/sanguinariojoe/pip-openexr.git
cd pip-openexr
pip3 install .


## Datasets

### Gathering

RawNIND is available on https://dataverse.uclouvain.be/dataset.xhtml?persistentId=doi:10.14428/DVN/DEQCIM

It can be downloaded with the following commands: `curl -s "https://dataverse.uclouvain.be/api/datasets/:persistentId/?persistentId=doi:10.14428/DVN/DEQCIM" | jq -r '.data.latestVersion.files[] | "https://dataverse.uclouvain.be/api/access/datafile/\(.dataFile.id)"' | wget -c -i -`

This will download a flat structure. The data loaders and pre-processors in this work expect the structure described in the following subsection (datasets/src/<bayer or X-Trans>/<SET_NAME>/<"gt/" if applicable><IMAGE.EXT>).

### Prepare the RawNIND dataset (Clean-noisy / paired images)

RawNIND files are organized as follow in `../../datasets/RawNIND`:

- `src/<bayer or X-Trans>/<SET_NAME>/<IMAGEY>.<EXT>`
- `src/<bayer or X-Trans>/<SET_NAME>/gt/<IMAGEX>.<EXT>`
- `proc/lin_rec2020/<SET_NAME>/<IMAGEY.EXT>`
- `proc/lin_rec2020/<SET_NAME>/gt/<IMAGEX>.<EXT>`
- `proc/lin_rec2020/<SET_NAME>/gt/<IMAGEX>.<EXT>.xmp` (processing pipeline for testing; each test-set ground-truth image has one)
- `proc/dt/<SET_NAME>/<IMAGEY>_aligned_to_<IMAGEX>.<EXT>` (manually processed test images)
- `proc/dt/<SET_NAME>/gt/<IMAGEX>_aligned_to_<IMAGEY>.<EXT>` (Ground-truths are aligned against one another. When that is the case, we don't place the "IMAGEY" ground-truth in the "gt" directory in order to avoid namespace conflict with different proccessing pipelines.)
- `masks/<IMAGEX-IMAGEY.EXT>.png`
- `metadata/xProfiledRGB_yBayer.yaml`
- `metadata/xProfiledRGB_yProfiledRGB.yaml`

```bash
# python tools/make_hdr_rawnind_files.py  # optional: full-size images are not needed by the model
# Demosaic x-trans files and convert them to pRGB OpenEXR files with darktable-cli 
python tools/xtrans_to_openexr_dataset.py
# Pre-crop dataset images (raw and pRGB, RawNIND) for faster loading during training
python tools/crop_datasets.py --dataset rawnind
# Compute images alignment, masks, gains
python tools/prep_image_dataset.py
# python tools/prep_image_dataset.py --dataset RawNIND_Bostitch # for additional camera / test images
cp ../../datasets/RawNIND/masks_overwrite/* ../../datasets/RawNIND/masks/
```

#### Optional: compute MS-SSIM loss (for filtered testing)

```
Run the following:
`python tools/add_msssim_score_to_dataset_yaml_descriptor.py`
```

### Prepare the RawNIND manual processing test images
Generate the dataset descriptor: `python libs/rawds_manproc.py`
Compute the MS-SSIM losses (for filtered testing): `python tools/add_msssim_score_to_dataset_yaml_descriptor.py --dataset_descriptor_fpath ../../datasets/RawNIND/manproc_test_descriptor.yaml`


### Optional: prepare the additional camera train/test images
#### Req. for training
Pre-crop dataset images (raw and pRGB, RawNIND) for faster loading during training: `python tools/crop_datasets.py --dataset RawNIND_Bostitch`
Compute images alignment, masks, gains: `python tools/prep_image_dataset.py --dataset RawNIND_Bostitch`
Optional, to get the full-size debayered images: `python tools/make_hdr_rawnind_files.py --data_dpath ../../datasets/RawNIND_Bostitch`
#### Req. for testing
Make the manproc descriptor: `python libs/rawds_manproc.py --test_descriptor_fpath ../../datasets/RawNIND_Bostitch/manproc_test_descriptor.yaml --rawnind_content_fpath ../../datasets/RawNIND_Bostitch/RawNIND_Bostitch_masks_and_alignments.yaml --test_reserve_fpath config/test_reserve_extdata.yaml`
Compute MS-SSIM scores on manproc descriptor (for filtered testing): `python tools/add_msssim_score_to_dataset_yaml_descriptor.py --dataset_descriptor_fpath ../../datasets/RawNIND_Bostitch/manproc_test_descriptor.yaml`



#### Optional: prepare the external paired dataset (deprecated)

Ensure that the dataset is in the `../../datasets/ext_raw_denoise_<train/test>/src/bayer/<SET_NAME>/[gt]` directory, with noisy images in SET_NAME and ground-truths in gt.
```bash
python tools/crop_datasets.py --dataset ext_raw_denoise_test
python tools/crop_datasets.py --dataset ext_raw_denoise_train

```

### Prepare the `extraraw` dataset (Clean-clean / unpaired images) for training

**pixl-us**

Run `rsync -avL rsync://raw.pixls.us/data/ raw-pixls-us-data/` from within `<TEMPORARY_DIRECTORY>`, multiple times as needed until the resulting files take up approximately 41 GB.

then run `python tools/gather_raw_gt_images.py --orig_dpath <TEMPORARY_DIRECTORY>/raw-pixls-us-data/ --orig_name raw-pixls`

**trougnouf-ISO_LE_100 (ie your own raw images)**

Adapt orig_dpath and orig_name according to your raw pictures directory
```bash
#eg first run:
python tools/gather_raw_gt_images.py --orig_name trougnouf --orig_dpath /orb/Pictures/ITookAPicture # change path to point to your image library

#eg update:
python tools/gather_raw_gt_images.py --overwrite --orig_name trougnouf --orig_dpath '/orb/Pictures/ITookAPicture/2022/'
```

**Then for all of the above**

Once clean-clean images have all been gathered, remove the duplicate files as follow:

```bash
cd ../../datasets/extraraw
rmlint . -S l
./rmlint.sh sh:remove  # add -d or user input will be required
rm rmlint.*  # rm the rmlint files too
cd ../../src/rawnind/
```

Process all of the ground-truth images into linear rec.2020 profile (ground-truth):

```bash
python tools/make_hdr_extraraw_files.py
bash logs/make_hdr_extraraw_files.py.log  # delete any file that couldn't be read
```

and check the dataset integrity with `python tools/check_dataset.py`

```bash
# Pre-crop dataset images (raw and pRGB, extraraw) for faster loading during training
python tools/crop_datasets.py --dataset extraraw
python tools/prep_image_dataset_extraraw.py  # Generate list of crops
```


`extraraw` files are organized as follow in ``../../datasets/extraraw`:
- `<SET_NAME>/src/<bayer or X-Trans>/<IMAGE.EXT>`
- `<SET_NAME>/src/proc/lin_rec2020/<IMAGE.EXT>.exr`

and crop the extraraw dataset with `python tools/crop_datasets.py --dataset extraraw` (or without argument to crop both extraraw and RawNIND)

### Prepare the `extraraw` `playraw` (unpaired) manually processed images for testing

If not already done, generate the linear rec.2020 images (same as in the training prep) with `python tools/make_hdr_extraraw_files.py` and create the list of crops with `python tools/prep_image_dataset_extraraw.py`
Then create the manually processed dataset descriptor with `python libs/rawds_manproc.py --rawnind_content_fpath ../../datasets/extraraw/play_raw_test/crops_metadata.yaml --test_descriptor_fpath ../../datasets/extraraw/play_raw_test/manproc_test_descriptor.yaml --unpaired_images --test_reserve_fpath ""`


# Tests
Add results to TODO
TODO


# Generate plots

First test all of the models with `bash scripts/test_all_needed.sh` (it will run for days), then generate the plots with `python tests/grapher.py`

# Troubleshooting

## Installing (python-)OpenEXR

If "pip install OpenEXR" fails, try installing the patches from "python-openexr" (aur repository). Don't forget to add the local OpenEXR paths to the library and include arrays in setup.py, then "pip install ."

# Known bugs

- images converted from X-Trans are saved with the wrong fpath in the yaml dataset descriptor (eg: actual fn: DSCF1735.RAF.exr, descriptor fn: DSCF1735.exr)
