## Source Code for "Learning Joint Denoising, Demosaicing, and Compression from the Raw Natural Image Noise Dataset"

This repository contains the source code used in the article "Learning Joint Denoising, Demosaicing, and Compression from the Raw Natural Image Noise Dataset".

This code is delivered as it was used throughout the years of research, in an "academic" state (as-is). There has not been an effort to clean up the code and make it more presentable.

For more detailed information about the project, please see the main README file located at [src/rawnind/README.md](src/rawnind/README.md).

A simpler, cleaner public repository focusing on denoising-only is planned for the relatively near future.

### Installation

#### Basic Installation
```bash
pip install -e .
```

### Downloading the RawNIND Dataset

If you just want to download the RawNIND dataset, you can use the following command:

```bash
curl -s "https://dataverse.uclouvain.be/api/datasets/:persistentId/?persistentId=doi:10.14428/DVN/DEQCIM" | jq -r '.data.latestVersion.files[] | "wget -c -O \"\(.dataFile.filename)\" https://dataverse.uclouvain.be/api/access/datafile/\(.dataFile.id)"' | bash
```

### Pre-trained models

Pre-trained models are available through the following link: https://drive.google.com/drive/folders/12Uc5sT4OWx02sviUS_boDk1zDSDRVorw?usp=sharing

### Citation

You can cite the "Learning Joint Denoising, Demosaicing, and Compression from the Raw Natural Image Noise Dataset" paper as follows:

```bibtex
@misc{brummer2025learningjointdenoisingdemosaicing,
	  title={Learning Joint Denoising, Demosaicing, and Compression from the Raw Natural Image Noise Dataset},
	  author={Benoit Brummer and Christophe De Vleeschouwer},
	  year={2025},
	  eprint={2501.08924},
	  archivePrefix={arXiv},
	  primaryClass={cs.CV},
	  url={https://arxiv.org/abs/2501.08924},
}
```
