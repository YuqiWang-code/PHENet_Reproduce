## Abstract
<sub>Generating high-precision normalized digital surface models (nDSMs) from a single remote sensing image remains a challenging and ill-posed problem due to the absence of reliable geometric constraints. In this work, we show that monocular depth provides structurally stable cues of local geometry but lacks the global scale and vertical reference required for absolute height recovery. This intrinsic mismatch limits direct depth-to-height regression, particularly when transferring across heterogeneous terrains, land-cover compositions, and imaging conditions. Building on this idea, we propose the Relative Depth–Absolute Height Prediction Network (RDAH-Net), a framework that exploits relative depth as a geometry-aware prior while learning terrain-dependent height mappings from image appearance to absolute height. As the backbone, we employ a lightweight MobileNetV2 enhanced with a Convolutional Block Attention Module (CBAM), and further incorporate a cross-modal bidirectional attention fusion scheme with positional encoding to achieve a deep and effective fusion of image appearance and depth prior cues. Finally, a PixelShuffle-based upsampling strategy is used to sharpen prediction details and mitigate typical upsampling artifacts. Extensive experiments across diverse regions demonstrate that RDAH-Net achieves robust and generalizable height estimation, providing a practical alternative for large-scale mapping and rapid update scenarios.</sub>

## Environment Setup

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## BCD-foggy Height-map Inference

RDAH-Net requires both the RGB image and a relative-depth prior. Following the
paper, `generate_heightmaps.py` obtains the frozen prior from Depth Anything V2,
then converts it to an nDSM-like height map with an RDAH-Net checkpoint. The
output is a single-channel uint16 PNG with the same filename as its source image.

```bash
python generate_heightmaps.py \
  --dataset-roots /storage/BCD-foggy/LEVIR-CD-foggy \
  --rdah-checkpoint /home/yqwang/project/PHENet/pre_checkpoint/checkpoints-track1/104best_model.pth \
  --device cuda:0 --amp
```

The first run downloads the configured Depth Anything V2 model from Hugging
Face. Existing output maps are skipped unless `--overwrite` is supplied.

## Model Checkpoints and Datasets available

The checkpoints trained separately by three datasets and the datasets used in the paper are provided by the author in the following link.

[Figshare website](https://doi.org/10.6084/m9.figshare.31986864)

## Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@Article{rs18071024,
AUTHOR = {Jiang, Liting and Wang, Feng and Jiao, Niangang and Zhu, Jingxing and Xiang, Yuming and You, Hongjian},
TITLE = {RDAH-Net: Bridging Relative Depth and Absolute Height for Monocular Height Estimation in Remote Sensing},
JOURNAL = {Remote Sensing},
VOLUME = {18},
YEAR = {2026},
NUMBER = {7},
ARTICLE-NUMBER = {1024},
URL = {https://www.mdpi.com/2072-4292/18/7/1024},
ISSN = {2072-4292},
DOI = {10.3390/rs18071024}
}
```
