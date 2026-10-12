# MedJiT

MedJiT is a PyTorch project for generating contrast-enhanced CT angiography
(CTA) slices from non-contrast CT (NCCT) slices. It uses a conditional JiT
transformer with flow-matching training and supports paired NIfTI volumes or
preprocessed PNG slices.

## Features

- Paired NCCT-to-CTA image translation
- NIfTI preprocessing and patient-level train/validation splitting
- Optional region masks
- Distributed training with exponential moving averages
- Inference metrics: PSNR, SSIM, NRMSE, MAE, and LPIPS

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yaml
conda activate jit
```

Alternatively, install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation

Convert paired `plain` and `enhance` NIfTI volumes into PNG slices:

```bash
python preprocess_new.py --src /path/to/source --out /path/to/ct2d_splits_512
```

The generated dataset contains `train` and `val` directories with paired
`plain_2d` and `enhance_2d` patient folders.

## Training

Set the dataset and output paths, then run:

```bash
SPLIT_ROOT=/path/to/ct2d_splits_512 \
OUTPUT_DIR=/path/to/output \
bash train_ct2d_preprocessed_512.sh
```

## Inference

```bash
python inference.py \
  --non_contrast_dir /path/to/val/plain_2d \
  --contrast_dir /path/to/val/enhance_2d \
  --ckpt_dir /path/to/checkpoint-last.pth \
  --save_root_dir /path/to/results
```

## License

This project is released under the MIT License.
