# Page Orientation Correction

A professional, production-ready README for the Page Orientation Correction project. This repository contains code and assets for detecting and correcting the orientation of scanned or photographed pages (rotations, flips, skew), using segmentation and model-based approaches.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start — Inference](#quick-start--inference)
- [Training](#training)
- [Evaluation and visualization](#evaluation-and-visualization)
- [Examples](#examples)
- [Tips for best results](#tips-for-best-results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Page Orientation Correction is a lightweight toolkit for detecting and correcting the orientation of document pages in images. It is useful for preprocessing scanned documents, mobile phone photos of pages, or any image-based document pipeline where consistent upright orientation is required.

The project contains model code, training scripts, test images and example outputs. The approach in this repository uses segmentation and/or deep models to identify page edges, text direction, and the correct rotation to apply.

## Features

- Detect and correct page rotation (multiples of 90°) and flipped pages
- Batch processing support for folders of images
- Training code and pretrained model integration (see TRAINING_CODE and deeplabv3_ folders)
- Visualization of predictions (viz_outputs)

## Repository structure

- codes/ — Miscellaneous scripts and utilities (inference, preprocessing, helpers). Inspect this folder for runnable scripts.
- deeplabv3_/ — Model code, loss functions, and model wrappers (semantic segmentation model implementations).
- TRAINING_CODE/ — Training configuration, dataset handling, and training scripts.
- test_img/ — Example inputs used for quick testing.
- output_images1/ — Example model outputs or intermediate results
- viz_outputs/ — Visualization examples showing inputs and corrected outputs.
- requirements.txt — Python package requirements used for this project.

Note: Filenames and entry-point scripts may be present inside the folders above. If you prefer a single entry point, create a small wrapper script that imports the provided modules to run inference/training consistently.

## Requirements

This project targets Python 3.8+ and common ML libraries. Install the exact versions listed in requirements.txt for best reproducibility.

Example (recommended to use a virtual environment):

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate     # Windows (PowerShell)
pip install --upgrade pip
pip install -r requirements.txt
```

If you use conda:

```bash
conda create -n page-orient python=3.8 -y
conda activate page-orient
pip install -r requirements.txt
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/aroob79/page_orientation_correction_.git
cd page_orientation_correction_
# optional: checkout the v2_dev branch if you work on that branch
git checkout v2_dev || true
pip install -r requirements.txt
```

If you plan to train models from source, make sure you have a CUDA-enabled machine and appropriate versions of torch + torchvision installed (check requirements.txt).

## Quick start — Inference

There may be multiple inference scripts provided in the `codes/` or `deeplabv3_/` folders. The following example shows the typical usage pattern (replace script name with the actual one in the repo):

```bash
# Single image inference (example)
python codes/infer.py --input test_img/sample.jpg --output viz_outputs/output.jpg --model checkpoints/best_model.pth

# Batch inference on a folder
python codes/infer.py --input test_img/ --output viz_outputs/ --model checkpoints/best_model.pth --batch-size 8
```

If the repository provides a function-based API, you can run inference from Python:

```python
from codes import inference_utils
pred = inference_utils.predict_image('test_img/sample.jpg', model_path='checkpoints/best_model.pth')
# pred should contain corrected image or rotation metadata
```

If scripts have different CLI options, run them with `-h` to get exact argument names:

```bash
python codes/infer.py -h
```

## Training

Training code and experiment configs are in the `TRAINING_CODE/` folder. A high-level training procedure:

1. Prepare dataset in the expected format (images + labels/masks). Check `TRAINING_CODE/README` or scripts for dataset format. Typical structure:

```
dataset/
  images/
  masks/  # optional, segmentation masks if used
  labels.csv  # optional, rotation labels
```

2. Configure hyperparameters and model path in the training config or script.
3. Run the training script (example — replace with actual training script name):

```bash
python TRAINING_CODE/train.py --config TRAINING_CODE/config.yaml --output-dir checkpoints/
```

4. Monitor training logs and validation metrics. Save the best checkpoint for inference.

If the training folder includes Jupyter notebooks, open them for guided experiments and visualizations.

## Evaluation and visualization

Use the evaluation scripts inside `TRAINING_CODE/` or `codes/` to compute metrics (accuracy of rotation detection, IoU for segmentation, etc.). Example:

```bash
python codes/eval.py --predictions viz_outputs/ --ground-truth test_labels/ --metrics accuracy,confusion
```

Visualization utilities are included in `viz_outputs/`. They show side-by-side input, predicted masks/rotations, and corrected images.

## Examples

- Correct a single image and save the rotated result:

```bash
python codes/infer.py --input test_img/IMG_0001.jpg --output viz_outputs/IMG_0001_corrected.jpg --model checkpoints/best_model.pth
```

- Run on a folder and visualize results:

```bash
python codes/batch_infer.py --input test_img/ --output viz_outputs/ --model checkpoints/best_model.pth
```

Replace script names above with the actual filenames provided in this repository.

## Tips for best results

- Use high-resolution images: the model performs better when the page boundary and text orientation are clearly visible.
- For photographed pages, try to crop out surrounding background and ensure the page occupies most of the image.
- If the model misclassifies heavily skewed pages, add additional augmentation (rotation, perspective transform) to the training data.
- Fine-tune the model on data captured from your target device/camera for best performance.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes, add tests if applicable
4. Open a pull request with a clear description of changes

Please follow standard best practices: clear commit messages, small focused PRs, and include documentation for new features.

## License

This repository does not include an explicit license. If you plan to publish or share, add a LICENSE file (e.g., MIT, Apache-2.0) to clarify the terms.

## Contact

Maintainer: aroob79

For questions, feature requests or bug reports, open an issue on GitHub.