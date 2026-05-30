# Pixel-Domain vs. Frequency-Domain (2D-DCT) CNNs for Pneumonia Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Handcull/pneumonia-dct-cnn/blob/main/notebooks/pneumonia_pixel_vs_dct.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A controlled comparison of two image representations — **raw pixels** vs. **2D Discrete Cosine Transform (DCT) frequency coefficients** — fed to *identical* convolutional neural networks for detecting **pneumonia in chest X-rays**.

The methodology deliberately mirrors my undergraduate thesis on *pixel- vs. frequency-domain (2D-DCT) features for StyleGAN deepfake detection*, transferred here to a **medical-imaging** domain. The research question is the same: **does moving to the frequency domain help a CNN, and where does the model actually look?**

> ⚠️ **Disclaimer:** This is a research/educational project. It is **not** a medical device and must not be used for clinical diagnosis.

---

## TL;DR

- 🧠 **Two identical CNNs, two input representations** — the only variable is *pixel* vs *2D-DCT*, so the comparison is fair.
- ☁️ **Runs 100% on Google Colab (free GPU)** — the dataset is **streamed from the cloud**, nothing is downloaded to your local machine.
- 📊 **Full evaluation** — accuracy, precision, **recall** (critical in medicine), F1, ROC-AUC, and confusion matrices.
- 🔍 **Grad-CAM interpretability** — visualizes *where* the model focuses inside the lungs (trustworthy AI).
- 🌈 **Frequency-domain visualization** — see how the DCT spectrum of *normal* vs *pneumonia* X-rays differs.

---

## Motivation

GAN- and diffusion-generated images leave tell-tale artifacts in the **frequency domain** that are hard to see in raw pixels — this is the core insight behind my deepfake-detection thesis. This project asks whether a similar idea is useful for **medical imaging**: pneumonia changes the *texture* and *opacity* of lung tissue, and texture is partly a frequency-domain phenomenon. By feeding a CNN the **2D-DCT** of each X-ray instead of the raw pixels, we test whether frequency features make the disease easier (or harder) to learn.

## Method

1. **Data** — Chest X-Ray (Pneumonia) images, streamed online into Colab (see below).
2. **Pixel branch** — grayscale X-ray → CNN.
3. **DCT branch** — compute the **2D-DCT** of each image, take the log-magnitude of the coefficients → feed this "frequency image" into a CNN with the **same architecture** as the pixel branch.
4. **Fair comparison** — identical architecture, hyper-parameters, and training budget; the *only* difference is the input representation.
5. **Evaluation & interpretability** — standard metrics + Grad-CAM heatmaps.

## Dataset

**Chest X-Ray Images (Pneumonia)** — ~5,800 labeled images (NORMAL / PNEUMONIA).

The notebook loads the data **directly from the cloud** (HuggingFace Datasets, with a Kaggle-API fallback) into the Colab runtime. **It is never downloaded to a local PC** — only this repository's text files (notebook + README) live on your machine.

## How to run (no local setup needed)

1. Click the **"Open in Colab"** badge at the top.
2. In Colab: **Runtime → Change runtime type → GPU**.
3. **Runtime → Run all.** The notebook installs its own dependencies, streams the dataset, trains both models, and renders all plots.

## Results

> _Numbers below are placeholders and will be updated after the first Colab training run._

| Model            | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------------------|:--------:|:---------:|:------:|:--:|:-------:|
| Pixel-domain CNN | _–_ | _–_ | _–_ | _–_ | _–_ |
| 2D-DCT CNN       | _–_ | _–_ | _–_ | _–_ | _–_ |

## Repository structure

```
.
├── notebooks/
│   └── pneumonia_pixel_vs_dct.ipynb   # main, self-contained Colab notebook
├── assets/                            # figures exported from the notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## Author

**Steven Jerry Gani** — B.Sc. Mathematics (Data Analytics), Universitas Katolik Parahyangan
- GitHub: [@Handcull](https://github.com/Handcull)
- Email: stevenjerrygani@gmail.com

## License

Released under the [MIT License](LICENSE).
