# 🧠 Brain Tumor MRI Classifier

A Streamlit app that classifies brain MRI scans as showing a tumor or not, using a ResNet50 transfer-learning model.

**Live demo:** _(add your Streamlit Cloud link here once deployed)_

## About

This is the deployment stage of a 3-part CNN learning project:
1. **Basic CNN** — built from scratch
2. **+ Data Augmentation** — Keras preprocessing layers (flip, rotation, zoom, contrast)
3. **+ Transfer Learning** — ResNet50 (ImageNet pretrained), fine-tuned on brain MRI data

The final model (this app) reached **88% test accuracy** after correcting the ResNet50 preprocessing (using `preprocess_input` instead of simple `/255.0` normalization) and fine-tuning the top layers of the pretrained backbone.

Model weights are hosted on [Hugging Face Hub](https://huggingface.co/karthikkkkl/brain-tumor-resnet50) and downloaded at runtime, rather than committed to this repo, since the `.keras` file is ~160MB.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

This is a student/learning project trained on a small dataset. It is **not** a medical diagnostic tool and should not be used for real medical decisions.

## Tech stack

- TensorFlow / Keras (ResNet50 transfer learning)
- Streamlit (web app)
- Hugging Face Hub (model hosting)
