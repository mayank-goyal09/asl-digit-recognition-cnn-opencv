# 🤟 ASL Digits Recognizer

<div align="center">

![ASL CNN Banner](assets/banner.png)

<br>

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://asl-digit-recognition-cnn-opencv-project.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github)](https://github.com/mayank-goyal09)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mayank-goyal-mg09/)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

**A high-performance deep learning application that translates American Sign Language digits into text in real-time.**

[🔴 Live Demo](https://asl-digit-recognition-cnn-opencv-project.streamlit.app/) • [✨ Features](#-features) • [🧠 How it Works](#--how-it-works) • [🚀 Installation](#--installation)

</div>

---

## 📸 Live Preview

<div align="center">
  <a href="https://asl-digit-recognition-cnn-opencv-project.streamlit.app/">
    <img src="assets/app_preview.png" alt="ASL App Preview" width="100%" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
  </a>
  <p><em>Click the image to try the live app!</em></p>
</div>

## 💡 Overview

The **ASL Digits Recognizer** uses a custom-trained **Convolutional Neural Network (CNN)** to classify hand gestures representing digits 0-9. Built with **TensorFlow** and deployed via **Streamlit**, it features a premium glassmorphism UI and instantaneous inference capabilities.

Whether you upload an image, draw on the digital canvas, or select a sample, the AI analyzes the visual patterns to deliver high-confidence predictions.

## ✨ Features

- **🧠 Advanced CNN Engine**: Powered by a multi-layer Convolutional Neural Network trained on the Sign Language Digits Dataset.
- **🎨 Interactive Canvas**: Draw digits directly in the browser with real-time feedback.
- **📤 Multi-Input Support**: Upload images, draw freely, or use pre-loaded validation samples.
- **📊 Live Confidence Metrics**: Visualizes the model's certainty across all 10 classes dynamically.
- **💎 Premium Aesthetics**: Modern, dark-themed UI with glassmorphism effects and smooth animations.

## 🧠 How it Works

The core of this application is a **Convolutional Neural Network**, designed to mimic the human visual cortex. It processes input images through layers of filters to extract features like edges, curves, and textures.

<div align="center">
  <img src="https://raw.githubusercontent.com/jerpint/cnn-cheatsheet/master/assets/valid.gif" alt="CNN Animation" width="600">
  <p><em>Visualization of a Convolution Operation (Credit: <a href="https://github.com/jerpint/cnn-cheatsheet">cnn-cheatsheet</a>)</em></p>
</div>

1.  **Input Processing**: Images are resized to 64x64 pixels and converted to grayscale.
2.  **Feature Extraction**: Three convolutional layers (32, 64, 128 filters) scan the image for key patterns.
3.  **Classification**: Dense layers interpret these features to determine the most likely digit.

### Model Architecture
| Layer Type | Specifications |
| :--- | :--- |
| **Input** | (64, 64, 1) Grayscale Image |
| **Conv2D** | 32 filters, 3x3 kernel, ReLU |
| **MaxPooling** | 2x2 pool size |
| **Conv2D** | 64 filters, 3x3 kernel, ReLU |
| **Conv2D** | 128 filters, 3x3 kernel, ReLU |
| **Dense** | 128 neurons, Dropout (0.3) |
| **Output** | 10 neurons (Softmax) |

## 🚀 Installation

Run the app locally in minutes:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/mayank-goyal09/asl-digits-recognizer.git
    cd asl-digits-recognizer
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the App**
    ```bash
    streamlit run app.py
    ```

## 🤝 Connect with Me

<div align="center">

| **Mayank Goyal** |
| :---: |
| [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mayank-goyal-mg09/) |
| [![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/mayank-goyal09) |
| [� More Projects](https://github.com/mayank-goyal09?tab=repositories) |

</div>

---

<div align="center">
  <p>Built with ❤️ and ☕ by Mayank Goyal</p>
  <p>
    <a href="#">Back to Top ⬆️</a>
  </p>
</div>
