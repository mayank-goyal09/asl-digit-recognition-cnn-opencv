# 🤟 ASL Digits Recognizer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-93%25+-10B981?style=for-the-badge)

**Deep Learning Powered American Sign Language Digit Recognition**

[Live Demo](#deployment) • [Features](#features) • [Installation](#installation) • [Usage](#usage)

</div>

---

## 🎯 Overview

A beautiful, interactive web application that uses a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) digits (0-9) from images. Built with TensorFlow and Streamlit, featuring a premium glassmorphism UI design.

## ✨ Features

- 🧠 **CNN Neural Network** - Deep learning model with 93%+ accuracy
- 📤 **Image Upload** - Upload any ASL digit image
- ✏️ **Canvas Drawing** - Draw ASL gestures directly in browser
- 🎲 **Sample Images** - Test with built-in sample images
- 📊 **Confidence Scores** - See prediction probabilities for all classes
- 🎨 **Premium UI** - Beautiful glassmorphism design with animations
- ⚡ **Real-time Prediction** - Instant results

## 🏗️ Model Architecture

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D (32 filters) | (64, 64, 32) | 320 |
| MaxPooling2D | (32, 32, 32) | 0 |
| Conv2D (64 filters) | (32, 32, 64) | 18,496 |
| MaxPooling2D | (16, 16, 64) | 0 |
| Conv2D (128 filters) | (16, 16, 128) | 73,856 |
| MaxPooling2D | (8, 8, 128) | 0 |
| Flatten | (8192) | 0 |
| Dense (128) | (128) | 1,048,704 |
| Dropout (0.3) | (128) | 0 |
| Dense (10) | (10) | 1,290 |

**Total Parameters:** 1,142,666

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/asl-digits-recognizer.git
   cd asl-digits-recognizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   ```
   http://localhost:8501
   ```

## 📱 Usage

1. **Choose Input Method**: Select from upload, draw, or sample images
2. **Provide Image**: Upload an ASL digit image or draw on canvas
3. **Click Predict**: Hit the prediction button
4. **View Results**: See the detected digit and confidence scores

## 🌐 Deployment

### Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select the main file (`app.py`)
5. Click Deploy!

### Deploy on Other Platforms

<details>
<summary>Heroku</summary>

1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. Deploy using Heroku CLI
</details>

<details>
<summary>Railway</summary>

1. Create a new project on Railway
2. Connect your GitHub repository
3. Railway will auto-detect Streamlit
4. Deploy!
</details>

## 📂 Project Structure

```
asl-digits-recognizer/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── X.npy                     # Image dataset
├── Y.npy                     # Labels dataset
├── main.ipynb               # Training notebook
└── models/
    ├── asl_digits_cnn.keras # Base CNN model
    └── asl_digits_aug.keras # Augmented model
```

## 📊 Dataset

- **Images**: 2,062 grayscale images (64×64 pixels)
- **Classes**: 10 (digits 0-9)
- **Split**: 70% train / 15% validation / 15% test
- **Source**: Sign Language Digits Dataset

## 🎨 UI Preview

The application features:
- Gradient backgrounds with purple/indigo theme
- Glassmorphism card effects
- Animated prediction display
- Interactive confidence bars
- Responsive design

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- Streamlit for the web application framework
- Sign Language Digits Dataset creators

---

<div align="center">
  <p>Built with ❤️ using TensorFlow & Streamlit</p>
  
  ⭐ Star this repository if you found it helpful!
</div>
