# Image Caption Generator with Voice Output for the Visually Impaired

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

An assistive deep learning system designed for visually impaired and low-vision individuals. The system takes images (via photo upload, demo samples, or live webcam capture), extracts rich visual representations using a pre-trained **ResNet-50** Convolutional Neural Network (CNN), generates natural English captions using an **LSTM** recurrent neural network, and synthesizes clear voice descriptions using **Google Text-to-Speech (gTTS)**.

---

## 📌 Project Overview

- **Project Title:** Image Caption Generator with Voice Output for the Visually Impaired
- **Domain:** Multimodal Deep Learning, Computer Vision & Natural Language Processing
- **Dataset:** [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) (8,000 images, 40,000 reference captions)
- **Visual Encoder:** Pre-trained ResNet-50 (2048-dimensional bottleneck feature vector)
- **Language Decoder:** Long Short-Term Memory (LSTM) with 512 hidden units & 256 embedding dimension
- **Vocabulary Size:** 8,782 tokens
- **Text-to-Speech:** Google Text-to-Speech (`gTTS`) with customizable international accents (US, UK, India, Canada, Australia)
- **User Interface:** Streamlit (Web application with high-contrast accessibility styling)

---

## 🏗️ Architecture Pipeline

```
  [ Input Image ]
         │
         ▼
┌────────────────────────┐
│ ResNet-50 CNN Encoder  │  (Removes classification head -> extracts 2048-dim feature)
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│   Linear Projection    │  (Projects 2048-dim vector -> 256-dim embedding)
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│      LSTM Decoder      │  (Initial state seeded by image; generates tokens iteratively)
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│    Generated Caption   │  (Capitalized, formatted English sentence)
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  gTTS Voice Synthesis  │  (High quality spoken narration with MP3 download)
└────────────────────────┘
```

---

## 📁 Repository Structure

```
image_caption_generator/
├── app.py                 # Streamlit web application & inference pipeline
├── caption_model.pth      # Pre-trained PyTorch model weights (~35.4 MB)
├── vocabulary.json        # Word-to-index mapping dictionary (8,782 words)
├── sample_images/         # Demonstration sample images
│   ├── hiker.jpg
│   └── architecture.jpg
├── requirements.txt       # Dependencies for local and cloud deployment
├── run_app.bat            # Windows 1-click launcher script
├── .gitignore             # Git ignore file (keeps repo clean)
└── README.md              # Project documentation
```

---

## 🚀 Running Locally

### Option 1: One-Click Launcher (Windows)
Double-click `run_app.bat` inside the project folder.

### Option 2: Command Line
1. Clone or extract the repository:
   ```bash
   cd image_caption_generator
   ```
2. Install required packages (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   py -3.13 -m streamlit run app.py
   # or
   streamlit run app.py
   ```
4. Open your browser at `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Community Cloud

The model weights file (`caption_model.pth`) is **35.4 MB**, well below GitHub's 100 MB limit, so it can be hosted directly in your GitHub repository!

### Step 1: Create a GitHub Repository
1. Go to [GitHub](https://github.com/) and log into your account.
2. Click **New** repository.
3. Name your repository (e.g., `image-caption-generator-voice`).
4. Set visibility to **Public** and leave "Add a README" unchecked (since we already have one).
5. Click **Create repository**.

### Step 2: Push Your Code to GitHub
Run the following commands in PowerShell from the project folder:

```powershell
cd d:\Users\Sherin\Downloads\image_caption_generator
git add .
git commit -m "Initial commit of Image Caption Generator project"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
git push -u origin main
```
*(Replace `<YOUR-USERNAME>` and `<YOUR-REPO-NAME>` with your GitHub username and repository name).*

### Step 3: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app**.
3. Select your repository: `<YOUR-USERNAME>/image-caption-generator-voice`.
4. Select Branch: `main`.
5. Set Main file path: `app.py`.
6. Click **Deploy!**.

Your application will be live on a public URL (e.g. `https://<your-app-name>.streamlit.app`) that you can present to your college evaluators!

---

## 🎓 College Viva / Evaluation FAQs

1. **What is the role of the CNN Encoder?**  
   The CNN (ResNet-50) serves as the visual perception engine. By using pre-trained weights from ImageNet, it extracts high-level semantic features (objects, shapes, relationships) without requiring millions of training images.

2. **Why use an LSTM for the decoder?**  
   LSTMs contain memory cells and gating units (input, forget, output gates) that prevent the vanishing gradient problem, allowing the model to capture syntax and long-range dependencies in captions.

3. **How does voice output benefit the visually impaired?**  
   By transforming visual scenes into immediate audible spoken feedback, the system acts as an assistive electronic guide, enhancing independence and situational awareness.
