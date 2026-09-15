# SkinCare AI – Skin Disease Detection & Severity Assessment

A deep learning-based image classification system that analyzes skin images to predict common skin conditions along with their severity levels, using **MobileNetV2 transfer learning**, **Grad-CAM explainability**, and a **Streamlit** web interface.

---

## 📌 Project Overview

SkinCare AI is an academic (MCA) deep learning project that classifies skin images into disease categories along with their severity levels (Mild, Moderate, Severe). The model is built on **MobileNetV2** using transfer learning and is deployed through an interactive **Streamlit** web application. To improve interpretability, the system uses **Grad-CAM** visualizations to highlight the image regions that influenced the model's prediction, and applies **Test-Time Augmentation (TTA)** to improve prediction robustness.

> ⚠️ This project is developed strictly for **academic and educational purposes** and is **not intended for real-world medical diagnosis**.

---

## ✨ Features

- Image-based skin disease classification with severity levels
- Transfer learning using pretrained MobileNetV2 (ImageNet weights)
- Test-Time Augmentation (original + horizontally flipped image)
- Grad-CAM based visual explainability
- Interactive Streamlit web application
- Confidence score display for predictions
- General care guidance based on predicted class

---

## 🩺 Classification Categories

The model classifies input images into **8 classes**:

| # | Class Name | Description |
|---|-------------|-------------|
| 1 | Acne_Mild | Acne – Mild severity |
| 2 | Acne_Moderate | Acne – Moderate severity |
| 3 | Acne_Severe | Acne – Severe severity |
| 4 | Eczema_Mild | Eczema – Mild severity |
| 5 | Eczema_Moderate | Eczema – Moderate severity |
| 6 | Eczema_Severe | Eczema – Severe severity |
| 7 | Normal | Healthy/normal skin |
| 8 | Xyz | Miscellaneous / unrelated images (negative class) |

> **Note:** `Xyz` is **NOT** a skin disease class. It contains miscellaneous/unrelated images such as birds, common objects, and other non-skin images. It is used as a **negative/unrelated class** to help the model reject irrelevant inputs.

---

## 📊 Dataset

**Dataset Used:** Multi-Class Skin Disease Image Dataset with Severity Levels

| Split | Number of Images |
|-------|------------------|
| Training | 3,642 |
| Validation | 777 |
| Testing | 792 |

- **Image Size:** 224 × 224
- **Number of Classes:** 8

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Streamlit
- Grad-CAM
- Google Colab

---

## 🧠 Model Architecture

The model is built using **MobileNetV2** as a feature extractor, followed by custom classification layers.

```
Input (224 x 224 x 3)
        │
        ▼
MobileNetV2 (Pretrained on ImageNet, Transfer Learning)
        │
        ▼
Global Average Pooling
        │
        ▼
Dropout
        │
        ▼
Dense Layer (128 neurons, ReLU)
        │
        ▼
Dropout
        │
        ▼
Dense Layer (8 classes, Softmax)
        │
        ▼
Output: Predicted Class + Confidence Score
```

**Architecture Summary:**
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Global Average Pooling layer
- Dropout layer (for regularization)
- Dense layer with 128 neurons
- Dropout layer
- Final Dense layer with 8 output classes and Softmax activation

---

## 🖼️ Image Preprocessing

Before being fed into the model, images undergo the following preprocessing steps:

- Resize images to **224 × 224**
- Normalize pixel values to the range **0–1**

---

## 🔄 Data Augmentation

To improve model generalization, the following augmentation techniques are applied during training:

- Rotation
- Width shift
- Height shift
- Zoom
- Horizontal flip

---

## 🔁 Test-Time Augmentation (TTA)

During inference, the model uses **Test-Time Augmentation** to improve prediction stability:

1. The **original image** is passed through the model.
2. A **horizontally flipped version** of the same image is also passed through the model.
3. The prediction probabilities from both versions are **averaged** to obtain the final prediction.

This helps reduce variance in predictions caused by orientation-sensitive features.

---

## 🔍 Explainable AI – Grad-CAM

The project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize which regions of the input image most influenced the model's prediction.

- Highlights the image regions contributing to the predicted class
- Improves transparency and trust in model predictions
- Useful for understanding model behavior during evaluation

> ⚠️ Grad-CAM provides **visual interpretability only**. It **does not confirm a medical diagnosis** and should not be used as clinical evidence.

---

## 📈 Model Performance

- **Test Accuracy:** **69.19%**

**Observations:**
- The **Normal** and **Xyz** classes performed relatively better than the disease classes.
- Some **moderate-severity** classes (e.g., distinguishing between Mild/Moderate/Severe stages) were more challenging for the model to classify correctly, likely due to subtle visual differences between severity levels.

> The reported accuracy reflects the actual model performance and has not been exaggerated or modified.

---

## 💻 Streamlit Application

The trained model is deployed using a **Streamlit** web application that allows users to upload a skin image and receive a prediction along with severity, confidence score, and Grad-CAM visualization.

---

## ⚙️ Application Workflow

```
   Upload Image
        │
        ▼
   Preprocessing
        │
        ▼
Original Image + Horizontally Flipped Image
        │
        ▼
   MobileNetV2 Prediction (TTA)
        │
        ▼
     Final Class
        │
        ▼
  Disease & Severity
        │
        ▼
   Confidence Score
        │
        ▼
      Grad-CAM
        │
        ▼
General Care Guidance
```

---

## 🗂️ Prediction Mapping

| Predicted Class | Disease | Severity |
|------------------|---------|----------|
| Acne_Mild | Acne | Mild |
| Acne_Moderate | Acne | Moderate |
| Acne_Severe | Acne | Severe |
| Eczema_Mild | Eczema | Mild |
| Eczema_Moderate | Eczema | Moderate |
| Eczema_Severe | Eczema | Severe |
| Normal | — | Healthy Skin |
| Xyz | — | Unrelated / Non-skin Image |

---

## 📁 Project Structure

```
SkinCare_AI/
├── app.py
├── best_skin_model.keras
├── README.md
└── Dataset/
    ├── train/
    ├── val/
    └── test/
```

---

## 🔧 Installation

Install the required libraries using the following command:

```bash
pip install tensorflow streamlit numpy pandas opencv-python matplotlib pillow
```

---

## ▶️ How to Run

1. Clone or download this repository.
2. Ensure `app.py` and `best_skin_model.keras` are in the same directory.
3. Install the required dependencies (see Installation section above).
4. Run the Streamlit application:

```bash
streamlit run app.py
```

5. Open the local URL provided by Streamlit in your browser.
6. Upload a skin image to view the predicted class, severity, confidence score, and Grad-CAM visualization.

---

## 🚀 Future Improvements

- Increase dataset size for better generalization across severity levels
- Experiment with additional/deeper architectures for improved accuracy
- Improve classification of moderate-severity classes
- Add support for multi-image batch predictions
- Enhance UI/UX of the Streamlit application
- Deploy the application on a cloud platform for public access

---

## 🎓 Academic Purpose

This project was developed as part of an **MCA (Master of Computer Applications)** academic project to demonstrate practical application of deep learning, transfer learning, model explainability, and web deployment concepts.

---

## ⚠️ Disclaimer

This project is intended **solely for academic and educational purposes**. It is **not a certified medical tool** and **does not provide medical diagnosis**. Predictions generated by this system should **not** be used as a substitute for professional medical consultation. Always consult a qualified dermatologist or healthcare provider for accurate diagnosis and treatment of skin conditions.
