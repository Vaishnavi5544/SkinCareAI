SkinCare AI -- Skin Disease Detection & Severity Assessment

SkinCare AI is a deep learning-based image classification system
designed to analyze skin images and predict the category and severity
level of common skin conditions. The project uses MobileNetV2 transfer
learning and provides an explainable prediction using Grad-CAM.

The model is integrated into a Streamlit web application where users
can upload a skin image and view the predicted category, severity,
confidence score, visual explanation, and general care guidance.

Disclaimer: This project is developed for academic and educational
purposes. It is not a medical diagnostic tool and should not be used
as a substitute for professional medical advice.

Features

Skin image classification using deep learning

Classification of Acne and Eczema severity levels

Normal image classification

Miscellaneous/Unrelated image detection

MobileNetV2-based transfer learning

Image preprocessing and data augmentation

Test-Time Augmentation (TTA) using original and horizontally flipped
images

Confidence score and class-wise confidence breakdown

Grad-CAM based explainability

General care guidance

Streamlit-based interactive web interface

Project Categories

The model uses the following 8 classes:

Class               Meaning

Acne_Mild         Acne -- Mild
Acne_Moderate     Acne -- Moderate
Acne_Severe       Acne -- Severe
Eczema_Mild       Eczema -- Mild
Eczema_Moderate   Eczema -- Moderate
Eczema_Severe     Eczema -- Severe
Normal            Normal skin image
Xyz               Miscellaneous / unrelated image

Note about Xyz

Xyz is not a skin disease. It contains miscellaneous or unrelated
images such as birds, common objects, and other non-skin images. It is
retained as a negative/unrelated class so that the model does not force
every uploaded image into a skin-disease category.

Dataset

The project uses the Multi-Class Skin Disease Image Dataset with
Severity Levels.

Dataset split used in the project:

Training: 3,642 images

Validation: 777 images

Testing: 792 images

Number of classes: 8

Image size: 224 × 224 pixels

Batch size: 32

The dataset is organized into separate training, validation, and testing
directories.

Dataset/
├── train/
│   ├── Acne_Mild/
│   ├── Acne_Moderate/
│   ├── Acne_Severe/
│   ├── Eczema_Mild/
│   ├── Eczema_Moderate/
│   ├── Eczema_Severe/
│   ├── Normal/
│   └── Xyz/
│
├── val/
│   └── ...
│
└── test/
    └── ...

Technologies Used

Python

TensorFlow

Keras

MobileNetV2

NumPy

Pandas

OpenCV

Matplotlib

Streamlit

Grad-CAM

Google Colab for model development and training

Model Architecture

The project uses MobileNetV2, a pretrained convolutional neural
network, as the feature extraction backbone.

The base MobileNetV2 model was initialized with ImageNet weights and
followed by a custom classification head:

Input Image
    ↓
Resize to 224 × 224
    ↓
Normalization
    ↓
MobileNetV2
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Dense Layer (128 neurons)
    ↓
Dropout
    ↓
Dense Layer (8 classes)
    ↓
Softmax
    ↓
Predicted Class

Why MobileNetV2?

MobileNetV2 was selected because it is a lightweight and efficient
architecture that can provide useful image features while being suitable
for deployment in an application.

Image Preprocessing

Before being given to the model:

The uploaded image is resized to 224 × 224 pixels.

Pixel values are normalized to the range 0--1.

The processed image is passed to the trained model.

Data Augmentation

Training images were augmented using:

Rotation

Width shift

Height shift

Zoom

Horizontal flipping

Data augmentation helps the model learn from variations in image
orientation and appearance.

Test-Time Augmentation (TTA)

During application inference, the system uses two versions of the
uploaded image:

Original image

Horizontally flipped image

The model predicts both images and their probabilities are averaged.

Original Image ──────┐
                     ├──→ Average Probabilities → Final Prediction
Flipped Image ───────┘

This is used to make the prediction more consistent during inference.

Explainable AI -- Grad-CAM

The project integrates Grad-CAM (Gradient-weighted Class Activation
Mapping) to make the model prediction more understandable.

Grad-CAM produces a heatmap showing the regions of the image that
contributed to the model's prediction.

Original Image
      ↓
MobileNetV2
      ↓
Prediction
      ↓
Grad-CAM
      ↓
Heatmap / Overlay

The application displays:

Original image

Grad-CAM heatmap

Grad-CAM overlay

Grad-CAM highlights regions associated with the model's prediction. It
does not prove that a highlighted region is a specific medical lesion.

Model Performance

The trained model achieved:

Test Accuracy: 69.19%

The model performed relatively better on the Normal and Xyz
classes, while some moderate-severity classes were more challenging to
classify.

The classification report and confusion matrix were used to evaluate
class-wise performance.

Streamlit Application

The trained model is deployed using Streamlit.

Application Workflow

Upload Skin Image
        ↓
Image Preprocessing
        ↓
Original + Flipped Image
        ↓
MobileNetV2 Prediction
        ↓
Final Class
        ↓
Disease + Severity
        ↓
Confidence Score
        ↓
Grad-CAM Explanation
        ↓
General Care Guidance

Prediction Mapping

For example:

Acne_Mild
    ↓
Disease: Acne
Severity: Mild

Eczema_Severe
    ↓
Disease: Eczema
Severity: Severe

Normal
    ↓
Disease: Normal
Severity: None

Xyz
    ↓
Category: Miscellaneous
Severity: Not Defined

Project Structure

SkinCare_AI/
│
├── app.py
├── best_skin_model.keras
├── README.md
│
└── Dataset/
    ├── train/
    ├── val/
    └── test/

The dataset does not need to be included in the GitHub repository if it
is large or subject to dataset-sharing restrictions.

Installation

1. Clone the repository

git clone <your-github-repository-link>
cd SkinCare_AI

2. Install required libraries

pip install tensorflow streamlit numpy pandas opencv-python matplotlib pillow

3. Make sure the trained model is present

Place:

best_skin_model.keras

in the same folder as app.py.

Run the Application

Run the following command from the project folder:

streamlit run app.py

If required, you can also use:

python -m streamlit run app.py

The Streamlit application will open in your browser.

Future Improvements

Use a larger and more diverse skin-image dataset

Add more skin conditions and severity categories

Evaluate model performance across different skin tones

Perform further fine-tuning of the pretrained model

Improve classification of visually similar severity levels

Obtain expert feedback for Grad-CAM explanations

Develop a mobile application

Add multilingual support

Validate the system using clinically collected and expert-verified
data

Academic Purpose

This project demonstrates the application of:

Deep Learning

Transfer Learning

Image Classification

Data Augmentation

Test-Time Augmentation

Explainable AI

Model Evaluation

Streamlit Deployment

It was developed as an academic project to explore how deep learning can
be applied to image-based skin condition classification and severity
assessment.

Disclaimer

SkinCare AI is an academic/educational project and is not intended to
diagnose, treat, or replace consultation with a qualified healthcare
professional. Predictions and care guidance should not be considered
medical advice
