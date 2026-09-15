import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkinCare AI",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

.main {
    background-color: #0e1117;
}

h1 {
    text-align: center;
    color: #ffffff !important;
    font-size: 48px !important;
}

h2, h3 {
    color: #ffffff !important;
}

p, li, label {
    color: #e5e7eb !important;
}

.subtitle {
    text-align: center;
    color: #aeb6c2 !important;
    font-size: 18px;
    margin-bottom: 35px;
}

.card {
    background-color: #171b24;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #303642;
    margin-bottom: 20px;
}

.result-title {
    color: #ffffff !important;
    font-size: 28px;
    font-weight: bold;
}

.result-text {
    color: #e5e7eb !important;
    font-size: 20px;
    margin: 10px 0;
}

.severity {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 20px;
    background-color: #d99a2b;
    color: white !important;
    font-weight: bold;
    margin: 8px 0;
}

.confidence {
    color: #6ee7b7 !important;
    font-size: 20px;
    font-weight: bold;
}

.info-box {
    background-color: #173b32;
    border-left: 5px solid #2dd4bf;
    padding: 15px;
    border-radius: 8px;
    color: #e5e7eb !important;
}

.warning-box {
    background-color: #4a3a18;
    border-left: 5px solid #facc15;
    padding: 15px;
    border-radius: 8px;
    color: #ffffff !important;
}

.small-text {
    color: #aeb6c2 !important;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "best_skin_model.keras"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_skin_model():

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


try:
    best_model = load_skin_model()

except Exception as e:

    st.error("Model could not be loaded.")

    st.code(str(e))

    st.stop()


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Acne_Mild",
    "Acne_Moderate",
    "Acne_Severe",
    "Eczema_Mild",
    "Eczema_Moderate",
    "Eczema_Severe",
    "Normal",
    "Xyz"
]


IMG_SIZE = (224, 224)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<h1>🩺 SkinCare AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-Based Skin Disease Detection, Severity Assessment & Care Guidance
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ABOUT SYSTEM
# ============================================================

with st.expander("ℹ️ How SkinCare AI Works"):

    st.markdown("""
    **1. Image Upload**  
    Upload a skin image for analysis.

    **2. MobileNetV2 Classification**  
    The pretrained MobileNetV2 model analyzes the image.

    **3. TTA Prediction**  
    Original and horizontally flipped versions of the image are evaluated.

    **4. Severity Assessment**  
    The predicted class determines the disease and severity level.

    **5. Grad-CAM Explainability**  
    Important image regions contributing to the prediction are highlighted.

    **6. Care Guidance**  
    General educational care guidance is displayed.
    """)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("## 📤 Upload Skin Image")

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# CARE GUIDANCE
# ============================================================

def get_care_guidance(disease, severity):

    if disease == "Acne":

        if severity == "Mild":
            return [
                "Maintain a gentle skincare routine.",
                "Avoid picking or squeezing lesions.",
                "Keep the skin clean and avoid harsh products.",
                "Consult a dermatologist if the condition persists."
            ]

        elif severity == "Moderate":
            return [
                "Maintain a gentle and consistent skincare routine.",
                "Avoid picking, squeezing, or irritating the affected area.",
                "Avoid harsh or excessive skincare products.",
                "Consider consulting a dermatologist for appropriate evaluation."
            ]

        else:
            return [
                "Avoid picking or squeezing affected areas.",
                "Use gentle skincare products and avoid irritation.",
                "Monitor changes in the affected skin.",
                "Professional dermatological evaluation is recommended."
            ]

    elif disease == "Eczema":

        if severity == "Mild":
            return [
                "Keep the skin moisturized regularly.",
                "Avoid known skin irritants.",
                "Use gentle, fragrance-free skincare products.",
                "Consult a dermatologist if symptoms continue."
            ]

        elif severity == "Moderate":
            return [
                "Keep the affected skin moisturized.",
                "Avoid irritating soaps and skincare products.",
                "Try to avoid scratching the affected area.",
                "Consider consulting a dermatologist."
            ]

        else:
            return [
                "Avoid scratching or irritating the affected skin.",
                "Keep the skin moisturized and protected.",
                "Avoid known triggers or irritants.",
                "Professional dermatological evaluation is recommended."
            ]

    elif disease == "Normal":

        return [
            "No specific skin condition was detected by the model.",
            "Maintain a regular and gentle skincare routine.",
            "Use appropriate sun protection.",
            "Monitor any unusual changes in the skin."
        ]

    else:

        return [
            "The image was classified as miscellaneous.",
            "Maintain a gentle skincare routine.",
            "Avoid irritating the affected area.",
            "Consider consulting a dermatologist for proper evaluation."
        ]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(img):

    img_resized = img.resize(IMG_SIZE)

    img_array = np.array(img_resized).astype("float32") / 255.0

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Original prediction
    original_prob = best_model.predict(
        img_array,
        verbose=0
    )

    # Horizontal flip
    flipped_img = np.flip(img_array, axis=2)

    flipped_prob = best_model.predict(
        flipped_img,
        verbose=0
    )

    # TTA
    final_prob = (original_prob + flipped_prob) / 2

    predicted_index = int(
        np.argmax(final_prob[0])
    )

    predicted_class = class_names[predicted_index]

    confidence = float(
        final_prob[0][predicted_index] * 100
    )

    # Disease and severity
    if predicted_class.startswith("Acne"):

        disease = "Acne"
        severity = predicted_class.replace(
            "Acne_", ""
        )

    elif predicted_class.startswith("Eczema"):

        disease = "Eczema"
        severity = predicted_class.replace(
            "Eczema_", ""
        )

    elif predicted_class == "Normal":

        disease = "Normal"
        severity = "None"

    else:

        disease = "Miscellaneous"
        severity = "Not Defined"

    return (
        predicted_class,
        disease,
        severity,
        confidence,
        final_prob[0],
        img_array
    )


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(img_array, predicted_index):

    try:

        # MobileNetV2 base model
        base_model = best_model.layers[0]

        # Last convolutional layer
        last_conv_layer = base_model.get_layer("Conv_1")

        # Input for Grad-CAM
        grad_input = tf.keras.Input(
            shape=(224, 224, 3)
        )

        # Run through base model until Conv_1
        conv_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=last_conv_layer.output
        )

        conv_output = conv_model(grad_input)

        x = conv_output

        # Continue MobileNetV2 after Conv_1
        found_layer = False

        for layer in base_model.layers:

            if layer.name == "Conv_1":
                found_layer = True
                continue

            if found_layer:
                x = layer(x)

        # Continue classification head
        for layer in best_model.layers[1:]:

            x = layer(x)

        grad_model = tf.keras.Model(
            inputs=grad_input,
            outputs=[conv_output, x]
        )

        # Gradient calculation
        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(
                img_array
            )

            class_channel = predictions[:, predicted_index]

        grads = tape.gradient(
            class_channel,
            conv_outputs
        )

        # Global average pooling
        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        if max_value > 0:

            heatmap /= max_value

        heatmap = heatmap.numpy()

        # Resize heatmap
        heatmap = cv2.resize(
            heatmap,
            (224, 224)
        )

        # Use matplotlib colormap safely
        cmap = plt.colormaps["jet"]

        colored_heatmap = cmap(
            heatmap
        )[:, :, :3]

        colored_heatmap = (
            colored_heatmap * 255
        ).astype(np.uint8)

        # Original image
        original = (
            img_array[0] * 255
        ).astype(np.uint8)

        # Overlay
        overlay = cv2.addWeighted(
            original,
            0.55,
            colored_heatmap,
            0.45,
            0
        )

        return colored_heatmap, overlay

    except Exception as e:

        st.warning(
            "Grad-CAM could not be generated."
        )

        st.code(str(e))

        return None, None


# ============================================================
# MAIN ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown("---")

    col1, col2 = st.columns(
        [1, 1]
    )

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        st.markdown(
            "## 🖼️ Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )

        analyze = st.button(
            "🔍 Analyze Skin Image",
            use_container_width=True
        )


    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    if analyze:

        with st.spinner(
            "Analyzing skin image..."
        ):

            (
                predicted_class,
                disease,
                severity,
                confidence,
                probabilities,
                img_array
            ) = predict_image(image)


        # ----------------------------------------------------
        # RIGHT COLUMN
        # ----------------------------------------------------

        with col2:

            st.markdown(
                "## 🔬 Prediction Result"
            )

            st.markdown(
                f"""
                <div class="card">

                <div class="result-title">
                Disease: {disease}
                </div>

                <div class="severity">
                {severity.upper()}
                </div>

                <div class="confidence">
                Confidence: {confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # CONFIDENCE BREAKDOWN
            # ------------------------------------------------

            st.markdown(
                "### 📊 Model Confidence Breakdown"
            )

            confidence_dict = {}

            for i, class_name in enumerate(
                class_names
            ):

                confidence_dict[
                    class_name
                ] = float(
                    probabilities[i] * 100
                )

            sorted_confidence = sorted(
                confidence_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for name, value in sorted_confidence:

                st.progress(
                    min(value / 100, 1.0),
                    text=f"{name}: {value:.2f}%"
                )


        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            "## 🧠 Explainable AI — Grad-CAM"
        )

        st.markdown(
            """
            <div class="info-box">
            Grad-CAM highlights the image regions that contributed
            to the model's prediction.
            </div>
            """,
            unsafe_allow_html=True
        )

        heatmap, overlay = generate_gradcam(
            img_array,
            class_names.index(
                predicted_class
            )
        )


        if heatmap is not None:

            grad_col1, grad_col2, grad_col3 = st.columns(
                3
            )

            with grad_col1:

                st.markdown(
                    "### Original Image"
                )

                st.image(
                    image.resize(IMG_SIZE),
                    use_container_width=True
                )

            with grad_col2:

                st.markdown(
                    "### Grad-CAM Heatmap"
                )

                st.image(
                    heatmap,
                    use_container_width=True
                )

            with grad_col3:

                st.markdown(
                    "### Grad-CAM Overlay"
                )

                st.image(
                    overlay,
                    use_container_width=True
                )


        # ----------------------------------------------------
        # CARE GUIDANCE
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            "## 💡 General Care Guidance"
        )

        guidance = get_care_guidance(
            disease,
            severity
        )

        for item in guidance:

            st.markdown(
                f"• {item}"
            )


        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="warning-box">
            ⚠️ <b>Disclaimer:</b> This AI system is intended
            for screening and educational support only. It does
            not provide a medical diagnosis and does not replace
            professional medical advice.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="small-text">
    SkinCare AI | Deep Learning Based Skin Disease Detection,
    Severity Assessment & Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)