# ==============================
# 🌾 Rice Leaf Disease Prediction App
# ==============================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="🌾 Rice Leaf Disease Classifier", layout="centered")

# ------------------------------
# Sidebar Section
# ------------------------------
with st.sidebar:
    st.title("🌾 Project Overview")
    st.markdown("""
    ### 🎯 **Objective**
    Build a robust image classification model to accurately detect **Rice Leaf Diseases**  
    from leaf images using **Deep Learning**.

    ### 🧠 **Approach**
    1. **Dataset Preparation**
       - Organized labeled images for each disease class.  
       - Used `image_dataset_from_directory` for efficient loading.  
       - Split into training, validation & test sets.

    2. **Preprocessing**
       - Resized all images to **256×256**.  
       - Applied pixel normalization using `preprocess_input` (MobileNetV2 standard).

    3. **Model Architecture**
       - Leveraged **MobileNetV2** (Transfer Learning).  
       - Added custom Dense layers for classification.  
       - Used **Softmax** activation for multi-class output.

    4. **Training & Evaluation**
       - Monitored validation accuracy to prevent overfitting.  
       - Evaluated using accuracy & loss metrics on test data.

    ### 🏆 **Outcome**
    The model achieved **high accuracy** in classifying:
    - Bacterial Leaf Blight  
    - Brown Spot  
    - Leaf Smut  

    Demonstrating the **power of deep learning** in **agricultural disease detection**.
    """)

    st.markdown("---")
    st.markdown("📘 *Developed by [Kiran Raj T](https://github.com/Kiranraj28)*")

# ------------------------------
# 1. Load Model (cached)
# ------------------------------
model_path = "rice_leaf_classifier_clean.keras"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ------------------------------
# 2. Class Names
# ------------------------------
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']

# ------------------------------
# 3. Preprocessing Function
# ------------------------------
def preprocess_image(image):
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# ------------------------------
# 4. Streamlit UI
# ------------------------------
st.title("🌾 Rice Leaf Disease Classification")
st.markdown("Upload a rice leaf image to predict its disease type using a **MobileNetV2** model.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)

    with st.spinner("🔍 Analyzing image..."):
        predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"✅ Predicted Disease: **{predicted_class}**")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))

# ------------------------------
# Footer
# ------------------------------
st.caption("Model: MobileNetV2 | Framework: TensorFlow | Deployment: Streamlit Cloud")
