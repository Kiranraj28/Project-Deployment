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
    Build a robust image classification model to accurately detect **Rice Leaf Diseases** from leaf images using **Deep Learning**.

    ### 🧠 **Approach**
    1. **Dataset Preparation** - Organized labeled images for each disease class. 
        - Used `image_dataset_from_directory` for efficient loading. 
        - Split into training, validation & test sets.

    2. **Preprocessing** - Resized all images to **256×256**. 
        - Applied pixel normalization using `preprocess_input` (MobileNetV2 standard).

    3. **Model Architecture** - Leveraged **MobileNetV2** (Transfer Learning). 
        - Added custom Dense layers for classification. 
        - Used **Softmax** activation for multi-class output.

    4. **Training & Evaluation** - Monitored validation accuracy to prevent overfitting. 
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
# 1. Load Model (cached) - RESOLVES NameError
# ------------------------------

# Define the relative path to the model file
# Ensure this path is correct based on your repo structure!
model_path = "Rice Leaf Disease Classification/rice_leaf_classifier_clean.keras"

@st.cache_resource
def load_model():
    """Loads the Keras model using Streamlit's resource caching."""
    # Check if the file exists before attempting to load (optional but helpful)
    # import os
    # if not os.path.exists(model_path):
    #     st.error(f"Model file not found at: {model_path}. Please check your repo structure.")
    #     st.stop()
        
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"🚨 CRITICAL ERROR: Could not load the model from {model_path}. \nCheck the path, file existence (and Git LFS if the file is large).")
        st.exception(e) # Display full exception details in the app
        st.stop() # Stop execution if the model can't be loaded

# Load the model object and assign it to the 'model' variable
# This makes 'model' available globally, fixing the NameError.
model = load_model()

# ------------------------------
# 2. Class Names (same order as training)
# ------------------------------
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']

# ------------------------------
# 3. Preprocessing Function
# ------------------------------
def preprocess_image(image):
    """Resizes, converts to array, expands dimensions, and preprocesses image pixels."""
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)  # Converts range to [-1, 1] for MobileNetV2
    return image


# ------------------------------
# 4. Streamlit UI & Prediction Logic
# ------------------------------
st.title("🌾 Rice Leaf Disease Classification")
st.markdown("Upload a rice leaf image to predict its disease type using a **MobileNetV2** model.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    with st.spinner("Analyzing image..."):
        img_array = preprocess_image(image)
        
        # The 'model' variable is now defined, so this line will work
        predictions = model.predict(img_array)
        
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    # Display results
    st.markdown("---")
    st.header("Diagnosis Result")
    st.success(f"✅ Predicted Disease: **{predicted_class}**")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence) / 100.0) # Progress expects a value between 0.0 and 1.0


# ------------------------------
# Footer
# ------------------------------
st.caption("Model: MobileNetV2 | Framework: TensorFlow | Deployment: Streamlit Cloud")
