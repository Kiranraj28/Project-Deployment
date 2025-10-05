# ==============================
# 🌾 Rice Leaf Disease Prediction App
# ==============================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
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
import os
import tensorflow as tf
# import streamlit as st # Assuming you are using Streamlit based on the original code snippet

# @st.cache_resource # Uncomment this if you are using Streamlit
def load_model():
    # 1. Get the absolute path to the directory containing the current script.
    #    This should be '/path/to/Project-Deployment/Rice Leaf Disease Classification'
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    # 2. Join the script directory with the model file name.
    #    This results in: '/path/to/.../Rice Leaf Disease Classification/rice_leaf_classifier_clean.keras'
    model_path = os.path.join(script_dir, "rice_leaf_classifier_clean.keras")
    
    # Optional: Print the resolved path to verify it is correct
    print(f"Attempting to load model from: {model_path}")

    # 3. Load the model
    # Note: tf.keras.models.load_model will raise the ValueError if the path is bad
    model = tf.keras.models.load_model(model_path, compile=False) 
    
    return model

model = load_model() # Uncomment this to test

# ------------------------------
# 2. Class Names (same order as training)
# ------------------------------
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']

# ------------------------------
# 3. Preprocessing Function
# ------------------------------
def preprocess_image(image):
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)  # Converts range to [-1, 1]
    return image


# ------------------------------
# 4. Streamlit UI
# ------------------------------
st.title("🌾 Rice Leaf Disease Classification")
st.markdown("Upload a rice leaf image to predict its disease type using a **MobileNetV2** model.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display results
    st.success(f"✅ Predicted Disease: **{predicted_class}**")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))

# ------------------------------
# Footer
# ------------------------------
st.caption("Model: MobileNetV2 | Framework: TensorFlow | Deployment: Streamlit Cloud")

