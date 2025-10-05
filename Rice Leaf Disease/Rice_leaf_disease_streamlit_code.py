import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Rice Leaf Disease Detection Platform",
    layout="centered"
)

st.sidebar.title("About This Project")
st.sidebar.markdown(
    """
    **Rice Leaf Disease Classification**  
    This project leverages deep learning to automate the identification of common rice leaf diseases from images.  
    Using a CNN with MobileNetV2 as backbone and custom classification layers, the model distinguishes between Bacterial Leaf Blight, Brown Spot, and Leaf Smut.

    **Key Features:**  
    - Robust deep learning architecture  
    - Real-time image analysis  
    - Data augmentation for resilience  
    - Practical application for agriculture  
    """
)
st.sidebar.markdown(
    """
    **Understanding Confidence Score**  
    The confidence score expresses how certain the model is about its prediction, ranging from 0 (no certainty) to 1 (maximum certainty).  
    - Closer to 1: highly reliable  
    - Moderate (0.5-0.7): more uncertainty  
    - Use to guide trust/follow-up
    """
)

@st.cache_resource
def load_model_file():
    model_path = "Rice Leaf Disease/best_model.h5"
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model_file()

# Adjust these names as per your trained classes
CLASS_NAMES = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

st.title("Rice Leaf Disease Prediction")

uploaded_file = st.file_uploader("Upload a rice leaf image (.jpg, .jpeg, .png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 256, 256, 3)

    preds = model.predict(img_array)
    result_idx = np.argmax(preds)
    confidence = preds[0][result_idx]

    st.success(f"Prediction: {CLASS_NAMES[result_idx]}")
    st.info(f"Confidence: {confidence:.2f}")

else:
    st.info("Please upload an image file to classify.")

