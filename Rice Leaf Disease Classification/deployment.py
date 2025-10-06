import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- Configuration ---
MODEL_PATH = 'rice_leaf_classifier_final.keras'
IMAGE_SIZE = (224, 224) # Ensure this matches your model's input size

# --- IMPORTANT: ADJUST THESE CLASS NAMES ---
# Replace the placeholder class names with the EXACT 3 DISEASE LABELS your model predicts.
CLASS_NAMES = [
    'Bacterial Blight',
    'Brown Spot',
    'Leaf Smut' 
    # NOTE: Only 3 class names are included, all representing a disease or status.
]

# --- Load Model ---
@st.cache_resource
def get_model():
    """Loads the Keras model with error handling."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_PATH}' not found. "
                 "Please ensure the Keras file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = get_model()

# --- Prediction Function ---
def predict(image, model):
    """
    Preprocesses the image, makes a prediction, and returns the results.
    """
    # 1. Preprocess the image
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img)
    
    # Add batch dimension and normalize (assuming 0-1 normalization)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0 

    # 2. Make prediction
    predictions = model.predict(img_array)
    # Use softmax to get probabilities 
    scores = tf.nn.softmax(predictions[0])

    # 3. Get the result
    predicted_class_index = np.argmax(scores)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(scores) * 100

    return predicted_class_name, confidence, scores

# --- Streamlit App UI ---
st.set_page_config(
    page_title="Rice Leaf Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

st.title("🚨 Rice Leaf Disease Classification (Disease Focus)")
st.markdown(f"**This model predicts among {len(CLASS_NAMES)} disease categories:** {', '.join(CLASS_NAMES)}")

if model is None:
    st.stop()

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a rice leaf image...", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    st.subheader("Uploaded Image")
    st.image(image, caption='Image for Prediction', use_column_width=True)

    # Prediction button
    if st.button('Classify Disease'):
        with st.spinner('Analyzing the image...'):
            try:
                # Get prediction
                class_name, confidence, all_scores = predict(image, model)

                st.subheader("Prediction Result")
                
                st.error(f"Prediction: **{class_name}** 🚨") # Highlight as it's a disease prediction
                    
                st.info(f"Confidence: **{confidence:.2f}%**")

                # Optional: Show all class probabilities
                st.markdown("---")
                st.subheader("Detailed Probabilities")
                
                results_dict = {
                    "Disease/Status": CLASS_NAMES,
                    "Probability (%)": [f"{s * 100:.2f}" for s in all_scores.numpy()]
                }
                st.dataframe(results_dict)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                
# --- Footer ---
st.markdown("---")
st.markdown("Application powered by a Keras CNN Model and Streamlit.")
