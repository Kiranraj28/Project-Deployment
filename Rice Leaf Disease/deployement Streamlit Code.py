import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model (ensure model file is in working directory)
model = load_model('best_model.keras')  
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

st.title("Rice Leaf Disease Classification")
st.write("Upload a rice leaf image to detect its disease type.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded rice leaf', use_column_width=True)
    img_resized = img.resize((256, 256))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    st.success(f"Prediction: {predicted_class} ({confidence:.2%} confidence)")
