#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import numpy as np

# Load the saved pipeline/model (assumed to include scaling if used)
with open('Classification/dep_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Activity Prediction: Walk or Run")

st.write("Input sensor data to predict activity:")

# Wrist input - 0 or 1 (left or right wrist)
wrist = st.selectbox("Wrist (0: left, 1: right)", options=[0, 1], index=0)

# Accelerometer inputs
acceleration_x = st.number_input("Acceleration X (m/s²)")
acceleration_y = st.number_input("Acceleration Y (m/s²)")
acceleration_z = st.number_input("Acceleration Z (m/s²)")

# Gyroscope inputs
gyro_x = st.number_input("Gyro X (degrees/sec)")
gyro_y = st.number_input("Gyro Y (degrees/sec)")
gyro_z = st.number_input("Gyro Z (degrees/sec)")

if st.button("Predict Activity"):
    input_features = np.array([[wrist, acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z]])
    prediction = model.predict(input_features)[0]

    label = "Running" if prediction == 1 else "Walking"
    st.success(f"Predicted Activity: {label}")

