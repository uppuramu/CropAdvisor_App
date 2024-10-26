import streamlit as st
import numpy as np
import pandas as pd
import joblib  
from tensorflow.keras.models import load_model

# Load the scaler and the trained model
scaler = joblib.load('crop_recommendation_scaler.joblib')
model = load_model('crop_recommendation_model.h5')

# Define class names
class_names = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
               'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
               'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
               'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']

# Function to predict crops based on user input
def predict_crops(N, P, K, temperature, humidity, ph, rainfall):
    new_input = [[N, P, K, temperature, humidity, ph, rainfall]]
    new_input_scaled = scaler.transform(new_input)
    pred_probs = model.predict(new_input_scaled)
    top_3_indices = np.argsort(pred_probs.flatten())[-3:][::-1]
    top_3_dict = {class_names[i]: round(pred_probs[0][i] * 100, 2) for i in top_3_indices}
    return top_3_dict

# Streamlit app code
st.title("Crop Recommendation System")
st.write("Enter the soil and climate conditions to predict the most suitable crops.")

# Arrange input fields in two columns and four rows
col1, col2 = st.columns(2)

with col1:
    N = st.number_input('Nitrogen (N) content in soil', min_value=0, max_value=200, value=None)
    K = st.number_input('Potassium (K) content in soil', min_value=0, max_value=200, value=None)
    ph = st.number_input('pH value of the soil', min_value=0.0, max_value=14.0, value=None)
    rainfall = st.number_input('Rainfall (in mm)', min_value=0.0, max_value=500.0, value=None)

with col2:
    P = st.number_input('Phosphorous (P) content in soil', min_value=0, max_value=200, value=None)
    temperature = st.number_input('Temperature (in Celsius)', min_value=0.0, max_value=50.0, value=None)
    humidity = st.number_input('Humidity (in %)', min_value=0.0, max_value=100.0, value=None)

if st.button("Predict"):
    if None not in [N, P, K, temperature, humidity, ph, rainfall]:
        top_3_dict = predict_crops(N, P, K, temperature, humidity, ph, rainfall)
        colors = ['#32CD32', '#FFA500', '#FF4500']
        st.title("Top 3 Recommended Crops:")
        for (key, value), color in zip(top_3_dict.items(), colors):
            st.markdown(f"<p style='color:{color}; font-size:20px;'>Crop: {key}, Probability: {value}%</p>", unsafe_allow_html=True)
    else:
        st.error("Please fill in all input fields.")
