import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("models/linear_regression_model.pkl")

# Page Config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

# Title
st.title("🏡 House Price Prediction System")
st.write("Predict California House Price using Linear Regression")

st.markdown("---")

# Input Fields
medinc = st.number_input("Median Income", min_value=0.0, value=3.5)
houseage = st.number_input("House Age", min_value=1, value=20)
averooms = st.number_input("Average Rooms", min_value=1.0, value=5.5)
avebedrooms = st.number_input("Average Bedrooms", min_value=0.1, value=1.0)
population = st.number_input("Population", min_value=1, value=1000)
aveoccup = st.number_input("Average Occupancy", min_value=1.0, value=3.0)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)

# Predict Button
if st.button("Predict House Price"):

    input_data = np.array([[
        medinc,
        houseage,
        averooms,
        avebedrooms,
        population,
        aveoccup,
        latitude,
        longitude
    ]])

    prediction = model.predict(input_data)

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="Estimated House Price",
        value=f"${prediction[0]*100000:,.2f}"
    )