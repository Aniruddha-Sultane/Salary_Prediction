import streamlit as st
import pickle
import numpy as np
import os
print(os.getcwd())

# Load model safely
with open(r"C:\Users\Aniruddha\OneDrive\Desktop\FSDS2\Machine_Learning\Regression\Simple_linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("Salary Prediction App")

st.write("This app predicts salary based on years of experience using Simple Linear Regression.")

# Input
years_experience = st.number_input(
    "Enter Years of Experience:",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.5
)

# Prediction
if st.button("Predict Salary"):
    experience_input = np.array([[years_experience]])
    prediction = model.predict(experience_input)

    st.success(
        f"Predicted salary for {years_experience} years of experience is: ₹{prediction[0]:,.2f}"
    )

st.write("Model built by A S")