import streamlit as st
import pickle
import pandas as pd

# Load models
scaler = pickle.load(open("scaler.pkl", "rb"))
nb_model = pickle.load(open("naive_bayes.pkl", "rb"))
lr_model = pickle.load(open("logistic_regression.pkl", "rb"))

st.set_page_config(page_title="Apple Stock Prediction", page_icon="🍏", layout="centered")
st.markdown("<h1 style='color:green;text-align:center;'>🍏 Apple Stock Movement Prediction</h1>", unsafe_allow_html=True)
st.write("Predict whether Apple's stock will go UP or DOWN based on numeric features.")

# Input numeric features
open_val = st.number_input("Open Price", value=0.0)
high_val = st.number_input("High Price", value=0.0)
low_val = st.number_input("Low Price", value=0.0)
close_val = st.number_input("Close Price", value=0.0)
volume_val = st.number_input("Volume", value=0.0)

if st.button("🔮 Predict"):
    # Create input array
    input_data = [[open_val, high_val, low_val, close_val, volume_val]]
    transformed = scaler.transform(input_data)
    
    # Predict
    nb_pred = nb_model.predict(transformed)[0]
    lr_pred = lr_model.predict(transformed)[0]
    
    st.subheader("Predictions:")
    st.success(f"Naive Bayes: {'📈 Stock UP' if nb_pred==1 else '📉 Stock DOWN'}")
    st.info(f"Logistic Regression: {'📈 Stock UP' if lr_pred==1 else '📉 Stock DOWN'}")
