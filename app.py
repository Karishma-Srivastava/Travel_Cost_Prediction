# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# load model + preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")  # optional if you used one

st.set_page_config(page_title="Travel Cost Predictor", layout="centered")
st.title("Travel Cost Predictor")

# input widgets (example)
origin = st.text_input("Origin city", "Delhi")
destination = st.text_input("Destination city", "Mumbai")
days = st.number_input("Number of days", min_value=1, value=3)
hotel_type = st.selectbox("Hotel type", ["Budget", "Standard", "Luxury"])
transport = st.selectbox("Transport", ["Bus", "Train", "Flight"])
food_pref = st.selectbox("Food preference", ["Veg", "Non-Veg"])

if st.button("Predict cost"):
    # construct dataframe row - adapt to your features
    row = pd.DataFrame([{
        "origin": origin,
        "destination": destination,
        "days": days,
        "hotel_type": hotel_type,
        "transport": transport,
        "food_pref": food_pref
    }])
    # apply preprocessor if any
    try:
        X = preprocessor.transform(row)
    except:
        X = row  # if model handles raw input
    pred = model.predict(X)
    st.success(f"Estimated total cost: ₹{float(pred[0]):,.2f}")
