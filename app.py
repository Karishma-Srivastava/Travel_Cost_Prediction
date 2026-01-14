import streamlit as st
import pandas as pd
import joblib
import os
from datetime import date, timedelta

from src.preprocess import enrich_features


MODEL_PATH = "models/best_model.joblib"


# ---------------------------------------
# FARE WEIGHT MAPPING (FULLY CORRECT)
# ---------------------------------------
def force_class_weight(df):
    fare_weight_map = {
        "Non-AC Seater": 1.0,
        "AC Seater": 1.4,
        "SL": 1.4,
        "3A": 1.6,
        "2A": 1.8,
        "AC Sleeper": 1.6,
        "Economy": 2.5,
        "Premium Economy": 2.7,
        "Business": 3.5,
        "First": 2.5
    }
    df["Class_Fare_Weight"] = df["Travel_Class"].map(fare_weight_map).fillna(1.0)
    return df


# ---------------------------------------
# MODEL LOADER
# ---------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please train the model first.")
        return None
    return joblib.load(MODEL_PATH)


# ---------------------------------------
# MAIN APP
# ---------------------------------------
def main():
    st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="centered")
    st.title("Travel Cost Prediction (India)")
    st.write("Estimate trip cost based on class, mode, distance, season and other travel factors.")

    model = load_model()
    if model is None:
        return

    # ---------------------------------------
    # TRIP DETAILS
    # ---------------------------------------
    st.subheader("Trip Details")

    col1, col2 = st.columns(2)

    with col1:
        source = st.text_input("Source City", "Mumbai")
        travel_mode = st.selectbox("Travel Mode", ["Train", "Bus", "Flight", "Car"])
        travel_class = st.selectbox(
            "Travel Class",
            [
                "Non-AC Seater",
                "AC Seater",
                "SL",
                "3A",
                "2A",
                "AC Sleeper",
                "Economy",
                "Premium Economy",
                "Business",
                "First"
            ]
        )
        hotel_type = st.selectbox("Hotel Type", ["Budget", "Standard", "Luxury"])
        food_pref = st.selectbox("Food Preference", ["Veg", "Non-Veg", "Vegan"])

    with col2:
        destination = st.text_input("Destination City", "Delhi")
        distance_km = st.number_input("Distance (km)", min_value=1.0, value=500.0)
        fuel_price = st.number_input("Fuel Price (₹/litre)", min_value=50.0, value=100.0)
        days = st.number_input("Trip Days", min_value=1, value=5)
        travelers = st.number_input("Number of Travelers", min_value=1, value=1)

    # ---------------------------------------
    # CONTEXT
    # ---------------------------------------
    st.subheader("Context")

    col3, col4 = st.columns(2)

    with col3:
        peak_season = st.selectbox("Peak Season?", ["No", "Yes"])
        holiday_type = st.selectbox(
            "Holiday Type", ["Vacation", "Business", "Festival", "Weekend"]
        )
        weather = st.selectbox(
            "Weather Condition", ["Sunny", "Rainy", "Cloudy", "Snowy"]
        )

    with col4:
        discount = st.selectbox("Discount Applied?", ["No", "Yes"])
        insurance = st.selectbox("Travel Insurance?", ["No", "Yes"])

    # ---------------------------------------
    # DATES
    # ---------------------------------------
    st.subheader("Travel Dates")

    start_date = st.date_input("Start Date", value=date.today())
    end_date = start_date + timedelta(days=int(days))

    # ---------------------------------------
    # PREDICT BUTTON
    # ---------------------------------------
    if st.button("Predict Cost"):

        # RAW INPUT RECORD
        record = {
            "Source": source,
            "Destination": destination,
            "Distance_km": distance_km,
            "Fuel_Price": fuel_price,
            "Peak_Season": peak_season,
            "Travel_Mode": travel_mode,
            "Hotel_Type": hotel_type,
            "Food_Preference": food_pref,
            "Days": days,
            "Travelers": travelers,
            "Weather_Sunny_Rainy_Snowy_Cloudy": weather,
            "Holiday_Type_Festival_Weekend_Vacation_Business": holiday_type,
            "Discount_Yes_No": discount,
            "Travel_Insurance_Yes_No": insurance,
            "Start_Date": start_date.strftime("%d-%m-%Y"),
            "End_Date": end_date.strftime("%d-%m-%Y"),
            "Travel_Class": travel_class
            # ❌ Price Multiplier removed completely
        }

        df = pd.DataFrame([record])

        # Add class weight FIRST
        df = force_class_weight(df)

        # Then apply preprocessing (dates, encoding, etc.)
        df = enrich_features(df)

        # Predict final cost
        pred = model.predict(df)[0]

        st.success(f"Estimated Trip Cost: ₹ {pred:,.2f}")


if __name__ == "__main__":
    main()
