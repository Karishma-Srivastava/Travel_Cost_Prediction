# src/predict.py
"""
CLI wrapper to test the saved model.

Usage (from project root):

python -m src.predict \
    --models_dir models \
    --input_json '{"Source": "Mumbai", "Destination": "Delhi", "Travel_Mode": "Flight", "Travel_Class": "Economy", "Distance_km": 1150, "Fuel_Price": 105, "Days": 3, "Travelers": 1, "Peak_Season": "Yes", "Hotel_Type": "Standard", "Food_Preference": "Veg", "Weather_Sunny_Rainy_Snowy_Cloudy": "Sunny", "Holiday_Type_Festival_Weekend_Vacation_Business": "Vacation", "Discount_Yes_No": "No", "Travel_Insurance_Yes_No": "Yes", "Start_Date": "11-03-2024", "End_Date": "14-03-2024"}'
"""

import argparse
import json
import os
import joblib
import pandas as pd

from src.preprocess import enrich_features


def main(args):
    model_path = os.path.join(args.models_dir, "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")

    # Load trained model
    model = joblib.load(model_path)

    # Convert input JSON → DataFrame
    record = json.loads(args.input_json)
    df = pd.DataFrame([record])

    # Remove Price_Multiplier if present
    if "Price_Multiplier" in df.columns:
        df = df.drop(columns=["Price_Multiplier"])

    # Process all features (dates, categories, etc.)
    df = enrich_features(df)

    # Predict
    pred = model.predict(df)[0]
    print(f"Predicted cost: {pred:.2f} INR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--input_json", type=str, required=True)
    args = parser.parse_args()
    main(args)
