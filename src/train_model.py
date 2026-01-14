# src/train_model.py
"""
Train models on Final1_india_travel_dataset_realistic.csv and save the best one.
"""

import argparse
import os
import joblib
import json
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from src.preprocess import split_features_target, build_preprocessor, enrich_features


# ---------------------------------------------------------
# Evaluation function (RMSE, R2, MAPE)
# ---------------------------------------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    return {"rmse": rmse, "r2": r2, "mape": mape}


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main(args):
    print(f"Loading: {args.data}")
    df = pd.read_csv(args.data)

    # Apply preprocess (date features, cleaning, etc.)
    df = enrich_features(df)

    # REMOVE Price_Multiplier BEFORE training
    if "Price_Multiplier" in df.columns:
        print("Removing Price_Multiplier from training dataset...")
        df = df.drop(columns=["Price_Multiplier"])

    print("Splitting features/target...")
    X, y, numeric_cols, categorical_cols = split_features_target(df, target=args.target)

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Models
    baseline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    rf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            n_jobs=-1,
            random_state=42
        )),
    ])

    # Train both
    models = {
        "LinearRegression": baseline,
        "RandomForestRegressor": rf,
    }

    results = {}

    for name, m in models.items():
        print(f"\nTraining model: {name}")
        m.fit(X_train, y_train)
        metrics = evaluate(m, X_test, y_test)
        results[name] = metrics
        print(f"Metrics for {name}: {metrics}")

    # Pick best model
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = models[best_name]
    best_metrics = results[best_name]

    print(f"\nBest model: {best_name}")
    print("Best metrics:", best_metrics)

    # Save
    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, "best_model.joblib")
    meta_path = os.path.join(args.models_dir, "best_model_metrics.json")

    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

    with open(meta_path, "w") as f:
        json.dump({
            "best_model": best_name,
            "metrics": best_metrics,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }, f, indent=2)

    print(f"Saved metrics to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/Final1_india_travel_dataset_realistic.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--target", type=str, default="Cost")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
