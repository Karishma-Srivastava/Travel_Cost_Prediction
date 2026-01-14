# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        c.strip()
         .replace(" ", "_")
         .replace("(", "")
         .replace(")", "")
         .replace(",", "")
         .replace("/", "_")
         .replace("-", "_")
        for c in df.columns
    ]
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Start_Date" in df.columns:
        df["Start_Date"] = pd.to_datetime(df["Start_Date"], format="%d-%m-%Y", errors="coerce")

    if "End_Date" in df.columns:
        df["End_Date"] = pd.to_datetime(df["End_Date"], format="%d-%m-%Y", errors="coerce")

    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_dates(df)

    if "Start_Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Start_Date"]):
        df["Start_Month"] = df["Start_Date"].dt.month
        df["Start_Year"] = df["Start_Date"].dt.year
        df["Start_DayofWeek"] = df["Start_Date"].dt.dayofweek

    if "Start_Date" in df.columns and "End_Date" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["Start_Date"]) and pd.api.types.is_datetime64_any_dtype(df["End_Date"]):
            df["Trip_Duration"] = (df["End_Date"] - df["Start_Date"]).dt.days

    return df


# ------------------------------------------
# NEW: Remove Price_Multiplier from ALL data
# ------------------------------------------
def remove_price_multiplier(df: pd.DataFrame) -> pd.DataFrame:
    if "Price_Multiplier" in df.columns:
        df = df.drop(columns=["Price_Multiplier"])
    return df


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)
    df = add_date_features(df)
    df = remove_price_multiplier(df)   # ← FIX applied here
    return df


def split_features_target(df: pd.DataFrame, target: str = "Cost"):
    df = enrich_features(df)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    # Drop raw date columns before modeling
    drop_cols = ["Start_Date", "End_Date"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target].astype(float)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    num_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    cat_pipe = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor
