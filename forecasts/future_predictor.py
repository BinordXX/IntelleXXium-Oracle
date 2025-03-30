import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from keras import losses

# Load model and scaler once
model = load_model("oracle_model.h5", custom_objects={"mse": losses.MeanSquaredError()})
scaler = joblib.load("scaler.pkl")

def preprocess_data(df):
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Keep original dataframe for returning with predictions
    df_original = df.copy()

    # Drop non-numeric columns
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.columns.difference(non_numeric_columns)

    # Clean data: Convert numeric columns to proper types, fill missing values
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    return df_original, df[numeric_columns]

def predict_from_df(df):
    df_original, X = preprocess_data(df)
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)

    # Add predicted sales to the original dataframe
    df_original['predicted_sales'] = predictions.flatten()

    return df_original
