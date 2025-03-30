import pandas as pd
import numpy as np
from keras.models import load_model
import joblib

# Load model and scaler once (global load)
from keras import losses

model = load_model("oracle_model.h5", custom_objects={"mse": losses.MeanSquaredError()})
scaler = joblib.load("scaler.pkl")

def preprocess_data(df):
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    
    # Drop non-numeric columns (or handle them as needed)
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.columns.difference(non_numeric_columns)

    # Convert to numeric & fill missing values
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Drop target column if it's present
    if 'sales' in df.columns:
        df = df.drop(columns=['sales'])

    return df[numeric_columns]

def predict_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return predict_from_df(df)

def predict_from_df(df):
    X = preprocess_data(df)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    df['predicted_sales'] = predictions.flatten()
    return df

# Optional testing
if __name__ == "__main__":
    predicted_df = predict_from_csv("data/processed_sales_data.csv")
    print(predicted_df[['predicted_sales']].head())
