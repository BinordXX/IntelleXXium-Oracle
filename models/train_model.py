import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def train_sales_model(csv_path):
    print("🚀 Starting feature engineering process...")

    # Load the data
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    print(f"Initial data shape: {df.shape}")
    print(f"Initial columns: {list(df.columns)}")

    # Drop rows with missing values for simplicity
    df.dropna(inplace=True)
    print(f"Data shape after dropping rows with missing values: {df.shape}")

    # Separate target variable
    if 'sales' not in df.columns:
        raise ValueError("'sales' column not found in dataset.")
    
    X = df.drop(columns=['sales'])
    y = df['sales'].astype(float)  # Convert target to numeric

    # Identify and encode non-numeric features
    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Non-numeric columns before encoding: {non_numeric_cols}")

    label_encoder = LabelEncoder()
    for col in non_numeric_cols:
        X[col] = label_encoder.fit_transform(X[col].astype(str))

    print(f"Shape of X after encoding: {X.shape}")
    print(f"Shape of y after processing: {y.shape}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = RandomForestRegressor(n_estimators=100, verbose=2, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📈 Model Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")

    # Optionally, save the model
    # import joblib
    # joblib.dump(model, 'models/sales_model.pkl')

if __name__ == "__main__":
    train_sales_model("data/processed_sales_data.csv")
