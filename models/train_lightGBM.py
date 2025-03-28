import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import numpy as np

def train_lightgbm_model(csv_path):
    print("⚡ Training LightGBM Model....")
    
    # Load the data
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.dropna(inplace=True)

    X = df.drop(columns=['sales'])
    y = df['sales'].astype(float)

    # Encode categorical features
    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
    label_encoder = LabelEncoder()
    for col in non_numeric_cols:
        X[col] = label_encoder.fit_transform(X[col].astype(str))

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the LightGBM regressor
    model = lgb.LGBMRegressor(verbose=-1, n_estimators=100)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Output
    print("\n📈 LightGBM Model Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")

if __name__ == "__main__":
    train_lightgbm_model("data/processed_sales_data.csv")
