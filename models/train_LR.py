from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def train_sales_model_with_linear_regression(csv_path):
    print("Starting feature engineering process...")

    # Load the data
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Drop rows with missing values for simplicity
    df.dropna(inplace=True)
    
    # After dropping rows with missing values
    print(f"Data shape after dropping rows with missing values: {df.shape}")

    # Feature selection (X) and target variable (y)
    X = df.drop(columns=['sales'])  # Assuming 'sales' is your target variable
    y = df['sales']
    
    # Check for non-numeric columns and convert them
    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Non-numeric columns before encoding: {non_numeric_cols}")

    # Encode non-numeric columns using Label Encoding
    label_encoder = LabelEncoder()
    for col in non_numeric_cols:
        X[col] = label_encoder.fit_transform(X[col].astype(str))
    
    # Print out the shapes of X and y to check if they are correct
    print(f"Shape of X after encoding: {X.shape}")
    print(f"Shape of y after processing: {y.shape}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate the metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE is just sqrt of MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Linear Regression Model Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    # Specify the path to your processed sales data
    train_sales_model_with_linear_regression("data/processed_sales_data.csv")
