import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_nn_model(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Inspect missing data
    print("Missing data per column before conversion:")
    print(df.isnull().sum())

    # Identify columns that should remain as non-numeric (e.g., strings, categorical data)
    non_numeric_columns = df.select_dtypes(include=['object']).columns

    # Convert numeric columns, excluding non-numeric ones
    numeric_columns = df.columns.difference(non_numeric_columns)
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Check the data types after conversion
    print("Data types after conversion:")
    print(df.dtypes)

    # Handle missing values by imputing (filling with median for numeric columns)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Recheck the missing values after imputation
    print("Missing data after imputation:")
    print(df.isnull().sum())

    # Drop any rows with missing values in non-numeric columns if needed
    df = df.dropna(subset=non_numeric_columns)

    # Check the shape after cleaning
    print(f"Data shape after cleaning: {df.shape}")

    # Ensure there are valid rows left after cleaning
    if df.shape[0] == 0:
        raise ValueError("No valid data left after cleaning. Please inspect the data.")

    # Starting feature engineering process
    print("Starting feature engineering process...")

    # Example: Suppose 'sales' is the target column (you can replace this with your actual target)
    target_column = 'sales'  
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    # Define the feature columns (exclude the target and non-numeric columns)
    feature_columns = df.columns.difference([target_column] + list(non_numeric_columns))

    # Prepare the features (X) and target (y)
    X = df[feature_columns].values
    y = df[target_column].values

    # Scale the data if necessary (optional, but good practice for neural networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define a simple neural network model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # For regression (linear output)
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    print("Training the neural network model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the validation data
    val_loss = model.evaluate(X_val, y_val)
    print(f"Validation loss: {val_loss}")

    # Make predictions on the validation set
    y_pred = model.predict(X_val)
    print("Predictions on validation set:")
    print(y_pred[:5])  # Show the first few predictions

    # Calculate MAPE (Mean Absolute Percentage Error) with handling divide-by-zero
    non_zero_indices = y_val != 0  # Only include non-zero actual values for MAPE calculation
    y_val_non_zero = y_val[non_zero_indices]
    y_pred_non_zero = y_pred.flatten()[non_zero_indices]

    # Calculate MAPE only on non-zero values
    if len(y_val_non_zero) == 0:
        print("Warning: No valid non-zero values found in the target for MAPE calculation.")
        mape = float('inf')  # Set MAPE to infinity if no valid non-zero values
    else:
        mape = np.mean(np.abs((y_val_non_zero - y_pred_non_zero) / y_val_non_zero)) * 100
    
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Calculate R² (Coefficient of Determination)
    r2 = r2_score(y_val, y_pred)
    print(f"R² (Coefficient of Determination): {r2:.4f}")

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

if __name__ == "__main__":
    train_nn_model("data/processed_sales_data.csv")
