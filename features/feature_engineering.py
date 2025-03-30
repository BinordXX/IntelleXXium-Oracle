import pandas as pd
print("Starting feature engineering process...")

def preprocess_sales_data(df):
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if 'order_date' not in df.columns:
        raise ValueError("Expected column 'order_date' not found. Check your CSV headers.")

    # Clean and parse
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    
    # Check for missing values after parsing
    print("After preprocessing, missing values in 'order_date':", df['order_date'].isna().sum())
    
    df = df.dropna(subset=['order_date'])  # Drop rows where 'order_date' is missing

    # Convert to numeric, check for missing values after conversion
    df['qty_ordered'] = pd.to_numeric(df['qty_ordered'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Check for missing values after conversion
    print("After conversion, missing values in 'qty_ordered':", df['qty_ordered'].isna().sum())
    print("After conversion, missing values in 'price':", df['price'].isna().sum())

    df = df.dropna(subset=['qty_ordered', 'price'])  # Drop rows with missing values in 'qty_ordered' or 'price'

    # Check missing values after dropping rows
    print("After dropping rows with missing values, remaining rows:", len(df))

    df['sales'] = df['qty_ordered'] * df['price']
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['month'] = df['order_date'].dt.month
    df['day'] = df['order_date'].dt.day
    df['hour'] = df['order_date'].dt.hour
    df['weekday'] = df['order_date'].dt.weekday
    df['year'] = df['order_date'].dt.year
    
    # Check for missing values after creating time features
    print("After creating time features, missing values:", df.isna().sum())
    
    return df


def create_lag_features(df: pd.DataFrame, target: str = 'sales') -> pd.DataFrame:
    df = df.sort_values('order_date')
    df[f'{target}_lag_7'] = df[target].shift(7)
    df[f'{target}_rolling_mean_7'] = df[target].shift(1).rolling(window=7).mean()
    
    # Check for missing values after creating lag features
    print("After creating lag features, missing values:", df.isna().sum())

    # Drop rows with missing values after creating lag features
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Load the preprocessed CSV file (processed_sales_data.csv)
    df = pd.read_csv("data/processed_sales_data.csv")
    
    # Preprocess and feature engineering
    df = preprocess_sales_data(df)
    print("Data after preprocessing:", df.head())  # Optionally check the result
    
    df = create_time_features(df)
    print("Data after time feature creation:", df.head())  # Check the time features

    df = create_lag_features(df, target='sales')
    print("Data after lag feature creation:", df.head())  # Check the lag features
    
    # After feature engineering, save the new dataset
    df.to_csv("data/feature_engineered_sales_data.csv", index=False)
    print("Feature-engineered data saved as 'feature_engineered_sales_data.csv'.")

