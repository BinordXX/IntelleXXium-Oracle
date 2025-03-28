import pandas as pd

def print_actual_sales(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Ensure 'sales' is in the dataset
    target_column = 'sales'
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Print out the actual sales values (training data)
    print("Actual sales values from the training data:")
    print(df[target_column].head(15))  # Print first 15 values

if __name__ == "__main__":
    print_actual_sales("data/processed_sales_data.csv")

