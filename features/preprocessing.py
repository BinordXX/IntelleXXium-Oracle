import pandas as pd

def preprocess_data(input_csv, output_csv):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Select the relevant columns (features + target)
    selected_features = [
        'qty_ordered', 'price', 'value', 'discount_amount', 'total', 
        'category', 'payment_method', 'year', 'month', 'age', 'Discount_Percent', 'order_date'
    ]

    # Filter the dataset to include only the relevant columns
    df_filtered = df[selected_features]

    # Save the filtered dataset to a new CSV file
    df_filtered.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

if __name__ == "__main__":
    # Example usage
    input_csv = "data/sales_data.csv"  # Replace with your input file path
    output_csv = "data/processed_sales_data.csv"   # Replace with your desired output file path
    preprocess_data(input_csv, output_csv)
