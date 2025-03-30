import pandas as pd

def inspect_csv_columns(csv_path):
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        print("\n Column names in the CSV:\n")
        for idx, col in enumerate(df.columns):
            print(f"{idx + 1}. '{col}'")
    except FileNotFoundError:
        print(f" File not found at: {csv_path}")
    except Exception as e:
        print(f" An error occurred: {e}")

if __name__ == "__main__":
    # Adjust path if needed
    csv_file_path = "data/feature_engineered_sales_data.csv"
    inspect_csv_columns(csv_file_path)
