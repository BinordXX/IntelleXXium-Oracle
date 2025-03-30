import pandas as pd

# Load the dataset
df = pd.read_csv("data/feature_engineered_sales_data.csv")

# Drop the 'sales' column
df_cleaned = df.drop(columns=['sales'])

# Save the cleaned dataset to a new file
df_cleaned.to_csv("data/feature_engineered_data_no_sales.csv", index=False)

print("Sales column dropped and new dataset saved as 'feature_engineered_data_no_sales.csv'")
