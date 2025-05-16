import pandas as pd

# Load the CSV file
input_file = 'data/final/punjab.csv'          # Replace with your filename
output_file = 'data/final/sindh__.csv'      # Output file name

# Read the CSV
df = pd.read_csv(input_file)

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Define columns that may have commas and need to be converted to float
columns_to_clean = ['Avg NDVI']

for col in columns_to_clean:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)                        # Ensure values are strings
            .str.replace(",", "", regex=False)  # Remove commas
            .replace({"": None})                # Convert empty strings to None
            .astype(float)                      # Convert to float
        )

# Save the cleaned DataFrame to a new CSV
df.to_csv(output_file, index=False)

print(f"Cleaned data saved to: {output_file}")
