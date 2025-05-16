# some float values were stored as string 

import pandas as pd

input_file = 'data/final/punjab.csv'
output_file = 'data/final/punjab_.csv'


df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

# columns that may have commas and need to be converted to float
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


df.to_csv(output_file, index=False)
