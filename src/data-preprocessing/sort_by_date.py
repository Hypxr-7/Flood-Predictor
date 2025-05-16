# This just sorts the data by dates
# Was needed as data became sorted alphabetically by date instead of chronologically

import pandas as pd

input_csv = "data/kpk/compiled/KPK_merged.csv"
df = pd.read_csv(input_csv)

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Convert 'month' column to datetime
df['month'] = pd.to_datetime(df['month'], format='%b %Y')

# Sort by month chronologically
df = df.sort_values(by='month')

# Convert back to 'Mon YYYY' string format if needed
df['month'] = df['month'].dt.strftime('%b %Y')

# Save sorted CSV
output_csv = input_csv.replace('.csv', '_sorted.csv')
df.to_csv(output_csv, index=False)
