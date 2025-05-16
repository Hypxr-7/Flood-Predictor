import pandas as pd
from sklearn.experimental import enable_iterative_imputer # not need bu still must be imported for proper functionality 

from sklearn.impute import IterativeImputer


input_csv = "data/final/gilgit.csv"
df = pd.read_csv(input_csv)

df.columns = df.columns.str.strip()

# Combine Month + Year into datetime for internal use
df['datetime'] = pd.to_datetime(df['Month'] + ' ' + df['Year'].astype(str), format='%b %Y')

print("Initial missing values per column:")
print(df.isnull().sum())

# Columns to impute
cols_to_impute = ['Avg LST', 'Avg NDSI', 'Avg NDVI', 'Avg Precipitation']

# Imputer setup
imputer = IterativeImputer(random_state=42)
df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Convert datetime back to separate Month and Year
df['Month'] = df['datetime'].dt.strftime('%b')
df['Year'] = df['datetime'].dt.year

# Drop helper column
df = df.drop(columns=['datetime'])

# Save the imputed data
output_csv = input_csv.replace('.csv', '_imputed.csv')
df.to_csv(output_csv, index=False)
