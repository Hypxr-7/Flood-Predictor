import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import sys

def main(input_csv):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Parse 'month' as datetime
    df['month'] = pd.to_datetime(df['month'], format='%b %Y')

    print("Initial missing values per column:")
    print(df.isnull().sum())

    # Columns to impute
    cols_to_impute = ['avg_LST_Day_1km', 'avg_NDSI', 'avg_NDVI']

    # Impute
    imputer = IterativeImputer(random_state=42)
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    print("\nMissing values after imputation:")
    print(df.isnull().sum())

    # Convert 'month' column back to desired string format
    df['month'] = df['month'].dt.strftime('%b %Y')

    # Save
    output_csv = input_csv.replace('.csv', '_imputed.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nImputed data saved to: {output_csv}")



if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python impute_with_sklearn.py your_data.csv")
    #     sys.exit(1)

    input_csv = "data/kpk/compiled/KPK_merged.csv"
    main(input_csv)
