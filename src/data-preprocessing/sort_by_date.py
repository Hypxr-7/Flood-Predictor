import pandas as pd
import sys

def main(input_csv):
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
    print(f"Sorted CSV saved as: {output_csv}")

if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Usage: python sort_by_month.py yourfile.csv")
    #     sys.exit(1)
    main("data/kpk/compiled/KPK_merged.csv")
