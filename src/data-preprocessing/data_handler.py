import csv
import os
from collections import defaultdict
from datetime import datetime

# List of original input files
input_files = [
    'data/gilgit-kashmir/compiled/LST.csv',
    'data/gilgit-kashmir/compiled/NDSI.csv',
    'data/gilgit-kashmir/compiled/NDVI.csv',
    'data/gilgit-kashmir/compiled/Prec.csv'
]

for input_file in input_files:
    # ------------------------------
    # Step 1: Clean the file
    # ------------------------------
    base, ext = os.path.splitext(input_file)
    cleaned_file = f"{base}_cleaned{ext}"

    with open(input_file, 'r') as infile, open(cleaned_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)  # Get header
        writer.writerow(header)  # Write header to cleaned file

        for row in reader:
            if len(row) >= 2 and row[1].strip():
                writer.writerow(row)

    print(f"Cleaned: {input_file} → {cleaned_file}")

    # ------------------------------
    # Step 2: Calculate monthly averages
    # ------------------------------
    monthly_avg_file = f"{base}_monthly_avg{ext}"

    # Determine column name from cleaned header
    with open(cleaned_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        variable_name = header[1] if len(header) > 1 else 'value'

    # Aggregate values by month
    monthly_data = defaultdict(list)
    with open(cleaned_file, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header again
        for row in reader:
            date_str, value = row
            
            value = float(value.replace(',', ''))

            date = datetime.strptime(date_str.strip('"'), "%b %d, %Y")
            monthly_data[(date.year, date.month)].append(float(value))

    # Write monthly averages
    with open(monthly_avg_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["month", f"avg_{variable_name}"])
        for (year, month), values in sorted(monthly_data.items()):
            avg = sum(values) / len(values)
            month_str = datetime(year, month, 1).strftime("%b %Y")
            writer.writerow([month_str, round(avg, 3)])

    print(f"Monthly average written: {monthly_avg_file}")
