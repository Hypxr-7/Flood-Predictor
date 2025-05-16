import csv
import os
from collections import defaultdict

input_files = [
    'data/gilgit-kashmir/compiled/LST_monthly_avg.csv',
    'data/gilgit-kashmir/compiled/NDSI_monthly_avg.csv',
    'data/gilgit-kashmir/compiled/NDVI_monthly_avg.csv',
    'data/gilgit-kashmir/compiled/Prec_monthly_avg.csv'
]

# Data dictionary: key = month, value = dict of {column_name: value}
merged_data = defaultdict(dict)
column_names = []

for file in input_files:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        date_col = header[0]
        value_col = header[1]

        column_names.append(value_col)

        for row in reader:
            month = row[0].strip()
            value = row[1].strip()
            merged_data[month][value_col] = value

# Write merged output
output_file = 'merged_monthly.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['month'] + column_names)

    for month in sorted(merged_data.keys(), key=lambda d: (int(d[-4:]), d[:-5])):
        row = [month] + [merged_data[month].get(col, '') for col in column_names]
        writer.writerow(row)

print(f"Merged file written to: {output_file}")
