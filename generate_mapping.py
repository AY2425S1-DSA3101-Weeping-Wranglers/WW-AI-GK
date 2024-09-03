import os
import json
import pandas as pd

# Define the directory path
directory_path = 'data/sec_submissions'

columns = ["cik", "company_name", "ticker"]
new_csv = pd.DataFrame(columns=columns)

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        file_path = os.path.join(directory_path, filename)
        
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(f"Processed file: {filename}")
            
            cik = data['cik']
            company_name = data['company_name']
            tickers = data['tickers']
            
            for ticker in tickers:
                new_row = pd.DataFrame([[cik, company_name, ticker]], columns=columns)
                new_csv = pd.concat([new_csv, new_row], ignore_index=True)

# Save the DataFrame to a CSV file
new_csv.to_csv('data/mapping_stock.csv', index=False)

print("DataFrame saved as data/mapping_stock.csv")