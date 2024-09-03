import os
import json
import pandas as pd

# Define the directory path
directory_path = 'data/sec_submissions'

# Define the columns for the DataFrame
columns = ["cik", "company_name", "ticker"]

# Define the batch size
batch_size = 1000

# Initialize an empty DataFrame
new_csv = pd.DataFrame(columns=columns)

# Get the list of all JSON files
json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

# Process files in batches
for i in range(0, len(json_files), batch_size):
    batch_files = json_files[i:i + batch_size]
    
    for filename in batch_files:
        file_path = os.path.join(directory_path, filename)
        
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(f"Processed file: {filename}")
            
            cik = data.get('cik', '')
            company_name = data.get('company_name', '')
            tickers = data.get('tickers', [])
            
            for ticker in tickers:
                new_row = pd.DataFrame([[cik, company_name, ticker]], columns=columns)
                new_csv = pd.concat([new_csv, new_row], ignore_index=True)
    
    # Save the DataFrame to a CSV file after processing each batch
    path = 'data/mapping_stock/'
    batch_csv_filename = f'mapping_batch_{i // batch_size + 1}.csv'
    new_csv.to_csv(path + batch_csv_filename, index=False)
    print(f"Batch {i // batch_size + 1} saved as {batch_csv_filename}")
    
    # Clear the DataFrame for the next batch
    new_csv = pd.DataFrame(columns=columns)