import pandas as pd
import os


folder_path = 'data/mapping_stock'
csv_files = [f for f in os.listdir(folder_path)]

df_list = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)

output_file = 'data/mapping_stock.csv'
merged_df.to_csv(output_file, index=False)

print(f'Merged {len(csv_files)} CSV files into {output_file}')
