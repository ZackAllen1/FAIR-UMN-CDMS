import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_save_data(filepath: str, output_dir: str):
    # load the data into a pandas DatFrame
    data = pd.read_csv(filepath)
    
    # the dataset is missing the PAstart columns, probably
    # because it is set as a reference sensor. So I have added the column
    # and set it to zero
    if "PAstart" not in data.columns:
        data["PAstart"] = 0.0
        
    channels = ["A", "B", "C", "D", "F"]
    ordered_cols = ["Row"]
    
    # amplitudes first
    for ch in channels:
        ordered_cols.append(f"P{ch}amp")
    
    for ch in channels:
        ordered_cols.extend([f"P{ch}start", f"P{ch}rise", f"P{ch}fall", f"P{ch}width"])
        
    ordered_cols.extend(["energy", "y"])
    
    data = data[ordered_cols]
        
    # take out the HOS subset [-12.502, -29.5, -41.9]
    hos_positions = [-12.502, -29.5, -41.9]
    hos_df = data[data["y"].isin(hos_positions)].copy()
    mls_df = data[~data["y"].isin(hos_positions)].copy()
    
    # now split the train into train/valid with 80/20 split
    train_df, valid_df = train_test_split(mls_df, test_size = 0.2, random_state = 42, stratify = mls_df["y"])
    
    # print the data shape
    print("\n--- Data Splitting Complete ---")
    print(f"Training set:   {train_df.shape}")
    print(f"Validation set: {valid_df.shape}")
    print(f"HOS (Test) set:  {hos_df.shape}")
    
    
    # save the dataframes as csv
    train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index = False)
    valid_df.to_csv(os.path.join(output_dir, "valid_data.csv"), index = False)
    hos_df.to_csv(os.path.join(output_dir, "test_data.csv"), index = False)
    
    print(f"\n--- Data Saved to {output_dir}---")
    

if __name__ == "__main__":
    # filepath
    filepath = "../data/data_all_v4.csv"
    output_dir = "../data/"
    if os.path.exists(filepath):
        split_save_data(filepath, output_dir)
    else:
        print(f"Error: {filepath} not found. Run 1_txt2csv.py first.")