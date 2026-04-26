import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_save_data(filepath: str, output_dir: str):
    # load the data into a pandas DatFrame
    data = pd.read_csv(filepath)
    print(len(data.columns))
    
    # the chronological order for timing markers
    timing_suffix = [
        'r10','r20','r30','r40', 'r50', 'r60', 'r70', 'r80', 'r90','r95', 'r100', # rise
        'f20', 'f40', 'f80', 'f90', 'f95', # fall
    ]
        
    channels = ["A", "B", "C", "D", "F"]
    ordered_cols = ["Row"]
    
    for ch in channels:
        ordered_cols.append(f"P{ch}amp")
        for suffix in timing_suffix:
            ordered_cols.append(f"P{ch}{suffix}")
    
        
    ordered_cols.append("y")
    
    data = data[ordered_cols]
    print(len(data.columns))
        
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
    train_df.to_csv(os.path.join(output_dir, "full_train_data.csv"), index = False)
    valid_df.to_csv(os.path.join(output_dir, "full_valid_data.csv"), index = False)
    hos_df.to_csv(os.path.join(output_dir, "full_test_data.csv"), index = False)
    
    print(f"\n--- Data Saved to {output_dir}---")
    
if __name__ == "__main__":
    # filepath
    filepath = "../data/full_data.csv"
    output_dir = "../data/"
    if os.path.exists(filepath):
        split_save_data(filepath, output_dir)
    else:
        print(f"Error: {filepath} not found.")