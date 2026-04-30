import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob

def split_save_data(filepath: str, output_dir: str, augmented_dir: str):
    # load the data into a pandas DatFrame
    data = pd.read_csv(filepath)
    # print(len(data.columns))
    
    # the chronological order for timing markers
    timing_suffix = [
        'r10','r20','r30','r40', 'r50', 'r60', 'r70', 'r80', 'r90','r95', 'r100', # rise
        'f95','f90','f80','f40','f20', # fall
    ]
        
    channels = ["A", "B", "C", "D", "F"]
    ordered_cols = []
    
    for ch in channels:
        for suffix in timing_suffix:
            ordered_cols.append(f"P{ch}{suffix}")
    
        
    ordered_cols.append("y")
    
    data = data[ordered_cols]
    print(f"Filtered the full dataset to time-only features. New shape: {data.shape} (80 features + 1 target)")
    # print(len(data.columns))
        
    # take out the Held-Out Subset (HOS) subset [-12.502, -29.5, -41.9]
    hos_positions = [-12.502, -29.5, -41.9]
    hos_df = data[data["y"].isin(hos_positions)].copy()
    mls_df = data[~data["y"].isin(hos_positions)].copy()
    
    # create another subset for valid_positions [-17.992]
    valid_positions = [-17.992]
    train_df = mls_df[~mls_df["y"].isin(valid_positions)].copy()
    valid_df = mls_df[mls_df["y"].isin(valid_positions)].copy()
    
    # Now to call the augmented data from the augmented directory and join with the training df.
    # I will take out the hos_positions and valid_positions from the augmented files before joining.
    # this way the training df won't be seeing any hos and valid data
    print(f"------ Training Augmentation ------")
    train_pieces = [train_df]
    aug_files = sorted(glob.glob(os.path.join(augmented_dir, "augmented_*.csv")))
    print(f"\nFound {len(aug_files)} augmented files")
    for path in aug_files:
        aug = pd.read_csv(path)
        aug = aug[ordered_cols]
        aug = aug[~aug["y"].isin(hos_positions)]
        aug = aug[~aug["y"].isin(valid_positions)]
        train_pieces.append(aug)
        
    train_combined = pd.concat(train_pieces, ignore_index = True)
    
    
    # print the data shape
    print("\n--- Data Splitting Complete ---")
    print(f"Training set:   {train_combined.shape}")
    print(f"Validation set: {valid_df.shape}")
    print(f"HOS (Test) set:  {hos_df.shape}")
    
    
    # save the dataframes as csv
    train_combined.to_csv(os.path.join(output_dir, "s1_temporal_train_data.csv"), index = False)
    valid_df.to_csv(os.path.join(output_dir, "s1_temporal_valid_data.csv"), index = False)
    hos_df.to_csv(os.path.join(output_dir, "s1_temporal_test_data.csv"), index = False)
    
    print(f"\n--- Data Saved to {output_dir}---")
    
if __name__ == "__main__":
    # filepath
    filepath = "../data/data_rearranged_copy.csv"
    output_dir = "../data/"
    augmented_dir = "../data/augmented/"
    if os.path.exists(filepath):
        split_save_data(filepath, output_dir, augmented_dir)
    else:
        print(f"Error: {filepath} not found.")