import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(filepath: str) -> pd.DataFrame:
    """
    Returns pd.DataFrame
    """
    return pd.read_csv(filepath)

def features(data: pd.DataFrame):
    """
    This function prepares the data with selected columns
    """
    # Order the features Temporally
    # Channels: A, B, C, D, F 
    channels = ["A", "B", "C", "D", "F"]
    timing_suffix = [
        'r10','r20','r30','r40', 'r50', 'r60', 'r70', 'r80', 'r90','r95', 'r100', # rise
        'f95','f90','f80','f40','f20', # fall
    ]
    ordered_features = []
    for ch in channels:
        for suffix in timing_suffix:
            ordered_features.append(f"P{ch}{suffix}")
        
    x = data[ordered_features].values
    y = data["y"].values
    
    return (x, y, ordered_features)

def compute_deltas_with_cross_channel(x: np.ndarray) -> np.ndarray:
    """
    Input [N, 80]
    
    Returns [N, 6, 16]:
      - rows 0-4: per-channel intra-channel deltas (existing behavior, but
                  with index 0 being r10 RELATIVE to channel A's r10)
      - row 5:    cross-channel arrival pattern - each channel's r10 minus
                  channel A's r10. Index 0 is the channel A reference (zero),
                  indices 1-4 are B,C,D,F arrival deltas. Indices 5-15 are
                  zero-padded so the time axis matches.
    """
    
    # get the number of samples
    n = x.shape[0]
    
    # reshape with [Number of samples, Channels, Features]
    x_2d = x.reshape(n, 5, 16)               # [N, 5, 16]

    # Channel A's r10 as the row-wise spatial reference
    a_r10 = x_2d[:, 0, 0:1]                  # [N, 1]: Channel A's r10 column

    # Per-channel intra deltas (existing)
    deltas = np.empty_like(x_2d)
    deltas[:, :, 0] = x_2d[:, :, 0] - a_r10  # spatially-referenced r10
    deltas[:, :, 1:] = np.diff(x_2d, axis=2)

    # Cross-channel arrival vector: each channel's r10 minus A's r10
    arrival = x_2d[:, :, 0] - a_r10          # [N, 5], with channel A = 0
    cross = np.zeros((n, 1, 16), dtype=x_2d.dtype)
    cross[:, 0, :5] = arrival                # rest stays zero

    out = np.concatenate([deltas, cross], axis=1)  # [N, 6, 16]
    return out.reshape(n, 6 * 16)

class CDMSDataset(Dataset):
    """
    Custom PyTorch Dataset for SuperCDMS wave
    """
    def __init__(self, x_tensor, y_array):
        self.x = torch.tensor(x_tensor, dtype = torch.float32)
        self.y = torch.tensor(y_array, dtype = torch.float32).unsqueeze(1) # Reshape to [Batch, 1]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def create_dataloaders(x_tensor, y_array, batch_size = 128, shuffle = True):
    """
    This function creates the DataLoaders for CNN
    """
    # Reshape this from [N, 80] to [N, 5, 16] for PyTorch DataLoader
    # print(x_tensor.shape)
    x_tensor = x_tensor.reshape(-1, 6, 16)
    # print(x_tensor.shape)
    
    # Now convert this tensor into PyTorch Dataset
    xy_dataset = CDMSDataset(x_tensor, y_array)
    
    # Wrap this into DataLoaders
    xy_loader = DataLoader(xy_dataset, batch_size = batch_size, shuffle = shuffle)
    
    return xy_loader

def cnn_main(filepaths):
    
    # Prepare the data:
    train_filepath, valid_filepath, test_filepath = filepaths
    
    print("Loading data...")
    train_data = load_data(train_filepath)
    valid_data = load_data(valid_filepath)
    test_data = load_data(test_filepath)
    
    # get x data and y data from the loaded dataframe, returned as numpy array
    train_x, train_y, _ = features(train_data)
    valid_x, valid_y, _ = features(valid_data)
    test_x, test_y, _ = features(test_data)
    
    # convert the absolute timestamps into deltas
    train_x = compute_deltas_with_cross_channel(train_x)
    valid_x = compute_deltas_with_cross_channel(valid_x)
    test_x = compute_deltas_with_cross_channel(test_x)
    
    # Apply fit scalar on the training set only, then ttransform the rest
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    valid_x = scaler.transform(valid_x)
    test_x = scaler.transform(test_x)
    
    # scale the target y also
    # first reshape to 2D as StandardScalar expects a 2D array
    train_y = np.array(train_y, dtype = np.float32).reshape(-1, 1)
    valid_y = np.array(valid_y, dtype = np.float32).reshape(-1, 1)
    test_y = np.array(test_y, dtype = np.float32).reshape(-1, 1)
    
    scaler_y = StandardScaler()
    train_y = scaler_y.fit_transform(train_y)
    valid_y = scaler_y.transform(valid_y)
    test_y = scaler_y.transform(test_y)
    
    # flatten the train_y, valid_y and test_y back to 1D for the create_dataloaders function
    train_y = train_y.flatten()
    valid_y = valid_y.flatten()
    test_y = test_y.flatten()
    
    # Create DataLoader
    train_loader = create_dataloaders(train_x, train_y, batch_size = 128, shuffle = True)
    valid_loader = create_dataloaders(valid_x, valid_y, batch_size = 128, shuffle = False)
    test_loader = create_dataloaders(test_x, test_y, batch_size = 128, shuffle = False)
    
    print("Train DataLoader Created")
    print("Validation DataLoader Created")
    print("Test DataLoader Created")
    
    return train_loader, valid_loader, test_loader, scaler_y

if __name__ == "__main__":
    
    # define the filepaths to train, valid and test dataset
    train_filepath = "../data/s1_temporal_train_data.csv"
    valid_filepath = "../data/s1_temporal_valid_data.csv"
    test_filepath = "../data/s1_temporal_test_data.csv"
    
    if os.path.exists(train_filepath):
        if os.path.exists(valid_filepath):
            if os.path.exists(test_filepath):
                filepaths = (train_filepath, valid_filepath, test_filepath)
                cnn_main(filepaths=filepaths)
            else:
                print(f"Error: {test_filepath} not found.")
        else:
            print(f"Error: {valid_filepath} not found.")
    else:
        print(f"Error: {train_filepath} not found.")