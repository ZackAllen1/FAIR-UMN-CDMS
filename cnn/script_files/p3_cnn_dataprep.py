import pandas as pd
import numpy as np
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
    # Order the features Spatially and Temporally
    # Features: Amplitude, Start, Rise, Fall, Width
    # Channels: A, B, C, D, F 
    channels = ["A", "B", "C", "D", "F"]
    timing_suffix = [
        'r10','r20','r30','r40', 'r50', 'r60', 'r70', 'r80', 'r90','r95', 'r100', # rise
        'f95', 'f90', 'f80', 'f40', 'f20' # fall
    ]
    ordered_features = []
    for ch in channels:
        ordered_features.append(f"P{ch}amp")
        for suffix in timing_suffix:
            ordered_features.append(f"P{ch}{suffix}")
        
    x = data[ordered_features].values
    y = data["y"].values
    
    return (x, y, ordered_features)

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
    # Reshape this from [N, 20] to [N, 5, 17] for PyTorch DataLoader
    x_tensor = x_tensor.reshape(-1, 5, 17)
    
    # Now convert this tensor into PyTorch Dataset
    xy_dataset = CDMSDataset(x_tensor, y_array)
    
    # Wrap this into DataLoaders
    xy_loader = DataLoader(xy_dataset, batch_size = batch_size, shuffle = shuffle)
    
    return xy_loader

if __name__ == "__main__":
    
    # load the data as pd.DataFrame
    train_filepath = "../data/full_train_augmented_data.csv"
    valid_filepath = "../data/full_valid_data.csv"
    test_filepath = "../data/full_test_data.csv"
    
    train_data = load_data(train_filepath)
    valid_data = load_data(valid_filepath)
    test_data = load_data(test_filepath)
    
    # get x data and y data from the loaded dataframe, returned as numpy array
    train_x, train_y, _ = features(train_data)
    valid_x, valid_y, _ = features(valid_data)
    test_x, test_y, _ = features(test_data)
    
    # Apply fit scalar on the training set only, then ttransform the rest
    scalar = StandardScaler()
    train_x = scalar.fit_transform(train_x)
    valid_x = scalar.transform(valid_x)
    test_x = scalar.transform(test_x)
    
    # Create DataLoader
    train_loader = create_dataloaders(train_x, train_y, batch_size = 128, shuffle = True)
    valid_loader = create_dataloaders(valid_x, valid_y, batch_size = 128, shuffle = False)
    test_loader = create_dataloaders(test_x, test_y, batch_size = 128, shuffle = False)
    
    print("Train DataLoader Created")
    print("Validation DataLoader Created")
    print("Test DataLoader Created")