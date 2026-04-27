import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset,random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def get_data(path,channels=5,signal_length=16):
    data = pd.read_csv(path)
    y = data['y'].to_numpy()
    x = data.drop(columns=['Row','y','PAamp','PBamp','PCamp','PDamp','PFamp']).to_numpy().reshape(-1, channels, signal_length)
    return torch.from_numpy(x.copy()).float(),torch.from_numpy(y.copy()).unsqueeze(1)

def hold_out_values(x_data, y_data, values_to_withold, tolerance=0.1, batch_size=32):

    holdout = torch.zeros(y_data.shape[0], dtype=torch.bool)

    held = []
    
    for val in values_to_withold:
        matches = (y_data >= (val - tolerance)) & (y_data <= (val + tolerance))
        if len(matches.shape) > 1:
            matches = matches.squeeze()
            held.append(val)
        holdout = holdout | matches
        
        
    train = ~holdout

    print(f"Held out: {held}")
    
    x_train, y_train = x_data[train], y_data[train]
    x_test, y_test = x_data[holdout], y_data[holdout]
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    full_train_dataset = train_loader.dataset

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    train_val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_loader,train_val_loader,test_loader

def hold_out_percent(x_data,y_data,test_size=0.20,random_state=790234,batch_size=32):

    x_train_ten, x_test_ten, y_train_ten, y_test_ten = train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
    
    x_train = x_train_ten.clone().detach()
    x_test  = x_test_ten.clone().detach()
    y_train = y_train_ten.clone().detach()
    y_test  = y_test_ten.clone().detach()
    
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset  = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    full_train_dataset = train_loader.dataset

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    train_val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader,train_val_loader,test_loader

def load_data(path,holdout=None,batch_size=32,random_state=790234):

    if random_state is None:
        rng = np.random.default_rng()
        random_state = rng.randint(np.iinfo(np.int64))


    x_data, y_data = get_data(path)
    scaler = StandardScaler()

    x_flat = x_data.permute(0, 2, 1).reshape(-1, 5).numpy()
    x_scaled = scaler.fit_transform(x_flat)
    x_data_scaled = x_scaled.reshape(x_data.shape[0], 16, 5)
    x_data = torch.from_numpy(x_data_scaled).permute(0, 2, 1).float()

    if holdout is None:
        train_loader,test_loader = hold_out_percent(x_data,y_data,test_size=0.20,random_state=random_state,batch_size=batch_size)
    else:
        train_loader,test_loader = hold_out_values(x_data, y_data, holdout,0.01,batch_size=batch_size)

    return train_loader,test_loader