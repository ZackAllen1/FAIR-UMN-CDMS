import numpy as np
import pandas as pd
from sklearn.model_selection import *


class CDMS_DataLoader:
    def __init__(self, loc, with_amp=True, sep=','):
        if isinstance(loc, list):
            data_list = []
            for l in loc:
                data_list.append(pd.read_csv(l))
            self.data = pd.concat(data_list)
        else:
            self.data = pd.read_csv(loc)

        # drop any columns that contain 'amp'
        if with_amp is False:
            amp_cols = self.data.filter(like='amp').columns
            print(f"Dropping {len(amp_cols)} columns: {amp_cols}")
            self.data = self.data.drop(amp_cols, axis=1)

        self.features = list(self.data.columns)[1:]
        self.x_data = self.data.values[:,1:-1]
        self.y_data = self.data.values[:, -1]
        self.n_features = len(self.features)
        self.y_norm = -41.9
        
    def split_data(self, split_type, labels=[]):
        if split_type == 'random':
            X_train, X_test, y_train, y_test = train_test_split(self.x_data, self.y_data, test_size=0.20, random_state=42)
        if split_type == 'label':
            X_train = self.data.loc[~self.data[self.features[-1]].isin(labels)].values[:, 1:-1]
            X_test = self.data.loc[self.data[self.features[-1]].isin(labels)].values[:, 1:-1]
            y_train =  self.data.loc[~self.data[self.features[-1]].isin(labels)].values[:, -1]
            y_test = self.data.loc[self.data[self.features[-1]].isin(labels)].values[:, -1]
        
        return X_train, X_test, y_train, y_test
    
    def normalize_data(self, X_train, X_test, y_train, y_test, x_mean = [], x_std = [], y_norm = []):
        if x_mean == []:
            x_mean = np.mean(X_train, axis = 0)
        if x_std == []:
            x_std  = np.std(X_train, axis = 0)
            x_std[x_std == 0] = 1 
        if y_norm == []:
            y_norm = self.y_norm
        X_train = (X_train - x_mean)/x_std
        X_test  = (X_test - x_mean)/x_std
        y_train = y_train/y_norm
        y_test  = y_test/y_norm
    
        return X_train, X_test, y_train, y_test, x_mean, x_std
        