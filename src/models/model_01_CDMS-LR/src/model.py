import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import seaborn as sns
import pandas as pd

class CDMS_LR:
    def __init__(self, X_train, y_train, X_test, y_test, x_mean, x_std, y_norm, mode="LR", alpha=1.0):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_norm = y_norm
        if mode == "LR":
            self.LR = LinearRegression()
        if mode == "Ridge":
            self.LR = Ridge(alpha = alpha)
        if mode == "Lasso":
            self.LR = Lasso(alpha = alpha)
    
    
    def get_bias(self):
        return self.LR.intercept_
    
    def set_bias(self, intercept):
        self.LR.intercept_ = intercept
    
    def get_coefficients(self):
        return self.LR.coef_
    
    def set_coefficients(self, coef):
        self.LR.coef_ = coef
        
    def predict(self, x):
        return self.LR.predict(x)
    
    def get_MSE(self, x, ytrue):
        ypred = self.predict(x)
        return np.sqrt(np.mean((ytrue*self.y_norm - ypred*self.y_norm)**2))
    
    # Zack: calculate per-label rmse
    def calculate_per_label_rmse(self, y_test_pred, y_norm, labels):
        per_label_rmse = {}
        for label in labels:
            mask = (self.y_test == label)
            if np.sum(mask) > 0:
                rmse = np.sqrt(np.mean((self.y_test[mask]*y_norm - y_test_pred[mask]*y_norm)**2))
                per_label_rmse[label*y_norm] = rmse
        return per_label_rmse
    
    def do_LR(self, do_plot=True):
        self.LR = self.LR.fit(self.X_train, self.y_train)
    
    def model_eval(self, do_plot=True):
        bias, coeffs = self.get_bias(), self.get_coefficients()
        coeffs = np.append(bias, coeffs)
        names = ['$\\beta_{' +str(i) +'}$' for i in range(len(coeffs))]
        
        y_train_pred = self.predict(self.X_train)
        MSE_train = self.get_MSE(self.X_train, self.y_train)
        
        y_test_pred = self.predict(self.X_test)

        # Zack: calculate per-label RMSE as well as overall RMSE
        labels_norm = [y for y in np.unique(self.y_test).tolist()]
        per_label_rmse = self.calculate_per_label_rmse(y_test_pred, self.y_norm, labels=labels_norm)
        MSE_test = self.get_MSE(self.X_test, self.y_test)
        
        if do_plot:
            print("RMSE from train data = {}".format(MSE_train))
            print("Overall RMSE from test data = {}".format(MSE_test))

            for label, rmse in per_label_rmse.items():
                print("RMSE for y = {:.4f}: {:.4f}".format(label, rmse))
        
            fig = plt.figure(figsize=(8,8))
            plt.scatter(names, coeffs)
            plt.plot(names, [0]*len(coeffs), color='k')
        
            fig = plt.figure(figsize=(8,8))
            plt.scatter(self.y_train*self.y_norm, y_train_pred*self.y_norm, marker='x', s=100, color='b', label='Train')
            plt.scatter(self.y_test*self.y_norm,  y_test_pred*self.y_norm, marker='.', color='r', label='Test')
            plt.plot(self.y_train*self.y_norm, self.y_train*self.y_norm, color='k')
            plt.xlabel("True Value")
            plt.ylabel("Predicted Value")
            plt.legend(fontsize = 20)
            plt.show()
        
        return MSE_train, MSE_test, coeffs