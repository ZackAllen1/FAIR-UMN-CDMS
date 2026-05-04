import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import custom modules
from o4_cnn_dataprep import load_data, features, create_dataloaders
from o5_advancedCNN import AdvancedCNN

def train_model(model, train_loader, valid_loader, learning_rate, weight_decay, epochs, device):
    """
    Trains the Advanced CNN model
    """
    
    # Loss function
    criterion = nn.MSELoss()
    # Optimizer: Select Adam
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    # Add a scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min = learning_rate / 100)
    
    best_val_rmse = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------#
        model.train()
        train_sq_errors = []
        
        for batch_x, batch_y in train_loader:
            # Get the data into device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Zero grad
            optimizer.zero_grad()
            predictions = model(batch_x)
            
            # Get loss
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_sq_errors.extend(((predictions - batch_y) ** 2).detach().cpu().numpy().tolist())
        
        scheduler.step()    
        # Calculate Training RMSE
        avg_train_rmse = float(np.sqrt(np.mean(train_sq_errors)))
        
        # ---------- VALIDATION ----------
        model.eval()
        val_sq_errors = []
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_sq_errors.extend(((predictions - batch_y) ** 2).cpu().numpy().tolist())
                
        # Calculate Validation RMSE
        avg_val_rmse = float(np.sqrt(np.mean(val_sq_errors)))
        lr_now = scheduler.get_last_lr()[0]
        
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train RMSE: {avg_train_rmse:.4f} | "
            f"Val RMSE: {avg_val_rmse:.4f} | "
            f"LR: {lr_now:.2e}"
        )
        
        # Implement an early stopage if the avg_val_rmse is not improving over 10 epochs
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            epochs_no_improve = 0
            # Save this best model
            torch.save(model.state_dict(), "best_advanced_cnn.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == 10:
                print(f"\nEarly stopping triggered! No improvement for 10 epochs.")
                break
            
def evaluation_hos(model_path, test_loader, device):
    """
    Evaluates the model on the HOS dataset
    """
    print("\n--- Evaluating on Held-Out Subset (HOS) ---")
    model = AdvancedCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    criterion = nn.MSELoss()
    test_sq_errors = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            test_sq_errors.extend(((predictions - batch_y) ** 2).cpu().numpy().tolist())
            
    avg_test_rmse = float(np.sqrt(np.mean(test_sq_errors)))
    print(f"HOS Test RMSE: {avg_test_rmse:.4f}")
    
    

if __name__ == "__main__":
    
    # Prepare the data:
    print("Prepareing data for training...\n")
    train_filepath = "../data/train_augmented_data.csv"
    valid_filepath = "../data/valid_data.csv"
    test_filepath = "../data/test_data.csv"
    
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
    train_loader = create_dataloaders(train_x, train_y, batch_size = 64, shuffle = True)
    valid_loader = create_dataloaders(valid_x, valid_y, batch_size = 64, shuffle = False)
    test_loader = create_dataloaders(test_x, test_y, batch_size = 64, shuffle = False)
    
    print("Train DataLoader Created")
    print("Validation DataLoader Created")
    print("Test DataLoader Created")
    
    # Initialize the model
    print("Initialize the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AdvancedCNN().to(device)
    
    # Train the model
    learning_rate = 1e-4
    weight_decay = 1e-3
    epochs = 100
    print("Starting traing loop...")
    train_model(model, train_loader, valid_loader, learning_rate, weight_decay, epochs, device)
    
    # Final Evaluation on the HOS data
    evaluation_hos("best_advanced_cnn.pth", test_loader, device)