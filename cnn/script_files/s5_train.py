import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import custom modules
from s3_cnn_dataprep import cnn_main
from s4_simpleCNN import TinyCNN, TwoBranchCNN

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_plot(tracking):
    train_loss = tracking["train_loss"]
    train_rmse = tracking["train_rmse"]
    val_loss = tracking["val_loss"]
    val_rmse = tracking["val_rmse"]
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red', linestyle='--')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../plots/training_results_q5_train.png')
    plt.show()
    
def mixup_batch(x, y, alpha=0.3):
    """
    Standard mixup. alpha ~ 0.2-0.4 is typical for regression;
    higher alpha = more aggressive blending.
    """
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    # Sample lam closer to 0 or 1 (less blending) by clipping if you want
    # to keep some "clean" examples; usually not necessary.
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]



def train_model(model, train_loader, valid_loader, scaler_y, learning_rate, weight_decay, patience, epochs, save_path, device):
    """
    Trains the Simple CNN model
    """
    
    # Loss function
    criterion = nn.SmoothL1Loss()
    # Optimizer: Select Adam
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    # Add a scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    
    tracking = {
        "train_loss": [], "val_loss": [],
        "train_rmse": [], "val_rmse": [],
    }
    
    best_val_rmse = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------#
        model.train()
        train_sq_errors = [] # for rmse
        batch_train_loss = []   # for train loss for the batch
        
        for batch_x, batch_y in train_loader:
            # Get the data into device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x, batch_y = mixup_batch(batch_x, batch_y, alpha=2.1)
            
            # Zero grad
            optimizer.zero_grad()
            predictions = model(batch_x)
            
            # Get loss
            loss = criterion(predictions, batch_y)
            loss.backward()

            optimizer.step()
            
            train_sq_errors.extend(((predictions - batch_y) ** 2).detach().cpu().numpy().tolist())  # square errors for this batch
            batch_train_loss.append(loss.item())    # training loss for this batch
           
        # Calculate Training RMSE
        # now scaler_y.scale_[0] is the standard deviation value. Multiply with the rmse to unscale it.
        avg_train_rmse = float(np.sqrt(np.mean(train_sq_errors))) * float(scaler_y.scale_[0])  # rmse for this epoch
        avg_train_loss = np.mean(batch_train_loss)  # training loss for this epoch
        
        # ---------- VALIDATION ----------
        model.eval()
        val_sq_errors = []
        batch_val_loss = []
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                val_loss = criterion(predictions, batch_y)

                val_sq_errors.extend(((predictions - batch_y) ** 2).cpu().numpy().tolist())
                batch_val_loss.append(val_loss.item())
                
        # Calculate Validation RMSE and Loss
        avg_val_rmse = float(np.sqrt(np.mean(val_sq_errors))) * float(scaler_y.scale_[0])
        avg_val_loss = np.mean(batch_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # track all the rmse and losses for all the epochs
        tracking["train_loss"].append(avg_train_loss)
        tracking["train_rmse"].append(avg_train_rmse)
        tracking["val_loss"].append(avg_val_loss)
        tracking["val_rmse"].append(avg_val_rmse)
        
        # get the last LR
        lr_now = optimizer.param_groups[0]["lr"]
        
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train RMSE: {avg_train_rmse:.4f} mm | "
            f"Val RMSE: {avg_val_rmse:.4f} mm | "
            f"LR: {lr_now:.2e}"
        )
        
        # Implement an early stopage if the avg_val_rmse is not improving over 10 epochs
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            epochs_no_improve = 0
            # Save this best model
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                break
    
    save_plot(tracking)
    print(f"\nBest Val RMSE: {best_val_rmse:.4f} mm")
            
def evaluation_hos(model_path, test_loader, scaler_y, device):
    """
    Evaluates the model on the HOS dataset
    """
    print("\n--- Evaluating on Held-Out Subset (HOS) ---")
    model = TwoBranchCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    criterion = nn.SmoothL1Loss()
    test_sq_errors = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)

            test_sq_errors.extend(((predictions - batch_y) ** 2).cpu().numpy().tolist())
            
    avg_test_rmse = float(np.sqrt(np.mean(test_sq_errors))) * float(scaler_y.scale_[0])
    print(f"HOS Test RMSE: {avg_test_rmse:.4f} mm")
    
    # inside evaluation_hos, after collecting predictions
    # all_preds, all_targets = [], []
    # with torch.no_grad():
    #     for batch_x, batch_y in test_loader:
    #         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #         preds = model(batch_x)
    #         all_preds.append(preds.cpu().numpy())
    #         all_targets.append(batch_y.cpu().numpy())

    # preds = np.concatenate(all_preds).flatten() * scaler_y.scale_[0] + scaler_y.mean_[0]
    # targets = np.concatenate(all_targets).flatten() * scaler_y.scale_[0] + scaler_y.mean_[0]

    # for pos in [-12.502, -29.5, -41.9]:
    #     mask = np.isclose(targets, pos)
    #     rmse = np.sqrt(np.mean((preds[mask] - targets[mask])**2))
    #     bias = np.mean(preds[mask] - targets[mask])
    #     print(f'pos={pos:>8.3f} | n={mask.sum():>4d} | '
    #         f'RMSE={rmse:>6.3f} mm | bias={bias:+.3f} mm')

if __name__ == "__main__":
    
    set_seed(seed = 42)
    
    # define the filepaths
    # define the filepaths to train, valid and test dataset
    train_filepath = "../data/s1_temporal_train_data.csv"
    valid_filepath = "../data/s1_temporal_valid_data.csv"
    test_filepath = "../data/s1_temporal_test_data.csv"
    
    filepaths = (train_filepath, valid_filepath, test_filepath)
    
    train_loader, valid_loader, test_loader, scaler_y = cnn_main(filepaths)
    
    # Initialize the model
    print("Initialize the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = 0.2
    print(f"Using device: {device}")
    
    
    model = TwoBranchCNN(dropout=dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    learning_rate = 6e-4
    weight_decay = 9e-5
    epochs = 500
    patience = 20
    model_path = "best_advanced_cnn_temporal_delta.pth"
    print("Starting traing loop...")
    print("Hyperparameters for this traing:")
    print(f"LR: {learning_rate} | WD: {weight_decay} | epochs: {epochs} | dropout: {dropout} | patience = {patience}")
    train_model(model, train_loader, valid_loader, scaler_y, learning_rate, weight_decay, patience, epochs, model_path, device)
    
    # Final Evaluation on the HOS data
    evaluation_hos(model_path, test_loader, scaler_y, device)