import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    """
    Residual block for 1D convolutions.
    """
    def __init__(self, channels, kernel_size = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding = kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding = kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        
        channels = 5
        base_filter = 64
        
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels = channels, out_channels = base_filter, kernel_size=1),
            nn.BatchNorm1d(base_filter),
            nn.ReLU(),
        )
        
        self.res1 = ResidualBlock1D(base_filter, kernel_size = 3)
        
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels = base_filter, out_channels = base_filter * 2, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(base_filter * 2),
            nn.ReLU()
        )
        
        self.res2 = ResidualBlock1D(base_filter * 2, kernel_size = 3)
        self.res3 = ResidualBlock1D(base_filter * 2, kernel_size = 3)
        
        # Global poolong
        # Concatenate avg and ,ax
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        feat_dim = base_filter * 2 * 2 # avg + max concatenate = 256
        
        # Regression
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        # x shape: [B, 5, 5]
        
        x = self.stem(x)        # [B, 64, 5]
        x = self.res1(x)        # [B, 64, 5]
        x = self.proj(x)        # [B, 128, 5]
        x = self.res2(x)        # [B, 128, 5]
        
        avg = self.gap(x)       # [B, 128, 1]
        mp = self.gmp(x)        # [B, 128, 1]
        x = torch.cat([avg, mp], dim = 1)       # [B, 256, 1]
        
        x = self.fc(x)          # [B, 1]
        
        return x