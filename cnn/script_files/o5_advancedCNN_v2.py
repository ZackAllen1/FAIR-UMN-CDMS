import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation block for 1D. 
    Helps the model focus on the most relevant sensors.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock1D_v3(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock1D(channels) # Added Attention
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # Apply attention before the skip connection
        return F.relu(out + residual)

class AdvancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Multi-Scale Stem: Captures local and global relationships immediately
        self.stem1 = nn.Conv1d(5, 32, kernel_size=1)
        self.stem3 = nn.Conv1d(5, 32, kernel_size=3, padding=1)
        self.stem5 = nn.Conv1d(5, 32, kernel_size=5, padding=2)
        self.stem_bn = nn.BatchNorm1d(96)
        
        base_filter = 96
        
        self.res1 = ResidualBlock1D_v3(base_filter, kernel_size=3)
        
        self.proj = nn.Sequential(
            nn.Conv1d(base_filter, base_filter * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filter * 2),
            nn.ReLU()
        )
        
        self.res2 = ResidualBlock1D_v3(base_filter * 2, kernel_size=3)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # Expanded Regression Head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filter * 2 * 2, 256), # Increased capacity
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        x1 = self.stem1(x)
        x3 = self.stem3(x)
        x5 = self.stem5(x)
        x = torch.cat([x1, x3, x5], dim=1)
        x = F.relu(self.stem_bn(x))
        
        x = self.res1(x)
        x = self.proj(x)
        x = self.res2(x)
        
        avg = self.gap(x)
        mp = self.gmp(x)
        x = torch.cat([avg, mp], dim=1)
        
        return self.fc(x)