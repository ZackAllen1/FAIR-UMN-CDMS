import torch
import torch.nn as nn
import torch.nn.functional as F
    
class TwoBranchCNN(nn.Module):
    """
    Input:  [B, 6, 16]   (5 detector timing rows + 1 spatial row)
    Output: [B, 1]
    Params: ~3-4k, comparable to TinyCNN.
    """
    def __init__(self, dropout: float = 0.2):
        super().__init__()

        # This is the pulse encoder: shared encoder for ONE detector's 16 timing markers.
        # apply to all 5 detectors with weight sharing.
        self.pulse_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(2),     # [B*5, 16, 2]
            nn.Flatten(),                 # [B*5, 32]
            nn.Linear(64, 16),
            nn.ReLU(),
        )

        # This is the spatial branch: takes the 5-element arrival vector
        self.spatial_encoder = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        # Regression: 5 detectors * 16 + spatial 16 = 96
        self.head = nn.Sequential(
            nn.Linear(5 * 16 + 16, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 6, 16]
        B = x.size(0)

        # Pulse branch: reshape so each detector's timing sequence is
        # treated as a separate batch item by the shared encoder.
        pulses = x[:, :5, :]                        # [B, 5, 16]
        pulses = pulses.reshape(B * 5, 1, 16)       # [B*5, 1, 16]
        pulse_emb = self.pulse_encoder(pulses)      # [B*5, 16]
        pulse_emb = pulse_emb.reshape(B, 5 * 16)    # [B, 80]

        # Spatial branch: take just the meaningful first 5 entries
        spatial = x[:, 5, :5]                       # [B, 5]
        spatial_emb = self.spatial_encoder(spatial) # [B, 16]

        # Fuse and regress
        fused = torch.cat([pulse_emb, spatial_emb], dim=1)  # [B, 96]
        return self.head(fused)