import torch
import torch.nn as nn

class ThermalAttentionHead(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.q = nn.Linear(d_model, head_dim)
        self.k = nn.Linear(d_model, head_dim)
        self.v = nn.Linear(d_model, head_dim)
        
        # Thermal parameters specific to this head
        self.log_beta = nn.Parameter(((torch.rand(1) * 8) - 4).float())
        self.mu = nn.Parameter(2*(torch.randn(1) * 0.1 - 0.05))

    def forward(self, x):
        # x shape: [Batch, Time, d_model]
        query = self.q(x) 
        key = self.k(x)
        value = self.v(x)

        # Dot-product attention scores
        dist = torch.cdist(query, key, p=2) # [Batch, Time, Time]
        scores = -dist

        scores = scores - scores.mean(dim=-1, keepdim=True)
        
        beta = torch.exp(self.log_beta)
        numerator = torch.exp(torch.clamp(beta * (scores - self.mu), min=-50, max=50))
        attn_weights = numerator / (torch.sum(numerator, dim=-1, keepdim=True) + 1e-15)
        
        return torch.matmul(attn_weights, value)


class ThermalTransformerRegressor(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, max_seq_len=500):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 1. Input and Positional Setup
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # NEW: The CLS token that will "summarize" the sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Note: pos_embedding should cover max_seq_len + 1 (for the token)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model))
        
        # 2. Multi-Head setup using Thermal Heads
        self.heads = nn.ModuleList([
            ThermalAttentionHead(d_model, self.head_dim) for _ in range(nhead)
        ])
        
        self.out_projection = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        
        # 3. Regression head targeting the -50 to 0 range
        # Input is now just the d_model (64) from the CLS token
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # 1. Reshape incoming [Batch, 5, 16] -> [Batch, 16, 5]
        x = x.permute(0, 2, 1)
        b, t, f = x.size()
        
        # 2. Project features to embedding space
        x = self.input_projection(x) # [Batch, 16, 64]
        
        # 3. Prepend CLS token
        # Expand the token to match batch size: [Batch, 1, 64]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [Batch, 17, 64]
        
        # 4. Add Positional Embedding
        # t + 1 because we added the token
        x = x + self.pos_embedding[:, :t + 1, :]
        
        # 5. Multi-Head Attention
        head_outputs = [head(x) for head in self.heads]
        x_attn = torch.cat(head_outputs, dim=-1)
        
        # 6. Apply Out Projection + Residual + Norm
        # This brings the information from the heads back into the main stream
        x = self.ln(x + self.out_projection(x_attn))
        
        # 7. Feed-Forward Network (The "Processing" step)
        # We usually add another LayerNorm here for stability
        x = self.ln(x + self.ffn(x))
        cls_output = x[:, 0, :] 
        
        # 8. Final regression
        # Softplus gives (0 to inf). Negative sign makes it (0 to -inf).
        # This aligns with your goal of a -50 to 0 range.
        x = self.regression_head(cls_output)

        return torch.sigmoid(x) * -50.0