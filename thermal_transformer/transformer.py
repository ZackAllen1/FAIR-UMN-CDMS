import torch
import torch.nn as nn

class ThermalAttentionHead(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.q = nn.Linear(d_model, head_dim)
        self.k = nn.Linear(d_model, head_dim)
        self.v = nn.Linear(d_model, head_dim)
        
        # Allow for wide range of betas.
        self.log_beta = nn.Parameter(((torch.rand(1) * 8) - 4).float())
        self.mu = nn.Parameter(2*(torch.randn(1) * 0.1 - 0.05))

    def forward(self, x):
        query = self.q(x) 
        key = self.k(x)
        value = self.v(x)

        # I use L2 norm (distance) as a way to create preliminary scores that go into
        # the custom softmax.
        dist = torch.cdist(query, key, p=2)
        scores = -dist

        scores = scores - scores.mean(dim=-1, keepdim=True)
        
        # Custom softmax derived from Maxwell-Boltzmann statistics. Beta is analogous to inverse
        # Temperature and mu is analogous to (chemical) potential energy.
        beta = torch.exp(self.log_beta)
        # Have to clamp the bounds of the exponent or we can get NAN gradients.
        numerator = torch.exp(torch.clamp(beta * (scores - self.mu), min=-50, max=50))
        # Classic attention
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
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model))
        #Multi head attention with one head for each input signal sequence.
        self.heads = nn.ModuleList([
            ThermalAttentionHead(d_model, self.head_dim) for i in range(nhead)
        ])
        
        self.out_projection = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        b, t, f = x.size()
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :t + 1, :]
        
        head_outputs = [head(x) for head in self.heads]
        x_attn = torch.cat(head_outputs, dim=-1)
        
        x = self.ln(x + self.out_projection(x_attn)) # Added residual connection
        
        x = self.ln(x + self.ffn(x))
        cls_output = x[:, 0, :] 
        
        x = self.regression_head(cls_output)

        # Enforce our position range, it must stay negative. 
        # Sigmoid is effectively a fermi-dirac dist, so combinaiton of heads is almost
        # like a energy being passed to FD.
        return torch.sigmoid(x) * -50.0