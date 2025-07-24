# mimo_model.py
import torch.nn as nn
import torch

class MIMOJointNet(nn.Module):
    def __init__(self, Nt, Nr, hidden_dim=128):
        super().__init__()
        self.Nt, self.Nr = Nt, Nr
        self.net = nn.Sequential(
            nn.Linear(2*Nr, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2*Nt)
        )

    def forward(self, y):
        y_real_imag = torch.cat([y.real, y.imag], dim=-1)
        out = self.net(y_real_imag)
        out_complex = torch.complex(out[..., :self.Nt], out[..., self.Nt:])
        return out_complex
