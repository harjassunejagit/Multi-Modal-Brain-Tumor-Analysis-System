import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, Fg, Fl, Fint):
        super().__init__()
        self.Wg = nn.Conv3d(Fg, Fint, 1)
        self.Wx = nn.Conv3d(Fl, Fint, 1)
        self.psi = nn.Conv3d(Fint, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        psi = self.relu(self.Wg(g) + self.Wx(x))
        psi = self.sigmoid(self.psi(psi))
        return x * psi