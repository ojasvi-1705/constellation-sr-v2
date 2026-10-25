"""RCAN - Residual Channel Attention Networks"""
import torch
import torch.nn as nn

class RCAN(nn.Module):
    """RCAN Model for Super-Resolution"""
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_rg=10, num_rcab=20):
        super(RCAN, self).__init__()
        
        self.initial = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # Residual groups
        self.rgs = nn.ModuleList([
            self._make_residual_group(num_features, num_rcab)
            for _ in range(num_rg)
        ])
        
        self.conv_before_upsample = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.final = nn.Conv2d(num_features, out_channels, 3, 1, 1)
    
    def _make_residual_group(self, num_features, num_rcab):
        return nn.Sequential(*[
            nn.Conv2d(num_features, num_features, 3, 1, 1)
            for _ in range(num_rcab)
        ])
    
    def forward(self, x):
        initial = self.initial(x)
        out = initial
        for rg in self.rgs:
            out = rg(out) + out
        out = self.conv_before_upsample(out)
        out = self.upsample(out)
        out = self.final(out)
        return out

def create_rcan(pretrained=False):
    """Create RCAN model"""
    model = RCAN(in_channels=3, out_channels=3, num_features=64)
    return model
