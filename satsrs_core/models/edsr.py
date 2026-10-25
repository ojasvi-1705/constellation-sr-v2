"""EDSR - Enhanced Deep Residual Networks"""
import torch
import torch.nn as nn

class EDSR(nn.Module):
    """EDSR Model for Super-Resolution"""
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=32):
        super(EDSR, self).__init__()
        
        self.initial = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Conv2d(num_features, num_features, 3, 1, 1)
            for _ in range(num_blocks)
        ])
        
        self.conv_after_blocks = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.final = nn.Conv2d(num_features, out_channels, 3, 1, 1)
    
    def forward(self, x):
        initial = self.initial(x)
        out = initial
        for block in self.blocks:
            out = block(out) + out
        out = self.conv_after_blocks(out) + initial
        out = self.upsample(out)
        out = self.final(out)
        return out

def create_edsr(pretrained=False):
    """Create EDSR model"""
    model = EDSR(in_channels=3, out_channels=3, num_features=64, num_blocks=32)
    return model
