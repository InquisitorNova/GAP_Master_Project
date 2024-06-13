"""
Author: Kaylen Smith Darnbrook
Date: 2021-06-01
Description: This module contains the helper functions and pytorch modules for the unet module used in the GAP model.
The Unet architecture consists of 
- Inception Blocks
- Attention Blocks
- Residual Blocks
- Upsampling Blocks
- Downsampling Blocks
- Final Convolutional Layer
This Unet is incorporated into the GAP model and is used by GAP to predict the next the position of the photon in the image. 
Instructions: This module is not meant to be run as a standalone script. It is meant to be imported into the GAP model.
"""

# Start by importing relevant libraries:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# Define the Inception Module:
class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(Inception_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=groups)

    def forward(self, x):
        branch_1 = F.relu(self.conv1(x))
        branch_2 = F.relu(self.conv2(x)) 
        branch_3 = F.relu(self.conv3(x))
        return torch.cat([branch_1, branch_2, branch_3], 1)
    
# Define the Experimental InceptionTranspose Module:
class InceptionTranspose_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionTranspose_Block, self).__init__()
        self.Transconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1)
        self.Transconv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.Transconv3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, padding=2)
    
    def forward(self, x):
        branch_1 = F.relu(self.Transconv1(x))
        branch_2 = F.relu(self.Transconv2(x))
        branch_3 = F.relu(self.Transconv3(x))
        return torch.cat([branch_1, branch_2, branch_3], 1)

# Define the Traditional Upsample Module:
def Up_Sample(in_channels, out_channels, mode = "transpose"):
    if "transpose" in mode:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
    
# Define the Attention Module:
class Attention_Block(nn.Module):
    def __init__(self, embeded_dimensions, num_channels, num_heads, dropout):
        super(Attention_Block, self).__init__()
        self.groups = num_channels // num_heads
        self.multihead_attn = nn.MultiheadAttention(embeded_dimensions, num_heads, dropout=dropout)
        self.ffn = nn.Linear(embeded_dimensions, embeded_dimensions)
        self.norm1 = nn.GroupNorm(self.groups, embeded_dimensions)
        self.norm2 = nn.GroupNorm(self.groups, embeded_dimensions)

    def forward(self, x):
        batch_size, C, width, height= x.size()
        x = x.view(batch_size, C, -1)
        X = x.permute(2,0,1)
        attn_output, _ = self.multihead_attn(X, X, X)
        X = X + attn_output
        X = self.norm1(self.ffn(X))

        output = X.permuite(1,2,0).view(batch_size, C, width, height)

        return output
    

# Define the Downsample Module:
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention = False, dropout_rate = 0.2, groups = 1, num_heads = 8):
        super(DownBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention = attention
        self.dropout = dropout_rate
        self.groups = groups
        self.num_heads = num_heads

        self.Inception_conv = Inception_Block(in_channels, out_channels, groups)
        self.Skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)
        self.Bottleneck_conv = nn.Conv2d(in_channels*3, out_channels, kernel_size=1, groups=groups)

        self.pool = nn.MaxPool2d(2, 2)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

        if self.attention:
            self.attention = Attention_Block(out_channels, out_channels, num_heads, dropout_rate)

    def forward(self, x):

        x_skip = self.Skip_conv(x)
        x = self.Inception_conv(x)
        x = nn.GELU(self.Bottleneck_conv(x))

        x = x + x_skip

        if self.attention:
            x = self.attention(x)

        x = self.norm(x)
        x = self.dropout(x)
        x = self.pool(x)

        return x



