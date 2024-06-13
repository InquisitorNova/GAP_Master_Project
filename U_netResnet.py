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
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        self.ffn = None
        self.norm1 = nn.LayerNorm(num_channels)

    def forward(self, x):
        batch_size, C, width, height= x.size()
        x = x.view(batch_size, C, -1)
        X = x.permute(2,0,1)
        input_dim = C*width*height
        self.ffn = nn.Linear(input_dim, input_dim)

        attn_output, _ = self.multihead_attn(X, X, X)
        X = X + attn_output

        output_x = X.permute(1,2,0).view(batch_size, C, width, height)
        output = output_x.view(batch_size, -1)

        output = output.view(batch_size, C, width, height)    
        output += output_x
        output = self.norm1(output)
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
        self.Bottleneck_conv = nn.Conv2d(out_channels*3, out_channels, kernel_size=1, groups=groups)

        self.pool = nn.MaxPool2d(2, 2)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_rate)

        if self.attention:
            self.attention = Attention_Block(out_channels, out_channels, num_heads, dropout_rate)

    def forward(self, x):

        x_skip = self.Skip_conv(x)
        x = self.Inception_conv(x)
        x = self.activation(self.Bottleneck_conv(x))

        x = x + x_skip

        if self.attention:
            x = self.attention(x)

        x = self.norm(x)
        transfer = self.dropout(x)
        x = self.pool(transfer)

        return x, transfer
    
# Defines the Bottlneck Module:
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention = False, dropout_rate = 0.2, groups = 1, num_heads = 8):
        super(BottleneckBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention = attention
        self.dropout = dropout_rate
        self.groups = groups
        self.num_heads = num_heads

        self.Inception_conv = Inception_Block(in_channels, out_channels, groups)
        self.Bottleneck_conv = nn.Conv2d(out_channels*3,out_channels, kernel_size=1, groups=groups)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()

        if self.attention:
            self.attention = Attention_Block(out_channels, out_channels, num_heads, dropout_rate)

    def forward(self, x):
        x_skip = self.skip_conv(x)
        x = self.Inception_conv(x)
        x = self.activation(self.Bottleneck_conv(x))

        x += x_skip

        if self.attention:
            x = self.attention(x)
            
        x = self.norm(x)
        x = self.dropout(x)
        return x

# Define the Upsample Module:
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode = "Transpose", Merger = "concat", skip_encoder_decoder = True, Experimental = False, dropout_rate = 0.2, groups = 1):
        super(UpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout_rate
        self.groups = groups
        self.up_mode = up_mode
        self.Experiments = Experimental
        self.Merger = Merger
        self.skip_encoder_decoder = skip_encoder_decoder

        if self.skip_encoder_decoder and self.Merger == "concat":
            self.output_conv = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, groups=groups)

        elif self.skip_encoder_decoder and self.Merger == "add":
            self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)

        if self.Experiments and up_mode == "Transpose":
            self.InceptionTranspose_conv = InceptionTranspose_Block(in_channels, in_channels)
            self.Bottleneck_conv = nn.Conv2d(in_channels*3, in_channels, kernel_size=1, groups=groups)

        elif not self.Experiments:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups)
            self.Transconv = Up_Sample(in_channels, in_channels, mode = up_mode)

        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x, x_skip):
        if (self.Experiments and self.up_mode == "Transpose"):
            x = self.InceptionTranspose_conv(x)
            x = self.activation(self.Bottleneck_conv(x))
        else:
            x = self.activation(self.conv(x))
            x = self.activation(self.Transconv(x))

        if self.skip_encoder_decoder and self.Merger == "concat":
            x = torch.cat([x, x_skip], 1)
            x = self.activation(self.output_conv(x))

        elif self.skip_encoder_decoder and self.Merger == "add":
            x = x + x_skip
            x = self.activation(self.output_conv(x))

        x = self.norm(x)
        x = self.dropout(x)

        return x
    
# Outlines the full Unet Module:
class Unet(pl.LightningModule):
    def __init__(self, in_channels, channels_per_layer, num_layers, Attention, num_heads, groups, dropout_rate, up_mode, Merger, Experimental, skip_encoder_decoder = True):
        super(Unet, self).__init__()
        
        # Start by initialising the class variables
        self.in_channels = in_channels
        self.channels_per_layer = channels_per_layer
        self.num_layers = num_layers
        self.Attention = Attention
        self.num_heads = num_heads
        self.groups = groups
        self.dropout = dropout_rate
        self.up_mode = up_mode
        self.Merger = Merger
        self.Experimental = Experimental
        self.skip_encoder_decoder = True

        self.channels_per_layer = [self.num_layers*in_channels] + self.channels_per_layer
        print(self.channels_per_layer)
        self.save_hyperparameters()

        # Define the Encoder
        self.encoder = nn.ModuleList()
        for index in range(self.num_layers):
            if index != self.num_layers - 1:
                self.encoder.append(DownBlock(self.channels_per_layer[index], self.channels_per_layer[index+1], self.Attention[index], self.dropout, self.groups, self.num_heads))
    
        # Define the Bottleneck 
        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(BottleneckBlock(self.channels_per_layer[-1], self.channels_per_layer[-1], self.Attention[-1], self.dropout, self.groups, self.num_heads))

        # Define the Decoder
        self.decoder = nn.ModuleList()
        for index in range(1,self.num_layers):
            self.decoder.append(UpBlock(self.channels_per_layer[-index], self.channels_per_layer[-index-1], self.up_mode, self.Merger, self.skip_encoder_decoder, self.Experimental, self.dropout, self.groups))
        
        # Define the Final Convolutional Layer
        self.final_conv = nn.Conv2d(self.channels_per_layer[0], self.in_channels, kernel_size=1)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
    
    def forward(self, x):

        epsilson = 1

        stack = None

        factor = 10.0
        for i in range(self.num_layers):
            scale = x.clone()*(factor**(-i))
            scale = torch.sin(scale)
            if stack is None:
                stack = scale
            else:
                stack = torch.cat([stack, scale], 1)
        
        x = stack

        encoder_out = []
        for i, layer in enumerate(self.encoder):
            x, transfer = layer(x)
            encoder_out.append(transfer)

        x = self.bottleneck[0](x)

        for i, layer in enumerate(self.decoder):
            x = layer(x, encoder_out[-i-1])

        x = self.final_conv(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr = 1e-3, weight_decay = 1e-5)
        scheduler = {
            "scheduler":ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = 10, verbose = True),
            "monitor":"val_loss"
        }

        return [optimizer], [scheduler]
    
    def photonLoss(self, result, target):
        expEneergy = torch.exp(result)
        perImage = -torch.mean(result*target, dim = (-1,-2,-3), keepdims = True)
        perImage += torch.log(torch.sum(expEneergy, dim = (-1,-2,-3), keepdims = True))*torch.mean(target, dim = (-1,-2,-3), keepdims = True)
        return torch.mean(perImage)
    
    def MSELoss(self, result, target):
        expEnergy = torch.exp(result)
        expEnergy /= torch.sum(expEnergy, dim = (-1,-2,-3), keepdims = True)
        target = target / (torch.mean(target, dim = (-1,-2,-3), keepdims = True) + 1e-6)
        return torch.mean((expEnergy - target)**2)
    
    def training_step(self, batch, batch_idx):
        loss = self.photonLoss(self(batch[:, self.in_channels:,...]), batch[:, :self.in_channels,...])
        self.log("train_loss", loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        val_loss = self.photonLoss(self(batch[:, self.in_channels:,...]), batch[:, :self.in_channels,...])
        self.log("val_loss", val_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self.photonLoss(self(batch[:, self.in_channels:,...]), batch[:, :self.in_channels,...])
        self.log("test_loss", test_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return test_loss







