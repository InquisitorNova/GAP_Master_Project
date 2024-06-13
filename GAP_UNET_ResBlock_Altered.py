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

# Define the Building Blocks:

# Define the Convolution 1x1 Block:
def Convolution1x1(in_channels, out_channels, stride = 1, 
                   padding = 0, bias = True, groups = 1):
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size = 1, 
                     stride = stride, 
                     padding = padding, 
                     bias = bias, 
                     groups = groups)

# Define the Convolution 3x3 Block:
def Convolution3x3(in_channels, out_channels, stride = 1, 
                   padding = 1, bias = True, groups = 1):
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size = 3, 
                     stride = stride, 
                     padding = padding, 
                     bias = bias, 
                     groups = groups)

# Define the Convolution 5x5 Block:
def Convolution5x5(in_channels, out_channels, stride = 1,
                   padding = 2, bias = True, groups = 1):
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size = 5, 
                     stride = stride, 
                     padding = padding, 
                     bias = bias, 
                     groups = groups)

# Define the Batch Normalization Block:
def BatchNorm2d(num_features):
    return nn.BatchNorm2d(num_features)

def ReLU():
    return nn.ReLU()

def GroupNorm(num_groups, num_channels):
    return nn.GroupNorm(num_groups, num_channels)

# Define the Inception Block:
class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups = 1):
        super(Inception_Block, self).__init__()

        self.conv1x1 = Convolution1x1(in_channels, out_channels, groups = groups)
        self.conv3x3 = Convolution3x3(in_channels, out_channels, groups = groups)
        self.conv5x5 = Convolution5x5(in_channels, out_channels, groups = groups)
        self.conv1x1_2 = Convolution1x1(out_channels*3, out_channels, groups = groups)

        self.bn = BatchNorm2d(out_channels)
    
    def forward(self, x):
        d1 = self.conv1x1(x)   
        d2 = self.conv3x3(x)
        d3 = self.conv5x5(x)
        d = torch.cat([d1, d2, d3], dim = 1)
        d = self.conv1x1_2(d)
        d = self.bn(d)
        return d


# UpSampling Layer:
def UpConvolution2x2(in_channels, out_channels, mode = "transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(in_channels, 
                                  out_channels, 
                                  kernel_size = 2, 
                                  stride = 2)
    else:
        return nn.Sequential([
            nn.Upsample(scale_factor = 2, mode = "bilinear"),
            nn.Conv2d(in_channels, out_channels, kernel_size = 1)
            ])
    

# DownSampling Block:
class DownSampling_Block(nn.Module):
    def __init__(self, in_channels, out_channels, pooling = True, dropout = False, dropout_rate = 0.2, groups = 1):
        super(DownSampling_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Pooling = pooling
        self.groups = groups
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.skip_Conv = Convolution3x3(in_channels, out_channels, groups = groups)
        self.Inception_Conv = Inception_Block(in_channels, out_channels, groups = groups)
        self.Conv = Convolution3x3(out_channels, out_channels, groups = groups)
        

        if self.Pooling:
            self.Pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.norm = BatchNorm2d(out_channels)
        
    def forward(self, x):
        x_skip = self.skip_Conv(x)
        x = ReLU()(self.Inception_Conv(x))
        x = ReLU()(self.Conv(x) + x_skip)

        x = self.norm(x)

        if self.dropout:
            x = F.dropout(x, p = self.dropout_rate)

        transfer = x
        if self.Pooling:
            x = self.Pool(x)

        return x, transfer

class UpSampling_Block(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode = "transpose", merger = "concat", dropout = False, dropout_rate = 0.2, groups = 1):
        super(UpSampling_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.groups = groups
        self.up_mode = up_mode
        self.merger = merger

        self.UpConv = UpConvolution2x2(in_channels, out_channels, mode = up_mode)

        if self.merger == "concat":
            self.Input_Conv = Convolution3x3(out_channels*2, out_channels, groups = groups)

        else:
            self.Input_Conv = Convolution3x3(out_channels, out_channels, groups = groups)
        
        self.Intermediate_Conv = Convolution3x3(out_channels, out_channels, groups = groups)
        self.Out_Conv = Convolution3x3(out_channels, out_channels, groups = groups)

        self.norm = BatchNorm2d(out_channels)

    def forward(self, x, transfer):
        
        x = self.UpConv(x)
        if self.merger == "concat":
            x = torch.cat([x, transfer], dim = 1)
        else:
            x = x + transfer

        x_skip = ReLU()(self.Input_Conv(x))
        x = ReLU()(self.Intermediate_Conv(x_skip))
        x = ReLU()(self.Out_Conv(x) + x_skip)

        if self.dropout:
            x = F.dropout(x, p = self.dropout_rate)
        
        x = self.norm(x)

        return x

# Outlines the Full Unet Architecture:
class Unet(pl.LightningModule):
    def __init__(self, levels, in_channels = 3, depth = 5, start_filts = 64, up_mode = "transpose", merge_mode = "add"):
        super(Unet, self).__init__()

        self.save_hyperparameters()
        self.levels = levels
        self.in_channels = in_channels
        self.depth = depth
        self.start_filts = start_filts
        self.up_mode = up_mode
        self.merger_mode = merge_mode


        # Define the Encoder:
        self.encoder = nn.ModuleList()
        # Create the channels for the encoder and then fill the Encoder:
        for index in range(depth):
            in_channel = self.in_channels * self.levels if index == 0 else out_channel
            out_channel = self.start_filts * (2**index)
            pooling = True if index < depth - 1 else False

            self.encoder.append(DownSampling_Block(in_channel, out_channel, pooling = pooling))

        # Define the Decoder:
        self.decoder = nn.ModuleList()
        # Create the channels for the decoder and then fill the Decoder:
        for index in range(depth - 1):
            in_channel = out_channel
            out_channel = in_channel // 2

            self.decoder.append(UpSampling_Block(in_channel, out_channel, up_mode = up_mode, merger = merge_mode))

        # Define the Final Convolutional Layer:
        self.final_conv = Convolution1x1(out_channel, self.in_channels)

        self.reset_params()

    def reset_params(self):
        for index, m in enumerate(self.modules()):
            self.weight_init(m)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        epilson = 1
        stack = None

        factor = 10.0
        for index in range(self.levels):
            scale = x.clone()*(factor*(-index))
            scale = torch.sin(scale)
            if stack is None:
                stack = scale
            else:
                stack = torch.cat([stack, scale], dim = 1)
        
        x = stack
        
        encoder_out = []
        for index, layer in enumerate(self.encoder):
            x, transfer = layer(x)
            encoder_out.append(transfer)
        
        for index, layer in enumerate(self.decoder):
            x = layer(x, encoder_out[-(index + 2)])
            
        x = self.final_conv(x)
        return x
            
    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr = 1e-4)
        scheduler = {
            "scheduler":ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = 10, verbose = True),
            "monitor":"val_loss"
        }

        return [optimizer], [scheduler]
    
    def photonLoss(self,result, target):
        expEnergy = torch.exp(result)
        perImage =  -torch.mean(result*target, dim =(-1,-2,-3), keepdims = True )
        perImage += torch.log(torch.mean(expEnergy, dim =(-1,-2,-3), keepdims = True ))*torch.mean(target, dim =(-1,-2,-3), keepdims = True )
        return torch.mean(perImage)
    
    def MSELoss(self,result, target):
        expEnergy = torch.exp(result)
        expEnergy /= (torch.mean(expEnergy, dim =(-1,-2,-3), keepdims = True ))
        target = target / (torch.mean(target, dim =(-1,-2,-3), keepdims = True ))
        return torch.mean((expEnergy-target)**2)
    
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
