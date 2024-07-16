import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import torch.optim as optim
import pytorch_lightning as pL


def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        xskip = self.conv1(x)
        x = F.relu(self.conv2(xskip))
        x = F.relu(self.conv3(x) + xskip)
#         x = F.dropout(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        xskip = self.conv1(x)
        x = F.relu(self.conv2(xskip))
        x = F.relu(self.conv3(x) + xskip)
#         x = F.dropout(x)
        return x


class Auxiliary_Network(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, levels):
        super(Auxiliary_Network, self).__init__()

        self.in_channels = in_channels
        self.channels = in_channels * levels
        self.out_channels = out_channels
        self.levels = levels
        self.epilson = 1
        self.factor = 10.0
        self.embedding_dim = embedding_dim

        self.Conv1 = conv3x3(self.channels, self.out_channels//2)
        self.Batch_Norm_1 = nn.BatchNorm2d(self.out_channels//2)
        self.Conv2 = conv3x3(self.out_channels//2, self.out_channels)
        self.Batch_Norm_2 = nn.BatchNorm2d(self.out_channels)
        self.Embedding_Layer = conv1x1(1, self.embedding_dim)
        self.Batch_Norm_3 = nn.BatchNorm2d(self.embedding_dim)
        self.Batch_Norm_4 = nn.BatchNorm2d(1)
        self.Bottleneck = conv1x1(self.embedding_dim, 1)

    def forward(self, x, Psnr_Map):
        stack = []

        for i in range(self.levels):
            scale = x.clone()*(self.factor**(-i))
                
            scale = torch.sin(scale)

            stack.append(scale)

        stack = torch.cat(stack, dim = 1)   

        x = stack

        x = F.leaky_relu(self.Batch_Norm_1(self.Conv1(x)))
        x = F.leaky_relu(self.Batch_Norm_2(self.Conv2(x)))

        #psnr_embedded = F.leaky_relu(self.Batch_Norm_3(self.Embedding_Layer(Psnr_Map)))
        #psnr_embedded = F.leaky_relu(self.Batch_Norm_4(self.Bottleneck(psnr_embedded))) 
        #psnr_embedded += Psnr_Map
        #x = torch.cat((x, psnr_embedded), dim = 1)
        #psnr_embedded = F.sigmoid(self.Batch_Norm_3(self.Embedding_Layer(Psnr_Map)))
        return x

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, levels, channels=3, out_channels = 3, depth=5, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='add'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        
        super(UNet, self).__init__()
        
        self.levels = levels
        self.channels = channels
        self.out_channels = out_channels
        self.depth = depth
        self.start_filts = start_filts
        self.up_mode = up_mode
        self.merge_mode = merge_mode

        
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.channels if i == 0 else outs
            outs = self.start_filts*(2**i)
#            outs = self.start_filts
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
#            outs = ins
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.out_channels)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        x = self.conv_final(x)
        return x


class GAPUNet(pL.LightningModule):
    def __init__(self, levels, in_channels=3, intermediary_channels = 8, 
                 out_channels = 3, depth=5, start_filts=64, up_mode='transpose', 
                 embedding_dim = 8, merge_mode='add'):
        
        self.save_hyperparameters()

        super(GAPUNet, self).__init__()

        self.levels = levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.start_filts = start_filts
        self.up_mode = up_mode
        self.merge_mode = merge_mode
        self.intermediary_channels = intermediary_channels

        self.Auxilliary_Network = Auxiliary_Network(in_channels, intermediary_channels, embedding_dim, levels)
        self.UNet = UNet(levels, intermediary_channels, out_channels, depth, start_filts, up_mode, merge_mode)

    def forward(self, x, psnr_map):
        x = self.Auxilliary_Network(x, psnr_map)
        x = self.UNet(x)
        return x
    
    def predict(self, x, psnr_map):
        return self.forward(x, psnr_map)
    
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
    
    def training_step(self, batch, batch_idx = None):
        img_input, psnr_image, target_img  = batch
        predicted = self.forward(img_input, psnr_image)
        train_loss = self.photonLoss(predicted, target_img)
        self.log("train_loss", train_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return train_loss

    def validation_step(self, batch, batch_idx = None):
        img_input, psnr_image, target_img = batch
        predicted = self.forward(img_input, psnr_image)
        valid_loss = self.photonLoss(predicted, target_img)
        self.log("val_loss", valid_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return valid_loss

    def test_step(self, batch, batch_idx = None):
        img_input, psnr_image, target_img = batch
        predicted = self.forward(img_input, psnr_image)
        test_loss = self.photonLoss(predicted, target_img)
        self.log("test_loss", test_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return test_loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'val_loss'
        }
    



