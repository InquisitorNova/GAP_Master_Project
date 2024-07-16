"""
* Filename: AttentionUnetArchitecture.py
* Description: This is a modifield version of the attention UNet architecture 
* used in diffusion models to predict the noise distribution in the image.
* This UNet architecture is used to predict the photon arrival probability distribution
* as part of the Generative Accumulation of Photons Framework. It is designed to address
* the original issues present in the UNet architecture such as the inability to capture
* long-range dependencies due to the lack of attention mechanisms and the inability to model
* image generation across various lightning illuminations to the lack of context.
* The attention UNet architecture is designed to address these issues by incorporating
* attention mechanisms into it's architecture to capture long-range dependencies and 
* by being conditioned on the lightning illumination to model image generation across
* various lightning illuminations.
* Date: 2024-07-11
* Author: Kaylen Smith Darnbrook
* Email: kaylendarnbrook@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pytorch_lightning as pL
import torch.optim as optim

# Define the Device Being Used:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the relevant helper functions for the attention UNet architecture
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def Swish():
    return nn.SiLU()

def GeLU():
    return nn.GELU()

def Mish():
    return nn.Mish()

def Sigmoid():
    return nn.Sigmoid()

# Define the transformation for converting psnr into timesteps.
def psnr_to_timestep(psnr, minpsnr, maxpsnr, num_timesteps):
    """
    psnr - the psnr value to convert to a timestep
    minpsnr - the minimum psnr value that can be achieved
    maxpsnr - the maximum psnr value that can be achieved
    num_timesteps - the number of timesteps in the simulation
    This function takes a psnr as import, performs min-max normalisation and 
    then converts it into a timestep.
    """
    psnr = torch.FloatTensor([psnr])
    minpsnr = torch.FloatTensor([minpsnr])
    maxpsnr = torch.FloatTensor([maxpsnr])
    
    psnr = torch.maximum(torch.minimum(psnr, maxpsnr), minpsnr)
    normalised_psnr = ((psnr - minpsnr) / (maxpsnr - minpsnr))
    timestep = (num_timesteps-1)*normalised_psnr
    return torch.Tensor([timestep]).to(torch.float32)

# Define the transformation for converting the timestep into a positional embedding.
class Temporal_Embedder(nn.Module):
    def __init__(self, n_channels):
        super(Temporal_Embedder, self).__init__()
        self.n_channels = n_channels
        self.Linear_1 = nn.Linear(self.n_channels//4, self.n_channels)
        self.Linear_2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        half_dim = self.n_channels//8
        constant = torch.FloatTensor([10000.0])
        emb = torch.log(constant) / (half_dim -1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim = 1)
        emb = emb.to(device)
        
        emb = F.leaky_relu(self.Linear_1(emb))
        emb = F.leaky_relu(self.Linear_2(emb))
        return emb

# Define the Squeeze and Extraction Block used to recaliberate feature maps from a convolution.
class SEblock(nn.Module):
    def __init__(self, units, n_channels, bottlenecks, dropout_rate):
        super(SEblock, self).__init__()
        
        self.units = units
        self.n_channels = n_channels
        self.bottlenecks = bottlenecks
        self.dropout_rate = dropout_rate
        
        # Define the SE Block layers
        self.Dense = nn.LazyLinear(units)
        self.Dense_2 = nn.Linear(units, n_channels)
        self.Dropout = nn.Dropout2d(dropout_rate)
        self.GlobalPool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x):
        x = self.GlobalPool(x)
        x = x.view(x.size(0), -1)
        x = self.Dense(x)
        x = F.leaky_relu(x)
        x = self.Dense_2(x)
        x = F.sigmoid(x)
        x = x.view(x.size(0), self.n_channels, 1, 1)
        return x

# Define the TimeDistributed Layer used to apply a layer to a sequence of images.
class TimeDistributed(nn.Module):
    def __init__(self, layer, time_steps, *args):        
        super(TimeDistributed, self).__init__()
        
        self.layers = nn.ModuleList([layer(*args) for i in range(time_steps)])

    def forward(self, x):

        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([])
        for i in range(time_steps):
          output_t = self.layers[i](x[:, i, :, :, :])
          output_t  = y.unsqueeze(1)
          output = torch.cat((output, output_t ), 1)
        return output
    
# Define the Different Convolution Blocks for the Encoder and Decoder
class Residual_Block(nn.Module):
    """
    in_channels = Number of Incoming Channels
    out_channels = Number of Outgoing Channels
    time_channels = Number of Time Channels
    dropout_rate = Dropout Rate
    n_groups = Number of Groups
    stride = Stride
    This Block is the traditional residual block used in standard diffusion models.
    """
    def __init__(self, in_channels, out_channels, time_channels,
                dropout_rate, psnr_enabled = True, n_groups = 8, stride=1):
        
        super(Residual_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.n_groups = n_groups
        self.psnr_enabled = psnr_enabled
        self.time_channels = time_channels

        # Define the residual block layers

        self.time_embedding = nn.Linear(time_channels, out_channels)
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv1x1(in_channels, out_channels)

        self.GroupNorm_1 = nn.GroupNorm(n_groups, in_channels)
        self.GroupNorm_2 = nn.GroupNorm(n_groups, out_channels)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, t):
        x_skip = self.conv3(x)
        
        x = self.conv1(F.silu(self.GroupNorm_1(x)))

        if self.psnr_enabled:
            h = F.silu(self.time_embedding(t)[:,:,None,None])
            x += h

        x = self.conv2(F.silu(self.GroupNorm_2(x)))
        x = self.dropout(x)
        x += x_skip
    
        return x

class Efficient_Residual_Block(nn.Module):
    """
    in_channels = Number of incoming channels
    out_channels = Number of outgoing channels
    time_channels = Number of time channels
    dropout_rate = Dropout Rate
    n_groups = Number of Groups
    bottleneck_channels = Number of channels in the Convolution Bottleneck
    stride = Stride
    This block is a computational efficient variant of the residual block used 
    in standard diffusion models. It increases the computational efficiency, by performing
    the 3v3 convolution in a reduced channel space before restoring the dimensionality of the 
    feature representation.
    """
    def __init__(self, in_channels, bottleneck_factor, time_channels, 
                 out_channels, dropout_rate, psnr_enabled = True, n_groups = 8, stride=1):
        
        super(Efficient_Residual_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.psnr_enabled = True
        self.time_channels = time_channels
        self.bottleneck_factor = bottleneck_factor

        bottleneck_channels = out_channels//bottleneck_factor
        self.bottleneck_channels = bottleneck_channels
        
        self.n_groups =  min(n_groups, bottleneck_channels)

        # Define the residual block layers

        self.time_embedding = nn.Linear(time_channels, bottleneck_channels)
        self.conv1 = conv1x1(in_channels, bottleneck_channels)
        self.conv2 = conv3x3(bottleneck_channels, bottleneck_channels)
        self.conv3 = conv1x1(bottleneck_channels, out_channels)
        self.conv4 = conv1x1(in_channels, out_channels)

        self.GroupNorm_1 = nn.GroupNorm(self.n_groups, in_channels)
        self.GroupNorm_2 = nn.GroupNorm(self.n_groups, bottleneck_channels)
        self.GroupNorm_3 = nn.GroupNorm(self.n_groups, bottleneck_channels)

        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x, t):
        x_skip = self.conv4(x)
        
        x = self.conv1(F.silu(self.GroupNorm_1(x)))

        x = self.conv2(F.silu(self.GroupNorm_2(x)))

        if self.psnr_enabled:
            h = F.silu(self.time_embedding(t)[:,:, None, None])
            x += h

        x = self.conv3(F.silu(self.GroupNorm_3(x)))
        x = self.dropout(x)

        x += x_skip

        return x
    
class SqueezeExtraction_Block(nn.Module):
    """
    filters = The number of outgoing channels
    units = The number of units in the SE block
    dropout_rate = The dropout rate
    time_channels = The number of time channels
    units_bottleneck = The number of units in the bottleneck of the SE Block
    n_groups = The number of groups in the Group Normalisation
    This block is a variant of the residual block used in standard diffusion models.
    It incorporates the squeeze and extraction blocks to recaliberate the feature maps 
    coming from the convolutions.
    """
    def __init__(self, in_channels, out_channels, units, dropout_rate,
                 time_channels, units_bottleneck, psnr_enabled = True, 
                 n_groups = 8):
        super(SqueezeExtraction_Block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.units = units
        self.dropout_rate = dropout_rate
        self.units_bottleneck = units_bottleneck
        self.time_channels = time_channels
        self.psnr_enabled = psnr_enabled
        self.n_groups = n_groups
        
        # Define the SqueezeExtraction_Block Layers
        
        self.time_embedding = nn.Linear(time_channels, out_channels)
        self.Conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1, stride = 1)
        self.Conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1, stride = 1)
        self.Conv_Bypass = nn.Conv2d(in_channels, out_channels, 1)

        self.GroupNorm_1 = nn.GroupNorm(self.n_groups, in_channels)
        self.GroupNorm_2 = nn.GroupNorm(self.n_groups, out_channels)

        self.dropout = nn.Dropout(dropout_rate)

        self.SE_Block = SEblock(units, out_channels, units_bottleneck, dropout_rate)

    def forward(self, x, t):
        x_skip = self.Conv_Bypass(x)

        x = self.Conv_1(F.silu(self.GroupNorm_1(x)))

        if self.psnr_enabled:
            h = F.silu(self.time_embedding(t)[:,:,None,None])
            x += h

        x = self.Conv_2(F.silu(self.GroupNorm_2(x)))

        y = self.SE_Block(x)
        y = x * y
        x = y + x_skip

        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        return x
    
class ResidualDense_Block(nn.Module):
    """
    in_channels = Number of incoming channels
    out_channels = Number of Outgoing channels
    dropout_rate = dropout_rate
    time_channels = Number of time channels
    n_groups = Number of groups in the group normalisation layer
    This block is a variant of the residual block used in standard diffusion models.
    The block incorporates dense connections between the convolutional layers to increase
    computational efficiency by incentising feature reuse. Skip connections continue to be 
    used to create a hybrid between a residual an dense connection design.
    """
    def __init__(self, in_channels, out_channels, dropout_rate,
                time_channels, psnr_enabled = True, n_groups = 8):
        super(ResidualDense_Block, self).__init__()
        
        self.in_channels = in_channels
        self.in_out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.n_groups = n_groups
        self.psnr_enabled = True
        
        # Define the Fully_Dense_Encoder layers

        self.time_embedding = nn.Linear(time_channels, in_channels + 2*out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels + 2*out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding = 0, stride=1)
        self.conv5 = nn.Conv2d(in_channels + 3*out_channels, out_channels, kernel_size=1, padding = 0, stride=1)


        self.GroupNorm_1 = nn.GroupNorm(self.n_groups, in_channels)
        self.GroupNorm_2 = nn.GroupNorm(self.n_groups, in_channels + out_channels)
        self.GroupNorm_3 = nn.GroupNorm(self.n_groups, in_channels + 2*out_channels)
        self.BatchNorm_1 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x, t):
        x_cat, x_skip = x, self.BatchNorm_1(self.conv4(x))

        x = self.conv1(F.silu(self.GroupNorm_1(x)))
        x = torch.cat((x, x_cat), 1)

        x_cat = x
        x = self.conv2(F.silu(self.GroupNorm_2(x)))
        x = torch.cat((x, x_cat), 1)

        x_cat = x
        if self.psnr_enabled:
            h = F.silu(self.time_embedding(t)[:,:, None, None])
            x += h

        x = self.conv3(F.silu(self.GroupNorm_3(x)))
        x = torch.cat((x, x_cat), 1)


        x = F.silu(self.conv5(x))
        x = self.dropout(x)
        x += x_skip

        return x      
    
class AttentionBlock(nn.Module):
    """
    n_channels = Number of incoming channels in the image.
    n_heads = Number of heads to use for the multiheaded attention.
    dim_k  = The desired dimensionality of the target key and query vectors.
    n_groups = The number of groups in the group normalisation layer.
    dropout_rate = The dropout rate.
    This block is the attention block used in the attention UNet architecture.
    It flattens the image and divides it into a query, key and value vector.
    It then applies the scaled dot product attention mechanism to the query and key vectors
    to obtain the attention weights. The attention weights are then used to weight the value
    vectors to obtain the final output of the attention block.
    The output is then reshaped back into an image representation.
    
    """
    def __init__(self, n_channels, n_heads, dim_k, n_groups, dropout_rate):
        super(AttentionBlock, self).__init__()

        self.n_channels = n_channels
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate
        self.scale = dim_k ** -0.5

        self.qkv = nn.Linear(n_channels, n_heads * dim_k * 3, bias = False)
        self.output = nn.Linear(n_heads*dim_k, n_channels, bias = False)
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1)
        x_skip = x
        x= x.permute(0,2,1)

        qkv = self.qkv(x).view(batch_size, -1, self.n_heads, 3*self.dim_k)

        q, k, v = qkv.chunk(3, dim=-1)

        attn = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = attn.softmax(dim = -1)
        attn_output = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
         
        attn_output = attn_output.contiguous().view(batch_size, -1, self.n_heads * self.dim_k)
        attn_output = self.dropout(self.output(attn_output))
        #print(attn_output.shape, x_skip.shape)
        
        attn_output = attn_output.permute(0,2,1)
        attn  = self.norm(attn_output + x_skip)
        attn_output = attn.view(batch_size, n_channels, height, width)
        return attn_output
    
class Transformer_Block(nn.Module):
    """
    n_channels = Number of incoming channels in the image.
    n_heads = Number of heads to use for the multiheaded attention.
    dim_k  = The desired dimensionality of the target key and query vectors.
    n_groups = The number of groups in the group normalisation layer.
    dropout_rate = The dropout rate.
    This block is the attention block used in the attention UNet architecture.
    It flattens the image and divides it into a query, key and value vector.
    It then applies the scaled dot product attention mechanism to the query and key vectors
    to obtain the attention weights. The attention weights are then used to weight the value
    vectors to obtain the final output of the attention block.
    The output is then reshaped back into an image representation.
    
    """
    def __init__(self, n_channels, n_heads, dim_k, n_groups, dropout_rate, height, width, depth):
        super(Transformer_Block, self).__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate
        self.scale = dim_k ** -0.5
        self.depth = depth
        self.height = height // (2**(depth-1))
        self.width = width // (2**(depth-1))
        self.seq_len = self.height * self.width

        self.qkv = nn.Linear(n_channels, n_heads * dim_k * 3, bias = False)
        self.output = nn.Linear(n_heads*dim_k, n_channels, bias = False)
        self.norm_1 = nn.LayerNorm(n_channels)
        self.feed_forward_network = nn.LazyLinear(n_channels)
        self.Swish = Swish()
        self.norm_2 = nn.LayerNorm(n_channels)

        self.dropout = nn.Dropout(dropout_rate)

        self.positional_embedding = nn.Parameter(torch.zeros(1, self.seq_len, n_channels))
        
    def forward(self, x):
        batch_size, _, _ = x.shape
        #print(x.shape)
        x = x.permute(0,2,1)
        #print(x.shape, self.seq_len, self.positional_embedding[:, :self.seq_len, :].shape)
        x = x + self.positional_embedding[:, :self.seq_len, :]
        x_skip = x
        
        qkv = self.qkv(x).view(batch_size, -1, self.n_heads, 3*self.dim_k)

        q, k, v = qkv.chunk(3, dim=-1)

        attn = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = attn.softmax(dim = -1)
        attn_output = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
         
        attn_output = attn_output.contiguous().view(batch_size, -1, self.n_heads * self.dim_k)
        attn_output = self.output(attn_output)

        attn_output = self.norm_1(attn_output + x_skip)

        x_skip = attn_output
        x = self.Swish(self.feed_forward_network(x_skip))
        x = self.dropout(x)
        x = self.norm_2(x + x_skip)
        x = x.permute(0,2,1).contiguous().view(batch_size, self.n_channels, -1)
        #print(x.shape)
        return x
    
class Vision_Transformer_Block(nn.Module):
    """
    n_channels = Number of incoming channels in the image.
    n_heads = Number of heads to use for the multiheaded attention.
    dim_k  = The desired dimensionality of the target key and query vectors.
    n_groups = The number of groups in the group normalisation layer.
    dropout_rate = The dropout rate.
    This block is the attention block used in the attention UNet architecture.
    It flattens the image and divides it into a query, key and value vector.
    It then applies the scaled dot product attention mechanism to the query and key vectors
    to obtain the attention weights. The attention weights are then used to weight the value
    vectors to obtain the final output of the attention block.
    The output is then reshaped back into an image representation.
    
    """
    def __init__(self, n_channels, n_heads, dim_k, n_groups, dropout_rate, height, width, depth, patch_size):
        super(Vision_Transformer_Block, self).__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate
        self.scale = dim_k ** -0.5
        self.depth = depth
        self.height = height // (2**(depth-1))
        self.width = width // (2**(depth-1))
        self.Patch_Size = min(patch_size, self.height, self.width)
        self.seq_len = (self.height // self.Patch_Size) * (self.width // self.Patch_Size)

        self.qkv = nn.Linear(n_channels * patch_size**2, n_heads * dim_k * 3, bias = False)
        
        self.output = nn.Sequential(
            nn.Linear(n_heads*dim_k, n_heads*dim_k, bias = False),
            nn.LeakyReLU(),
            nn.Linear(n_heads*dim_k, n_channels * patch_size**2, bias = False),
            nn.LeakyReLU()
        )
        self.norm_1 = nn.LayerNorm(n_channels * patch_size**2)

        #self.ffn_1 = nn.Linear(n_channels * patch_size**2, n_channels * patch_size**2)
        #self.act = nn.LeakyReLU()
        #self.ffn_3 = nn.Linear(n_channels*patch_size**2, n_channels * patch_size**2)
        #self.ffn_4 = nn.LeakyReLU()
    
        self.norm_2 = nn.LayerNorm(n_channels * patch_size**2)
        self.dropout = nn.Dropout(dropout_rate)
        #print(self.seq_len, self.height, self.width, self.depth)
        self.Positional_Embedding  = nn.Parameter(torch.zeros(1, self.seq_len, n_channels * patch_size ** 2))
        
    def forward(self, x):
        batch_size, n_channels, height, width = x.shape
        
        assert height % self.Patch_Size == 0 and width % self.Patch_Size == 0, "Height and Width must be divisible by the Patch Size"

        patches = x.unfold(2, self.Patch_Size, self.Patch_Size).unfold(3, self.Patch_Size, self.Patch_Size)
        patches = patches.contiguous().view(batch_size, n_channels, -1, self.Patch_Size, self.Patch_Size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, -1, n_channels * self.Patch_Size**2)

        seq_len = patches.shape[1]
        #print(patches.shape, self.seq_len)
        patches += self.Positional_Embedding[:, :seq_len, :].to(patches.device)

        x_skip = patches
        
        qkv = self.qkv(patches).view(batch_size, -1, self.n_heads, 3*self.dim_k)

        q, k, v = qkv.chunk(3, dim=-1)

        attn = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = attn.softmax(dim = -1)
        attn_output = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
         
        attn_output = attn_output.contiguous().view(batch_size, -1, self.n_heads * self.dim_k)
        attn_output = self.output(attn_output)

        attn_output = self.norm_1(attn_output + x_skip)

        x_skip = attn_output
        #x = self.act(self.ffn_1(x_skip))
        x = self.dropout(x_skip)
        x = self.norm_2(x + x_skip)
        
        x = x.view(batch_size, height //self.Patch_Size, width // self.Patch_Size, n_channels, self.Patch_Size, self.Patch_Size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(batch_size, n_channels, height, width)
        
        return x

class Transformer(nn.Module):
    """
    n_channels = Number of incoming channels in the image.
    n_heads = Number of heads to use for the multiheaded attention.
    dim_k  = The desired dimensionality of the target key and query vectors.
    n_groups = The number of groups in the group normalisation layer.
    dropout_rate = The dropout rate.
    n_layers = The number of transformer layers to use.
    This block is the transformer block used in the attention UNet architecture.
    It applies the scaled dot product attention mechanism to the query and key vectors
    to obtain the attention weights. The attention weights are then used to weight the value
    vectors to obtain the final output of the attention block.
    The output is then reshaped back into an image representation.
    """
    def __init__(self, n_channels, n_heads, dim_k, n_groups, dropout_rate, num_layers, height, width, depth):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.height = height
        self.depth = depth
        self.width = width
        self.transformer_layers = nn.ModuleList([Transformer_Block(n_channels, n_heads, dim_k, n_groups, dropout_rate, height, width, depth) for _ in range(self.num_layers)])

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape
        #print(height, width, height*width)
        x = x.view(batch_size, n_channels, -1)
        x_skip = x

        for i in range(self.num_layers):
            x = self.transformer_layers[i](x)

        x += x_skip
        x = x.view(batch_size, n_channels, height, width)
        return x
    

class Vision_Transformer(nn.Module):
    """
    n_channels = Number of incoming channels in the image.
    n_heads = Number of heads to use for the multiheaded attention.
    dim_k  = The desired dimensionality of the target key and query vectors.
    n_groups = The number of groups in the group normalisation layer.
    dropout_rate = The dropout rate.
    n_layers = The number of transformer layers to use.
    This block is the transformer block used in the attention UNet architecture.
    It applies the scaled dot product attention mechanism to the query and key vectors
    to obtain the attention weights. The attention weights are then used to weight the value
    vectors to obtain the final output of the attention block.
    The output is then reshaped back into an image representation.
    """
    def __init__(self, n_channels, n_heads, dim_k, n_groups, dropout_rate, num_layers, height, width, depth, transformer_bottleneck, patch_size):
        super(Vision_Transformer, self).__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.Patch_Size = patch_size
        self.height = height
        self.width = width
        self.depth = depth
        self.bottleneck = transformer_bottleneck
        self.n_channel = min(n_channels, transformer_bottleneck)

        self.transformer_layers = nn.ModuleList([Vision_Transformer_Block(self.n_channel, n_heads, dim_k, n_groups, dropout_rate, height, width, depth, patch_size) for _ in range(self.num_layers)])
        self.bottleneck = nn.Conv2d(n_channels,transformer_bottleneck, 1)
        self.unbottlenck = nn.Conv2d(transformer_bottleneck, n_channels, 1)

    def forward(self, x):
        x_skip = x
        x = self.bottleneck(x_skip)
        #print(x.shape)

        for i in range(self.num_layers):
            x = self.transformer_layers[i](x)

        x = self.unbottlenck(x) + x_skip
        return x

class GroupQueryAttentionBlock(nn.Module):
    """
    n_channels = Number of incoming channels in the image.
    n_heads = Number of heads to use for the multiheaded attention.
    dim_k  = The desired dimensionality of the target key and query vectors.
    n_groups = The number of groups in the group normalisation layer.
    dropout_rate = The dropout rate.
    group_size = The size of the groups to divide the image into.
    This block is the group query attention block used in the attention UNet architecture.
    It divides the image into groups and applies the scaled dot product attention mechanism
    to the query and key vectors to obtain the attention weights. The attention weights are then
    used to weight the value vectors to obtain the final output of the attention block.
    The output is then reshaped back into an image representation.
    This variant of the attention mechanism is designed to be more computational efficient than
    the standard attention block at the cost of some global context
    """
    def __init__(self, n_channels, n_heads, dim_k, n_groups, dropout_rate, num_groups):
        super(GroupQueryAttentionBlock, self).__init__()

        self.n_channels = n_channels
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate
        self.scale = dim_k ** -0.5
        self.num_groups = num_groups

        self.qkv = nn.Linear(n_channels, n_heads * dim_k * 3, bias = False)
        self.output = nn.Linear(n_heads*dim_k, n_channels, bias = False)
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1)
        x_skip = x
        x= x.permute(0,2,1)

        qkv = self.qkv(x).view(batch_size, -1, self.n_heads, 3*self.dim_k)

        q, k, v = qkv.chunk(3, dim=-1)

        group_size = max(1, q.shape[1]//self.num_groups)
        #print(group_size, q.shape[1], self.num_groups)

        q_groups = q.view(batch_size, self.num_groups, group_size, self.n_heads, self.dim_k)
        k_groups = k.view(batch_size, self.num_groups, group_size, self.n_heads, self.dim_k)
        v_groups = v.view(batch_size, self.num_groups, group_size, self.n_heads, self.dim_k)

        attn_weights = torch.einsum("bgnhd, bgnhd -> bgnh", q_groups, k_groups) * self.scale
        attn_weights = F.softmax(attn_weights, dim = -1)
        attn_output = torch.einsum("bgnh, bgnhd -> bgnhd", attn_weights, v_groups)
        
        attn_output = attn_output.contiguous().view(batch_size, -1, self.n_heads * self.dim_k)
        attn_output = self.dropout(self.output(attn_output))
        #print(attn_output.shape, x_skip.shape)
        attn_output = attn_output.permute(0,2,1)
        attn  = self.norm(attn_output + x_skip)
        attn_output = attn.view(batch_size, n_channels, height, width)
        return attn_output
    
class ConvGroupQueryAttentionBlock(nn.Module):
    """
    n_channels = Number of incoming channels in the image.
    n_heads = Number of heads to use for the multiheaded attention.
    dim_k  = The desired dimensionality of the target key and query vectors.
    n_groups = The number of groups in the group normalisation layer.
    dropout_rate = The dropout rate.
    reduction_factor = The factor by which to reduce the amount of channels by. 
    group_size = The size of the groups to divide the image into.
    This block is the convolution group query attention block used in the attention UNet architecture.
    It first reduces the channel size using a convolution layer to map the image into a subspace where the
    the attention can more efficiently be computed. It divides the image into groups and 
    applies the scaled dot product attention mechanism to the query and key vectors 
    to obtain the attention weights. The attention weights are then used to weight the value vectors to obtain the final output of the attention block.
    The output is then reshaped back into an image representation.
    This variant of the attention mechanism is designed to be more computational efficient than
    the standard attention block at the cost of some global context
    """
    def __init__(self, n_channels, n_heads, dim_k, n_groups, dropout_rate, num_groups, reduction_factor):
        super(ConvGroupQueryAttentionBlock, self).__init__()

        self.original_n_channels = n_channels
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate
        self.scale = dim_k ** -0.5
        self.num_groups = num_groups
        self.reduction_factor = reduction_factor
        self.n_channels = n_channels//reduction_factor

        self.bottleneck = nn.LazyConv2d(self.original_n_channels//reduction_factor, kernel_size= 1)
        self.unbottleneck = nn.LazyConv2d(self.original_n_channels, kernel_size = 1)

        self.qkv = nn.LazyLinear(n_heads * dim_k * 3, bias = False)
        self.output = nn.Linear(n_heads * dim_k, self.n_channels, bias = False)
        self.norm = nn.GroupNorm(n_groups, self.n_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        
        x = self.bottleneck(x)
        
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1)
        x_skip = x

        x = x.permute(0,2,1)

        qkv = self.qkv(x).view(batch_size, -1, self.n_heads, 3*self.dim_k)

        q, k, v = qkv.chunk(3, dim=-1)

        group_size = max(1, q.shape[1]//self.num_groups)
        #print(num_groups, q.shape[1], self.group_size)

        q_groups = q.view(batch_size, self.num_groups, group_size, self.n_heads, self.dim_k)
        k_groups = k.view(batch_size, self.num_groups, group_size, self.n_heads, self.dim_k)
        v_groups = v.view(batch_size, self.num_groups, group_size, self.n_heads, self.dim_k)

        attn_weights = torch.einsum("bgnhd, bgnhd -> bgnh", q_groups, k_groups) * self.scale
        attn_weights = F.softmax(attn_weights, dim = -1)
        attn_output = torch.einsum("bgnh, bgnhd -> bgnhd", attn_weights, v_groups)
        
        attn_output = attn_output.contiguous().view(batch_size, -1, self.n_heads * self.dim_k)
        attn_output = self.dropout(self.output(attn_output))
        #print(attn_output.shape, x_skip.shape)
        attn_output = attn_output.permute(0,2,1)
        attn  = self.norm(attn_output + x_skip)
        
        attn_output = attn.view(batch_size, n_channels, height, width)
        
        attn_output = self.unbottleneck(attn_output)
        return attn_output

# Define A Down_Sample Block
class Down_Block(nn.Module):
    """
    in_channels = Number of incoming channels
    out_channels = Number of outgoing channels
    time_channels = Number of time channels
    has_attn = Whether the block has an attention mechanism
    n_groups = Number of groups in the group normalisation layer
    dropout_rate = Dropout Rate
    conv_type = The type of convolution block to use
    attn_type = The type of attention block to use
    downsample = Whether to downsample the image
    Pooling = Whether to use pooling or strided convolutions for downsampling
    bottleneck_channels = Number of channels in the bottleneck of the Efficient Residual Block
    units = Number of units in the SE Block
    bottleneck_units = Number of units in the bottleneck of the SE Block
    This block is the downsample block used in the attention UNet architecture. It is used to
    downsample the image and increase the number of channels in the image. It can incorporate
    attention mechanisms and different types of convolution blocks to increase the computational
    efficiency of the model.
    """
    def __init__(self, in_channels, out_channels, time_channels, n_groups,
                dropout_rate, conv_type = "Residual_Block", attn_type = "Attention",
                n_heads = None, dim_k = None, num_groups = None, downsample = True, 
                Pooling = True, bottleneck_factor = None, units = None, depth = None,
                reduction_factor = None, bottleneck_units = None, psnr_enabled = True,
                transformer_bottleneck = None,height = 256, width = 256, num_layers = 3, patch_size = 16):
        
        super(Down_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.dropout_rate = dropout_rate
        self.n_groups = n_groups
        self.bottleneck_factor = bottleneck_factor
        self.units = units
        self.bottleneck_units = bottleneck_units
        self.conv_type = conv_type
        self.attn_type = attn_type
        self.Pooling = Pooling
        self.downsample = downsample
        self.psnr_enabled = psnr_enabled
        self.num_layers = num_layers
        self.dim_k = dim_k
        self.num_groups = num_groups
        self.n_heads = n_heads
        self.reduction_factor = reduction_factor
        self.height = height
        self.width = width
        self.depth = depth
        self.transformer_bottleneck = transformer_bottleneck
        self.patch_size = patch_size

        # Define the Down_Block layers
        if "Residual_Block" in self.conv_type:
            self.Conv_Block = Residual_Block(in_channels = in_channels, out_channels = out_channels, 
                                             time_channels=time_channels, dropout_rate=dropout_rate, 
                                             n_groups=n_groups, psnr_enabled = psnr_enabled)
            
        if "Efficient_Residual_Block" in self.conv_type:
            self.Conv_Block = Efficient_Residual_Block(in_channels=in_channels, bottleneck_factor=bottleneck_factor, 
                                                       time_channels=time_channels, out_channels=out_channels, 
                                                       dropout_rate=dropout_rate, n_groups=n_groups, psnr_enabled = psnr_enabled)
            
        if "SqueezeExtraction_Block" in self.conv_type:
            self.Conv_Block = SqueezeExtraction_Block(in_channels = in_channels, out_channels = out_channels, units = units, dropout_rate= dropout_rate, 
                                                      time_channels= time_channels, units_bottleneck= bottleneck_units, n_groups= n_groups, psnr_enabled= psnr_enabled)
        if "ResidualDense_Block" in self.conv_type:
            self.Conv_Block = ResidualDense_Block(in_channels = in_channels, out_channels = out_channels, 
                                                 time_channels = time_channels, dropout_rate = dropout_rate, n_groups = n_groups, psnr_enabled = psnr_enabled)
        
        if attn_type is not None:
            self.has_attn = True
            if "Attention" in attn_type:
                self.Attention = AttentionBlock(n_channels= out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                                dropout_rate = dropout_rate)
            if "GroupQueryAttention" in attn_type:
                self.Attention = GroupQueryAttentionBlock(n_channels=out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                                          dropout_rate = dropout_rate, num_groups = num_groups)
            if "ConvGroupQueryAttention" in attn_type:
                self.Attention = ConvGroupQueryAttentionBlock(n_channels=out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                                             dropout_rate = dropout_rate, num_groups = num_groups, reduction_factor = reduction_factor,
                                                             )
            if "Vision_Attention" in attn_type:
                self.Attention = Vision_Transformer_Block(n_channels = out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                            dropout_rate = dropout_rate, height = height, width = width, depth = depth, 
                                            patch_size = patch_size)
            if "Transformer" in attn_type:
                self.Attention = Transformer(n_channels = out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                            dropout_rate = dropout_rate, num_layers = num_layers, height = height, width = width,
                                            depth = depth)
                
            if "VisionTransformer" in attn_type:
                self.Attention = Vision_Transformer(n_channels = out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                            dropout_rate = dropout_rate, num_layers = num_layers, height = height, width = width, depth = depth,
                                            patch_size = patch_size, bottleneck=self.transformer_bottleneck)
        else:
            self.has_attn = False
            self.Attention = None

        if self.downsample:
            if self.Pooling:
                self.Pool = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                self.stride = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1, stride=2) 

    def forward(self, x, t):
        x = self.Conv_Block(x, t)

        if self.has_attn:
            x = self.Attention(x)
        
        if self.downsample:
            before_pool = x
            if self.Pooling:
                before_pool = x
                x = self.Pool(x)
            else:
                before_pool = x
                x = self.stride(x)
            
            return x, before_pool
        else:
            return x
    
# Define the Up_Sample Block
class Up_Block(nn.Module):
    """
    in_channels = Number of incoming channels
    out_channels = Number of outgoing channels
    time_channels = Number of time channels
    has_attn = Whether the block has an attention mechanism
    n_groups = Number of groups in the group normalisation layer
    dropout_rate = Dropout Rate
    conv_type = The type of convolution block to use
    attn_type = The type of attention block to use
    up_sample = Whether to upsample the image.
    transpose = Whether to use transpose convolutions or upsample layers for upsampling.
    merge_type = The type of merge operation to use for the skip connections.
    bottleneck_channels = Number of channels in the bottleneck of the Efficient Residual Block
    units = Number of units in the SE Block
    bottleneck_units = Number of units in the bottleneck of the SE Block
    This block is the upsample block used in the attention UNet architecture. It is used to
    upsample the image and decrease the number of channels in the image. It can incorporate
    attention mechanisms and different types of convolution blocks to increase the computational
    efficiency of the model.
    """ 
    def __init__(self, in_channels, out_channels, time_channels, n_groups,
                dropout_rate, conv_type = "Residual_Block", attn_type = "Attention",
                up_sample = True, transpose = True, merge_type = "concat", n_heads = None,
                dim_k = None, num_groups = None, reduction_factor = None,
                height = None, width = None, depth = None, bottleneck_factor = None, 
                units = None, bottleneck_units = None, psnr_enabled = True, 
                transformer_bottleneck = None, num_layers = 3, patch_size = 16):
        super(Up_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.dropout_rate = dropout_rate
        self.n_groups = n_groups
        self.bottleneck_factor = bottleneck_factor
        self.units = units
        self.bottleneck_units = bottleneck_units
        self.conv_type = conv_type
        self.attn_type = attn_type
        self.transpose = transpose
        self.merge_type = merge_type
        self.up_sample = up_sample
        self.dim_k = dim_k
        self.psnr_enabled = psnr_enabled
        self.num_groups = num_groups
        self.reduction_factor = reduction_factor
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.height = height
        self.width = width
        self.depth = depth
        self.transformer_bottleneck = transformer_bottleneck
        self.patch_size = patch_size

        # Define the Up_Block layers
        if up_sample:
            if "concat" in merge_type:
                if "Residual_Block" in self.conv_type:
                    self.Conv_Block = Residual_Block(out_channels, out_channels=out_channels, time_channels=time_channels, 
                                                dropout_rate=dropout_rate, n_groups= n_groups, psnr_enabled = psnr_enabled)
                
                if "Efficient_Residual_Block" in self.conv_type:
                    self.Conv_Block = Efficient_Residual_Block(in_channels = out_channels, bottleneck_factor=bottleneck_factor, time_channels= time_channels, 
                                                       out_channels= out_channels, dropout_rate=dropout_rate, n_groups=n_groups, psnr_enabled = psnr_enabled)
            
                if "SqueezeExtraction_Block" in self.conv_type:
                    self.Conv_Block = SqueezeExtraction_Block(in_channels = out_channels, out_channels= out_channels, units = units, dropout_rate= dropout_rate, 
                                                      time_channels= time_channels, units_bottleneck= bottleneck_units, psnr_enabled = psnr_enabled) 
            
                if "ResidualDense_Block" in self.conv_type:
                    self.Conv_Block = ResidualDense_Block(out_channels, out_channels= out_channels, 
                                                  dropout_rate= dropout_rate, time_channels= time_channels, 
                                                  n_groups= n_groups, psnr_enabled = psnr_enabled)
            else:
                if "Residual_Block" in self.conv_type:
                    self.Conv_Block = Residual_Block(in_channels = out_channels, out_channels=out_channels, time_channels=time_channels, 
                                                dropout_rate=dropout_rate, n_groups= n_groups, psnr_enabled = psnr_enabled)
                
                if "Efficient_Residual_Block" in self.conv_type:
                    self.Conv_Block = Efficient_Residual_Block(in_channels = out_channels, bottleneck_factor=bottleneck_factor, time_channels= time_channels, 
                                                       out_channels= out_channels, dropout_rate=dropout_rate, n_groups=n_groups, psnr_enabled = psnr_enabled)
            
                if "SqueezeExtraction_Block" in self.conv_type:
                    self.Conv_Block = SqueezeExtraction_Block(in_channels = out_channels, out_channels = out_channels, units = units, dropout_rate= dropout_rate, 
                                                      time_channels= time_channels, units_bottleneck= bottleneck_units, psnr_enabled = psnr_enabled) 
            
                if "ResidualDense_Block" in self.conv_type:
                    self.Conv_Block = ResidualDense_Block(in_channels = out_channels, out_channels= out_channels, 
                                                  dropout_rate= dropout_rate, time_channels= time_channels, 
                                                  n_groups= n_groups, psnr_enabled = psnr_enabled)
                
        else:
            if "Residual_Block" in self.conv_type:
                self.Conv_Block = Residual_Block(in_channels, out_channels=out_channels, time_channels=time_channels, 
                                                dropout_rate=dropout_rate, n_groups= n_groups, psnr_enabled = psnr_enabled)
                
            if "Efficient_Residual_Block" in self.conv_type:
                self.Conv_Block = Efficient_Residual_Block(in_channels = in_channels, bottleneck_factor=bottleneck_factor, time_channels= time_channels, 
                                                       out_channels= out_channels, dropout_rate=dropout_rate, n_groups=n_groups, psnr_enabled = psnr_enabled)
            
            if "SqueezeExtraction_Block" in self.conv_type:
                self.Conv_Block = SqueezeExtraction_Block(in_channels = in_channels, out_channels = out_channels, units = units, dropout_rate= dropout_rate, 
                                                      time_channels= time_channels, units_bottleneck= bottleneck_units, psnr_enabled = psnr_enabled) 
            
            if "ResidualDense_Block" in self.conv_type:
                self.Conv_Block = ResidualDense_Block(in_channels, out_channels= out_channels, 
                                                  dropout_rate= dropout_rate, time_channels= time_channels, 
                                                  n_groups= n_groups, psnr_enabled = psnr_enabled)

        
        if attn_type is not None:
            self.has_attn = True
            if "Attention" in attn_type:
                self.Attention = AttentionBlock(n_channels= out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                                dropout_rate = dropout_rate)
            if "GroupQueryAttention" in attn_type:
                self.Attention = GroupQueryAttentionBlock(n_channels=out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                                          dropout_rate = dropout_rate, num_groups = num_groups)
            if "ConvGroupQueryAttention" in attn_type:
                self.Attention = ConvGroupQueryAttentionBlock(n_channels=out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                                             dropout_rate = dropout_rate, num_groups = num_groups, reduction_factor = reduction_factor
                                                             )
            if "Vision_Attention" in attn_type:
                self.Attention = Vision_Transformer_Block(n_channels = out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                            dropout_rate = dropout_rate, height=height, width = width, depth = depth, 
                                            patch_size = patch_size)
                
            if "Transformer" in attn_type:
                self.Attention = Transformer(n_channels = out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                            dropout_rate = dropout_rate, num_layers = num_layers, height = height, width = width,
                                            depth = depth)
                
            if "VisionTransformer" in attn_type:
                self.Attention = Vision_Transformer(n_channels = out_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                            dropout_rate = dropout_rate, num_layers = num_layers, height = height, width = width, 
                                            depth = depth, patch_size = patch_size, transformer_bottleneck = self.transformer_bottleneck)
        else:
            self.has_attn = False
            self.Attention = None
        
        if up_sample:
            if self.transpose:
                self.Upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            else:
                self.Upsample = nn.Sequential([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                               nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1)])
        
    def forward(self, x, before_pool, t):
        if self.up_sample:
            x = self.Upsample(x)
            #print(x.shape, before_pool.shape)
            if "concat" in self.merge_type:
                x = torch.cat((x, before_pool), 1)
            else: 
                x += before_pool

        x = self.Conv_Block(x, t)

        if self.has_attn:
            x = self.Attention(x)

        return x

class MiddleBlock(nn.Module):
    """
    in_channels = Number of incoming channels
    time_channels = Number of time channels
    has_attn = Whether the block has an attention mechanism
    n_groups = Number of groups in the group normalisation layer
    dropout_rate = Dropout Rate
    conv_type = The type of convolution block to use
    attn_type = The type of attention block to use
    bottleneck_channels = Number of channels in the bottleneck of the Efficient Residual Block
    units = Number of units in the SE Block
    bottleneck_units = Number of units in the bottleneck of the SE Block
    This block is the middle block used in the attention UNet architecture. It is used to
    process the image and incorporate attention mechanisms to capture long-range dependencies
    in the image. It can incorporate different types of convolution blocks to increase the computational
    efficiency of the model.
    """
    def __init__(self, in_channels, time_channels, n_groups,
                dropout_rate, conv_type = "Residual_Block", attn_type = "Attention",
                bottleneck_factor = None, bottleneck_units = None, units = None,
                n_heads = None, dim_k = None, num_groups = None, reduction_factor = None,
                psnr_enabled = True, num_layers = 3, height = None, width = None, 
                transformer_bottleneck = None, depth = None, patch_size = 16):
        super(MiddleBlock, self).__init__()

        self.in_channels = in_channels
        self.time_channels = time_channels
        self.dropout_rate = dropout_rate
        self.n_groups = n_groups
        self.bottleneck_factor = bottleneck_factor
        self.units = units
        self.bottleneck_units = bottleneck_units
        self.conv_type = conv_type
        self.attn_type = attn_type
        self.n_heads = n_heads
        self.psnr_enabled = psnr_enabled
        self.dim_k = dim_k
        self.num = num_layers
        self.num_groups = num_groups
        self.reduction_factor = reduction_factor
        self.height = height
        self.width = width
        self.depth = depth
        self.transformer_bottleneck = transformer_bottleneck
        self.patch_size = patch_size
        
        # Define the MiddleBlock layers
        if "Residual_Block" in self.conv_type:
            self.Conv_Block_1 = Residual_Block(in_channels = in_channels, out_channels = in_channels, 
                                             time_channels=time_channels, dropout_rate=dropout_rate, 
                                             n_groups=n_groups, psnr_enabled = psnr_enabled)
            
            self.Conv_Block_2 = Residual_Block(in_channels = in_channels, out_channels = in_channels, 
                                             time_channels=time_channels, dropout_rate=dropout_rate, 
                                             n_groups=n_groups, psnr_enabled = psnr_enabled)
            
        if "Efficient_Residual_Block" in self.conv_type:
            self.Conv_Block_1 = Efficient_Residual_Block(in_channels=in_channels, bottleneck_factor=bottleneck_factor, 
                                                       time_channels=time_channels, out_channels=in_channels, 
                                                       dropout_rate=dropout_rate, n_groups=n_groups, psnr_enabled = psnr_enabled)
            
            self.Conv_Block_2 = Efficient_Residual_Block(in_channels=in_channels, bottleneck_factor=bottleneck_factor, 
                                                       time_channels=time_channels, out_channels=in_channels, 
                                                       dropout_rate=dropout_rate, n_groups=n_groups, psnr_enabled = psnr_enabled)
    
        if "SqueezeExtraction_Block" in self.conv_type:
            self.Conv_Block_1 = SqueezeExtraction_Block(in_channels = in_channels, out_channels = in_channels, units = units, dropout_rate= dropout_rate, 
                                                      time_channels= time_channels, units_bottleneck= bottleneck_units, n_groups= n_groups, psnr_enabled = psnr_enabled)
            
            self.Conv_Block_2 = SqueezeExtraction_Block(in_channels = in_channels, out_channels = in_channels, units = units, dropout_rate= dropout_rate, 
                                                      time_channels= time_channels, units_bottleneck= bottleneck_units, n_groups= n_groups, psnr_enabled = psnr_enabled)
        if "ResidualDense_Block" in self.conv_type:
            self.Conv_Block_1 = ResidualDense_Block(in_channels = in_channels, out_channels = in_channels, 
                                                 time_channels = time_channels, dropout_rate = dropout_rate, n_groups = n_groups, psnr_enabled = psnr_enabled)
            
            self.Conv_Block_2 = ResidualDense_Block(in_channels = in_channels, out_channels = in_channels, 
                                                 time_channels = time_channels, dropout_rate = dropout_rate, n_groups = n_groups, psnr_enabled = psnr_enabled)
            
        if attn_type is not None:
            self.has_attn = True
            if "Attention" in attn_type:
                self.Attention = AttentionBlock(n_channels= in_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                                dropout_rate = dropout_rate)
            if "GroupQueryAttention" in attn_type:
                self.Attention = GroupQueryAttentionBlock(n_channels=in_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                                          dropout_rate = dropout_rate, num_groups = num_groups)
            if "ConvGroupQueryAttention" in attn_type:
                self.Attention = ConvGroupQueryAttentionBlock(n_channels=in_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                                             dropout_rate = dropout_rate, num_groups= num_groups, reduction_factor = reduction_factor
                                                             )
            if "Vision_Attention" in attn_type:
                self.Attention = Vision_Transformer_Block(n_channels = in_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                            dropout_rate = dropout_rate, width = width, height = height, 
                                            depth = depth, patch_size = patch_size)
                
            if "Transformer" in attn_type:
                self.Attention = Transformer(n_channels = in_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                            dropout_rate = dropout_rate, num_layers = num_layers, height = height, width = width,
                                            depth = depth)
                
            if "VisionTransformer" in attn_type:
                self.Attention = Vision_Transformer(n_channels = in_channels, n_heads = n_heads, dim_k = dim_k, n_groups = n_groups, 
                                            dropout_rate = dropout_rate, num_layers = num_layers, height = height, depth = depth,
                                            width = width, patch_size = patch_size, transformer_bottleneck = self.transformer_bottleneck)
        else:
            self.has_attn = False
            self.Attention = None

    def forward(self, x, t):
        x = self.Conv_Block_1(x, t)
        if self.has_attn:
            x = self.Attention(x)
        x = self.Conv_Block_2(x, t)
        return x
    
class AttentionUNet(pL.LightningModule):
    """
    initial_channels = Number of channels to initially project the image into.
    channels_list = Number of channels in each layer of the UNet.
    blocks_per_channel = Number of blocks per channel in the UNet.
    n_groups = Number of groups in the group normalisation layer.
    n_heads = Number of heads to use for the multiheaded attention.
    dim_k  = The desired dimensionality of the target key and query vectors.
    group_size = The size of each group in the group query attention block.
    reduction_factor = The factor by which the convolution reduces the number of channels
    in the image representation in the convolution group querry attention block.
    dropout_rate = Dropout Rate.
    time_channels = Number of time channels.
    bottleneck_channels = Number of channels in the bottleneck of the Efficient Residual Block.
    units = Number of units in the SE Block.
    bottleneck_units = Number of units in the bottleneck of the SE Block.
    has_attn = Whether the block has an attention mechanism.
    conv_type = The type of convolution block to use.
    attn_type_list = The type of attention block to use.
    merge_type = The type of merge operation to use for the skip connections.
    maxpsnr = The maximum psnr value in the dataset.
    minpsnr = The minimum psnr value in the dataset.
    num_timesteps = The number of timesteps in the dataset.
    image_channels = The number of channels in the image.
    This class defines the Attention UNet architecture. It is a variant of the standard UNet
    architecture that incorporates attention mechanisms to capture long-range dependencies 
    in the image. This architecture is designed to be used in conjunction with the GAP framework
    it takes the psnr level of the image as input and uses it to generate the temporal embedding
    and then a positional encoding which when combined with the photon count distribution 
    of the image is used to predict the photon arrival distribution.
    """
    def __init__(self, channels, initial_channels, levels, depth, n_groups, n_heads, dim_k, reduction_factor, num_groups, dropout_rate, time_channels, 
                bottleneck_factor, units, bottleneck_units, conv_type, attn_type_list, merge_type, upsample_type, maxpsnr, minpsnr, 
                num_timesteps, middle_attn_type, psnr_enabled = True, image_channels = 1, blocks_per_channels = 1, 
                num_layers = 3, height = 256, width = 256, patch_size = 16, transformer_bottleneck = 32):
        super(AttentionUNet, self).__init__()

        self.channels = channels
        self.initial_channels = initial_channels
        #self.channels_list = channels_list
        self.blocks_per_channels = blocks_per_channels
        self.n_groups = n_groups
        self.n_heads = n_heads
        self.reduction_factor = reduction_factor
        self.num_groups = num_groups
        self.middle_attn_type = middle_attn_type
        self.dim_k = dim_k
        self.dropout_rate = dropout_rate
        self.time_channels = time_channels
        self.bottleneck_factor = bottleneck_factor
        self.bottleneck_units = bottleneck_units
        #self.blocks_per_channels = blocks_per_channel
        self.units = units
        self.transpose = upsample_type
        self.conv_type = conv_type
        self.attn_type_list = attn_type_list
        self.merge_type = merge_type
        #self.depth = len(self.channels_list)
        self.maxpsnr = maxpsnr
        self.levels = levels
        self.depth = depth
        self.minpsnr = minpsnr
        self.num_timesteps = num_timesteps
        self.image_channels = image_channels
        self.psnr_enabled = psnr_enabled
        self.num_layers = num_layers
        self.height = height
        self.width = width
        self.transformer_bottleneck = transformer_bottleneck
        self.patch_size = patch_size

        self.save_hyperparameters()

        # Define the Psnr_to_Timestep function
        self.Psnr_Converter = lambda psnr: psnr_to_timestep(psnr, self.maxpsnr, self.minpsnr, self.num_timesteps)

        # Define the AttentionUNet layers
        self.DownBlocks = []
        self.MiddleBlocks = []
        self.UpBlocks = []
    

        self.Image_Projection = nn.LazyConv2d(initial_channels, kernel_size=3, padding=1, stride=1)
        self.Temporal_Embedding = Temporal_Embedder(initial_channels *4)
        self.input_norm = nn.GroupNorm(n_groups, self.initial_channels)
        self.input_Swish = Swish()

        for index in range(self.depth):

            input_channels = self.initial_channels if index == 0 else output_channels
            output_channels = self.channels*(2**index)
            Pooling = True if index < depth-1 else False
            attn_type = attn_type_list[index]

            self.DownBlocks.append(Down_Block(in_channels = input_channels,
                        out_channels = output_channels, 
                        time_channels = initial_channels*4,
                        n_groups = n_groups,
                        dropout_rate = dropout_rate,
                        conv_type = conv_type,
                        attn_type = attn_type,
                        downsample = Pooling,
                        Pooling = True,
                        bottleneck_factor = bottleneck_factor,
                        units = units,
                        bottleneck_units = bottleneck_units,
                        n_heads= n_heads,
                        num_groups = num_groups,
                        dim_k = dim_k,
                        reduction_factor = reduction_factor,
                        psnr_enabled = psnr_enabled,
                        num_layers = num_layers,
                        patch_size = patch_size,
                        height = self.height,
                        width = self.width,
                        depth = index+1,
                        transformer_bottleneck = transformer_bottleneck
                    ))
            
        
        self.MiddleBlocks.append(MiddleBlock(in_channels= output_channels,
                                            time_channels= initial_channels*4,
                                            n_groups= n_groups,
                                            dropout_rate= dropout_rate,
                                            conv_type= conv_type,
                                            attn_type= middle_attn_type,
                                            bottleneck_factor= bottleneck_factor,
                                            units= units,
                                            bottleneck_units= bottleneck_units,
                                            n_heads= n_heads,
                                            num_groups = num_groups,
                                            dim_k = dim_k,
                                            reduction_factor = reduction_factor,
                                            psnr_enabled = psnr_enabled,
                                            num_layers = num_layers,
                                            patch_size= patch_size,
                                            height = self.height,
                                            width = self.width,
                                            depth = index+1,
                                            transformer_bottleneck = transformer_bottleneck
                                            )
                                        )      
        

        for index in range(self.depth-1):
            input_channels = output_channels
            output_channels = input_channels//2
            attn_type = attn_type_list[index]
            current_depth = self.depth - index - 1

            self.UpBlocks.append(Up_Block(in_channels = input_channels, 
                                            out_channels= output_channels,
                                            time_channels= initial_channels*4,
                                            n_groups=n_groups,
                                            dropout_rate=dropout_rate,
                                            conv_type=conv_type,
                                            attn_type=attn_type,
                                            up_sample=True,
                                            transpose=self.transpose,
                                            merge_type=merge_type,
                                            bottleneck_factor=bottleneck_factor,
                                            units=units,
                                            bottleneck_units=bottleneck_units,
                                            n_heads= n_heads,
                                            num_groups = num_groups,
                                            dim_k = dim_k,
                                            reduction_factor = reduction_factor,
                                            psnr_enabled = psnr_enabled,
                                            num_layers = num_layers,
                                            patch_size = patch_size,
                                            height = self.height,
                                            width = self.width,
                                            depth = current_depth,
                                            transformer_bottleneck = transformer_bottleneck
                                            ))
                
            
        self.Final_Upsample = Up_Block(
            in_channels = output_channels,
            out_channels = initial_channels,
            time_channels= initial_channels*4,
            n_groups=n_groups,
            dropout_rate=dropout_rate,
            conv_type=conv_type,
            attn_type=attn_type,
            up_sample=True,
            transpose=self.transpose,
            merge_type=merge_type,
            bottleneck_factor=bottleneck_factor,
            units=units,
            bottleneck_units=bottleneck_units,
            n_heads= n_heads,
            num_groups = num_groups,
            dim_k = dim_k,
            reduction_factor = reduction_factor,
            psnr_enabled = psnr_enabled,
            num_layers = num_layers,
            patch_size = patch_size,
            height = self.height,
            width = self.width,
            depth = 1,
            transformer_bottleneck = transformer_bottleneck
            )    
            

        self.DownBlocks = nn.ModuleList(self.DownBlocks)
        self.MiddleBlocks = nn.ModuleList(self.MiddleBlocks)
        self.UpBlocks = nn.ModuleList(self.UpBlocks)

        self.output_norm = nn.GroupNorm(n_groups, self.channels)
        self.output_Swish = Swish()
        self.Output_Layer = nn.Conv2d(self.channels, self.image_channels, kernel_size=1, stride=1)
    
    def forward(self, x, psnr):
        epilson = 1
        
        if self.psnr_enabled:
            t = self.Psnr_Converter(psnr)
            t = self.Temporal_Embedding(t)
        else:
            psnr = torch.FloatTensor([-40.0])
            t = self.Temporal_Embedding(psnr)
        
        
        stack = None
        
        factor = 10.0
        for i in range(self.levels):
            scale = x.clone()*(factor**(-i))
            scale = torch.sin(scale)
            if stack is None:
                stack = scale
            else:
                stack = torch.cat((stack,scale),1)
        
        x = stack

        x = self.Image_Projection(x)
        x = self.input_norm(x)
        x = self.input_Swish(x)
        
        Encoder_Skip_Connections = []
        for block in self.DownBlocks:
            #print(x.shape, "Downsampling")
            if block.downsample:
                x, before_pool = block(x, t)
                Encoder_Skip_Connections.append(before_pool)
            else:
                x = block(x, t)
        

        for block in self.MiddleBlocks:
            x = block(x, t)
        
        ptr = 0
        slow_ptr = 0
        Reversed_Encoder_Skip_Connections = Encoder_Skip_Connections[::-1]
        while ptr <= len(self.UpBlocks)-1:
            #print(x.shape, ptr, slow_ptr, before_pool.shape, "Upsampling")
            before_pool = Reversed_Encoder_Skip_Connections[slow_ptr]
            if ptr % self.blocks_per_channels == 0:
                #print(x.shape, before_pool.shape)
                x = self.UpBlocks[ptr](x, before_pool, t)
                slow_ptr += 1
            else:
                x = self.UpBlocks[ptr](x, before_pool, t)
            ptr += 1
        
        before_pool = Encoder_Skip_Connections[0]
        #x = self.Final_Upsample(x, before_pool, t)

        x = self.Output_Layer(self.output_Swish(self.output_norm(x)))
        return x

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)
    
    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'val_loss'
        }
    
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
        psnr = psnr_image.min()
        predicted = self.forward(img_input, psnr)
        train_loss = self.photonLoss(predicted, target_img)
        self.log("train_loss", train_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return train_loss
    
    def validation_step(self, batch, batch_idx = None):
        img_input, psnr_image, target_img = batch
        psnr = psnr_image.min()
        predicted = self.forward(img_input, psnr)
        valid_loss = self.photonLoss(predicted, target_img)
        self.log("val_loss", valid_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return valid_loss
    
    def test_step(self, batch, batch_idx = None):
        img_input, psnr_image, target_img = batch
        psnr = psnr_image.min()
        predicted = self.forward(img_input, psnr)
        test_loss = self.photonLoss(predicted, target_img)
        self.log("test_loss", test_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return test_loss

    def predict(self, x, psnr):
        return self.forward(x, psnr)

if __name__ == "__main__":
    # Define the Attention Unet Model

    # Define the Hyperparameters of the Model:
    channels = 28
    initial_channels = 4
    levels = 10
    depth = 6
    n_heads = 8
    dim_k = 64
    reduction_factor = 4
    num_groups = 4
    dropout_rate = 0.40
    time_channels = 64
    bottleneck_factor = 2
    units = 32
    bottleneck_units = 64
    conv_type = "Residual_Block"
    attn_type_list = [None, None, None, None, None, None, None]
    merge_type = "add"
    middle_attn_type = None
    upsample_type = True
    maxpsnr = -5.0
    transpose = True
    minpsnr = -40.0
    num_timesteps = 1024
    psnr_enabled = True
    n_groups = 4

    # Define the Attention Network:
    AttnUNet = AttentionUNet(
        initial_channels = initial_channels,
        channels = channels,
        levels=levels,
        depth=depth,
        n_heads = n_heads,
        dim_k = dim_k,
        dropout_rate = dropout_rate,
        time_channels = time_channels,
        bottleneck_factor = bottleneck_factor,
        units = units,
        bottleneck_units= bottleneck_units,
        conv_type = conv_type,
        attn_type_list = attn_type_list,
        merge_type = merge_type,
        maxpsnr = maxpsnr,
        minpsnr = minpsnr,
        num_timesteps = num_timesteps,
        reduction_factor = reduction_factor,
        num_groups = num_groups,
        middle_attn_type = middle_attn_type,
        upsample_type = upsample_type,
        psnr_enabled = psnr_enabled,
        n_groups=n_groups
        )
