
import torch
import numpy as np
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt

# Define the class for Generating the NoisyFaces Dataset:
class FacesDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, names, minPSNR, maxPSNR, augment = True, 
                 maxProb = 0.99, virtSize = None, grayscale = True, amplification_factor = 1.0):
        super(FacesDataset, self).__init__()
        # Initialize the dataset:
        self.root_dir = root_dir
        self.names = names
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
        self.augment = augment
        self.maxProb = maxProb
        self.virtSize = virtSize
        self.grayscale = grayscale
        self.amplification_factor = torch.FloatTensor([amplification_factor])
        self.Image_Paths = [os.path.join(self.root_dir, name) for name in self.names]
        self.length = len(self.Image_Paths)
        
        # Preprocessing Transformations
        self.crop = transforms.CenterCrop((200,200))
        self.resize = transforms.Resize((128,128))
        self.tensorize = transforms.PILToTensor()

        # Random Transformations
        self.flipH = transforms.RandomHorizontalFlip(p = 0.5)
        #self.flipV = transforms.RandomVerticalFlip(p = 0.5)
        #self.scale = transforms.RandomAffine(degrees = 0, translate = (0,0), scale = (1.0,1.3), shear = 0)
        #self.rotate = transforms.RandomAffine(degrees = 5, scale = (1,1), shear = 0)

    def __len__(self):
        if self.virtSize is not None:
            return self.virtSize
        else:
            return self.length

    def __getitem__(self, idx):
        # Load the image:
        idx_ = idx
        if self.virtSize is not None:
            idx_ = np.random.randint(self.length)
        image = Image.open(self.Image_Paths[idx_])
        if self.grayscale:
            image = image.convert('L')
        image = self.crop(image)
        image = self.resize(image)
        image = self.tensorize(image)
        
        # Apply augmentations:
        if self.augment:
            if torch.rand(1) < self.maxProb:
                image = self.flipH(image)
            #if torch.rand(1) < self.maxProb:
                #image = self.flipV(image)
            #if torch.rand(1) < self.maxProb:
                #image = self.scale(image)
            #if torch.rand(1) < self.maxProb:
                #image = self.rotate(image)
        
        # Generate the noisy image:
        image = image.type(torch.float32)
        uniform = torch.rand(1) * (self.maxPSNR - self.minPSNR) + torch.FloatTensor([self.minPSNR])
        constant = torch.FloatTensor([10.0])
        level = (constant**(uniform/constant))
        #print(image.dtype, self.amplification_factor.dtype)
        image *= self.amplification_factor
        ImageNoise = torch.poisson((image/(image.type(torch.float32).mean().item())) * level)
    
        ImageNoise = ImageNoise.type(torch.float32)
        psnr = torch.FloatTensor([uniform])
        psnr_map = psnr.unsqueeze(-1).unsqueeze(-1).expand(image.shape).type(torch.float32)
        
        image_target = image /(image.type(torch.float32).mean().item() + 1e-8)

        #return image, image_input, ImageNoise, psnr_map
        return ImageNoise, psnr_map, image_target

    def show_image(self, index):
        """A method to display the image at the specified index"""
        img = Image.open(self.Image_Paths[index])
        plt.imshow(img)
        plt.xlabel("X_Axis")
        plt.ylabel("Y_Axis")
        plt.show()
    
    def show_transformed_image(self, index):
        """A method to display the transformed image at the specified index"""
        ImageNoise, _, _ = self.__getitem__(index)

        plt.imshow(ImageNoise.permute(1, 2, 0), cmap = "gray")
        plt.xlabel("X_Axis")
        plt.ylabel("Y_Axis")
        plt.show()

    
if __name__ == "__main__":
    print("BinomDataset.py is being run directly")

        