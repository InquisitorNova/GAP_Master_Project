# Import relevant modules
import torch as torch
import numpy as np
from torchvision import transforms

# Define the class for the Generating Datasets
class BinomDataset(torch.utils.data.Dataset):
    """
    Returns a BinomDataset that will randomly split an image into input and target using a binomial distribution for each pixel.

        Parameters:
            data (numpy array): A 3D numpy array (image_index, y, x) with integer photons counts
            windowSize (int): The size of the window to split the image into input and target
            minPSNR (float): minimum psuedo PSNR of sampled data (see supplementary material)
            maxPSNR (float): maximum psuedo PSNR of sampled data (see supplementary material)
            virtSize (int): virtual size of dataset (default is None, i.e., the real size)
            augment (bool): whether to augment the data (default is False)
            maxProb (float): the maximum sucess probability of the binomial distribution (default is 0.5)

        Returns:
            A BinomDataset object

    """

    def __init__(self, data, windowSize, minPSNR, maxPSNR, virtSize = None, augment = True, maxProb = 0.99):
        self.data = torch.from_numpy(data.astype(np.int32))
        
        self.crop = transforms.RandomCrop(windowSize)
        self.flipH = transforms.RandomHorizontalFlip()
        self.flipV = transforms.RandomVerticalFlip()
        self.scale = transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1))
        self.rotate = transforms.RandomAffine(degrees = 5, scale = (1, 1), shear = 0)

        self.windowSize = windowSize
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
        self.maxProb = maxProb
        self.std = data.std()

        self.virtSize = virtSize
        self.augment = augment

    def __len__(self):
        if self.virtSize is not None:
            return self.virtSize
        else:
            return self.data.shape[0]
        
    def __getitem__(self, idx):
        idx_ = idx 
        if self.virtSize is not None:
            idx_ = np.random.randint(self.data.shape[0])
        img = self.crop(self.data[idx_])

        uniform = np.random.rand() * (self.maxPSNR - self.minPSNR) + self.minPSNR

        level = (10**(uniform/10.0))/(img.type(torch.float).mean().item() + 1e-5)
        level = min(level, self.maxProb)

        binom = torch.distributions.Binomial(total_count = img, probs = torch.tensor([level]))
        imgNoise = binom.sample()

        img = (img - imgNoise)[None,...].type(torch.float)
        img /= (img.mean() + 1e-8)

        imgNoise = imgNoise[None,...].type(torch.float)
        out = torch.cat((img, imgNoise), dim = 0)

        if not self.augment:
            return out
        
        else:
            if np.random.rand() <0.5:
                out = torch.transpose(out, -1, -2)
            
            if np.random.rand() <0.5:
                return self.flipV(self.flipH(out))

            else:
                return self.scale(self.rotate(out))


if __name__ == "__main__":
    print("BinomDataset.py is being run directly")