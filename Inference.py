"""
Samples an image using the Generative Accumulation of Photons Model (GAP), based on an initial photon image. 
If the initial photon image contains only zeros the model samples from scratch.
If it contains photon numbers, the model performs diversity denoising.

    Parameters:
        input image (torch.Tensor): the initial photon image, consisting integers (batch, channel, y, x)
        model: the network used to predict the next photon location.
        max_photons (int): the maximum number of photons to sample.
        max_its(int): the maximum number of iterations to stop sampling after.
        max_psnr (float): the maximum PSNR to stop sampling after.
        save_every_n (int): store and return images at every nth step.
        augment (bool): uses 8-fold data augmentation if True (default is False).
        beta (float): photon number is increased expponentially by factor beta in each step.

    Returns:
        denoised image (torch.Tensor): the denoised photon image, consisting integers (batch, channel, y, x)
        photon image (torch.Tensor): the sampled photon image, consisting integers (batch, channel, y, x)
        stack (list): the list of sampled photon images at every nth step.
        i (int): the number of iterations.
"""

import torch
import numpy as np

def sample_image(input_image,
                 model,
                 max_photons = None,
                 max_its = 500000,
                 max_psnr = -15,
                 save_every_n =  5,
                 beta = 0.1,
                 augment = False):
    
    start = input_image.clone()
    photons = start
    photnum = 1

    denoised = None
    stack = []

    sumDenoised = start
    region = 64

    for index in range(max_its):

        # Compute the pseudo PSNR
        psnr = np.log10(photons.mean().item() + 1e-50) * 10
        psnr = max(-40, psnr)

        # Stop if the PSNR is below the threshold
        if (max_photons is not None) and (photons.sum().item() > max_photons):
            break

        if psnr > max_psnr:
            break

        denoised = model(photons).detach()
        denoised = denoised - denoised.max()
        denoised = torch.exp(denoised)

        # Here we save an image to the stack
        if (save_every_n is not None) and (index % save_every_n == 0):
            
            imgsave = denoised[0,0,:,...].detach().cpu()
            imgsave /= imgsave.max()
            photsave = photons[0,0,:,...].detach().cpu()
            photsave /= max(photsave.max(),1)
            combi = torch.cat((imgsave, photsave), dim = 1)
            stack.append(combi.numpy())

        # Increase the photon number
        photnum = max(beta * photons.sum(), 1)

        # Draw new photons
        new_photons = torch.poisson(denoised*(photnum))

        # Add the new photons to the photon image
        photons = photons + new_photons

    
    return denoised[...].detach().cpu().numpy(), photons[...].detach().cpu().numpy(), stack, index