# Import the modules for data preprocessing
import os
from astroquery.sdss import SDSS
from astroquery.vizier import Vizier
from astropy import coordinates as coords
import asyncio 
import aiohttp
from tqdm import tqdm
from astropy.io import fits
import torch

async def fetch_image(session, position, band = "g", retries = 5):
    for attempt in range(retries):
        try:
            image_data = SDSS.get_images(coordinates = position, band = band)
            return image_data[0][0].data
        except (ConnectionError, aiohttp.client_exceptions.ClientConnectorError, asyncio.TimeoutError) as error:
            print(f" {error} at {position.ra.deg}, {position.dec.deg} at attempt {attempt}, trying again...")
            await asyncio.sleep(2)
    raise Exception("Failed to fetch image")

async def fetch_images(positions, pbar):
    async with aiohttp.ClientSession() as session:
        fits_data = []
        for pos in tqdm(positions, desc = "Downloading Fits Data"):
            try:
                coord = coords.SkyCoord(pos[0], pos[1], unit = "deg")
                fits_data.append(await fetch_image(session, coord))
                pbar.update(1)
            except Exception as e:
                print(f"Error: {e}")
        return fits_data

def fits_to_tensor(fits_data):
    image_tensor = torch.tensor(fits_data, dtype = torch.float32)
    return image_tensor
