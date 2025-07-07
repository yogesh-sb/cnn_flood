import os
import rasterio
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class RasterDepthDataset(Dataset):
    def __init__(self, raster_folder, depth_folder, target_size=(128, 128)):
        self.raster_files = sorted([f for f in os.listdir(raster_folder) if f.endswith('.tif')])
        self.depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.tif')])
        self.raster_folder = raster_folder
        self.depth_folder = depth_folder
        self.target_size = target_size
        
        if len(self.depth_files) != 1:
            raise ValueError("Expected exactly one depth file in the Depth folder")
        
        self.depth_file = self.depth_files[0]  # Single depth file
    
    def __len__(self):
        return len(self.raster_files)
    
    def __getitem__(self, idx):
        # Load raster
        raster_path = os.path.join(self.raster_folder, self.raster_files[idx])
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            # Check for NaN or inf values
            if np.any(np.isnan(raster_data)) or np.any(np.isinf(raster_data)):
                print(f"Warning: NaN or inf found in raster file: {raster_path}")
                raster_data = np.nan_to_num(raster_data, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip to prevent overflow
            raster_data = np.clip(raster_data, -1e6, 1e6)
            raster_data = Image.fromarray(raster_data)
            raster_data = raster_data.resize(self.target_size, Image.BILINEAR)
            raster_data = np.array(raster_data)
        
        # Load single depth file (same for all rasters)
        depth_path = os.path.join(self.depth_folder, self.depth_file)
        with rasterio.open(depth_path) as src:
            depth_data = src.read(1)
            # Check for NaN or inf values
            if np.any(np.isnan(depth_data)) or np.any(np.isinf(depth_data)):
                print(f"Warning: NaN or inf found in depth file: {depth_path}")
                depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip to prevent overflow
            depth_data = np.clip(depth_data, -1e6, 1e6)
            depth_data = Image.fromarray(depth_data)
            depth_data = depth_data.resize(self.target_size, Image.BILINEAR)
            depth_data = np.array(depth_data)
        
        # Normalize with epsilon to avoid division by zero
        eps = 1e-8  # Small constant to prevent division by zero
        raster_mean = np.mean(raster_data)
        raster_std = np.std(raster_data)
        if raster_std == 0:
            print(f"Warning: Standard deviation is zero for raster file: {raster_path}")
            raster_data = raster_data - raster_mean  # Center only, no division
        else:
            raster_data = (raster_data - raster_mean) / (raster_std + eps)
        
        depth_mean = np.mean(depth_data)
        depth_std = np.std(depth_data)
        if depth_std == 0:
            print(f"Warning: Standard deviation is zero for depth file: {depth_path}")
            depth_data = depth_data - depth_mean  # Center only, no division
        else:
            depth_data = (depth_data - depth_mean) / (depth_std + eps)
        
        # Convert to tensors
        raster_tensor = torch.from_numpy(raster_data).float().unsqueeze(0)  # Add channel dim
        depth_tensor = torch.from_numpy(depth_data).float().unsqueeze(0)
        
        return raster_tensor, depth_tensor

def create_data_loader(raster_folder, depth_folder, batch_size, shuffle=True):
    dataset = RasterDepthDataset(raster_folder, depth_folder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)