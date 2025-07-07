import rasterio
import numpy as np
from PIL import Image
import torch
from cnn_cnn_model import CNNModel
import os

def predict_new_depth(model_path, rainfall_file, device, output_file='Result/predicted_depth.tif', target_size=(128, 128)):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the model
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Load and preprocess new rainfall raster
    with rasterio.open(rainfall_file) as src:
        raster_data = src.read(1)
        meta = src.meta  # Get metadata for output
        raster_data = Image.fromarray(raster_data)
        raster_data = raster_data.resize(target_size, Image.BILINEAR)
        raster_data = np.array(raster_data)
    
    # Normalize
    raster_data = (raster_data - np.mean(raster_data)) / np.std(raster_data)
    
    # Convert to tensor
    raster_tensor = torch.from_numpy(raster_data).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        predicted_depth = model(raster_tensor)
    
    predicted_depth = predicted_depth.squeeze().cpu().numpy()
    
    # Update metadata for output
    meta.update({
        'count': 1,
        'dtype': 'float32',
        'height': target_size[0],
        'width': target_size[1]
    })
    
    # Save prediction as GeoTIFF
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(predicted_depth, 1)
    
    return predicted_depth, output_file