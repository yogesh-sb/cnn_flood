from cnn_batch_processing import create_data_loader
from torch.utils.data import random_split

def prepare_dataset(raster_folder, depth_folder, batch_size=32, test_size=0.2):
    # Create full dataset loader
    dataset = create_data_loader(raster_folder, depth_folder, batch_size, shuffle=False).dataset
    
    # Split into train and validation sets
    total_size = len(dataset)
    test_size = int(test_size * total_size)
    train_size = total_size - test_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = create_data_loader(raster_folder, depth_folder, batch_size, shuffle=True)
    val_loader = create_data_loader(raster_folder, depth_folder, batch_size, shuffle=False)
    
    return train_loader, val_loader