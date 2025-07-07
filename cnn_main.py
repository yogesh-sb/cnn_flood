import torch
import os
import sys
# Add parent directory to sys.path to resolve 'cnn' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cnn.cnn_data_preparation import prepare_dataset
from cnn.cnn_train_model import train_model
from cnn.cnn_validate_model import validate_model
from cnn.cnn_predict_depth import predict_new_depth
from cnn.cnn_visualize_performance import nn_model_training_summary

def main():
    # Paths
    raster_folder = "Rasters"
    depth_folder = "Depth"
    test_rain_folder = "test_rain"
    model_path = "surrogate_model.pth"
    plot_file = "model_performance.png"
    output_prediction_file = "Result/predicted_depth.tif"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_prediction_file), exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare dataset
    print("Preparing dataset...")
    train_loader, val_loader = prepare_dataset(raster_folder, depth_folder)
    
    # Train model
    print("Training model...")
    model, losses, maes = train_model(train_loader, device)
    
    # Validate model
    print("Validating model...")
    _, _, val_losses, val_maes = validate_model(model, val_loader, device)
    
    # Visualize performance
    print("Generating performance plot...")
    nn_model_training_summary(model, losses, val_losses, maes, val_maes, 
                             epoch=50, fig_file=plot_file)
    
    # Predict new depth
    print(f"Available rainfall files in {test_rain_folder}:")
    rainfall_files = [f for f in os.listdir(test_rain_folder) if f.endswith('.tif')]
    if not rainfall_files:
        print(f"No .tif files found in {test_rain_folder}")
        return
    for i, f in enumerate(rainfall_files, 1):
        print(f"{i}. {f}")
    file_idx = input(f"Select a file number (1-{len(rainfall_files)}) or enter a custom path: ")
    
    if file_idx.isdigit() and 1 <= int(file_idx) <= len(rainfall_files):
        rainfall_file = os.path.join(test_rain_folder, rainfall_files[int(file_idx) - 1])
    else:
        rainfall_file = file_idx  # Treat as custom path
    
    if os.path.exists(rainfall_file):
        predicted_depth, saved_file = predict_new_depth(model_path, rainfall_file, device, output_file=output_prediction_file)
        print(f"Prediction completed. Shape of predicted depth: {predicted_depth.shape}")
        print(f"Predicted depth saved to: {saved_file}")
    else:
        print(f"Invalid file path: {rainfall_file}")

if __name__ == "__main__":
    main()