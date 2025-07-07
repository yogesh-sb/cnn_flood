# CNN-Based Flood Depth Prediction Model

This is the framework for training, validating, and predicting flood depth maps using a Convolutional Neural Network (CNN). It is designed to work with geospatial raster data and supports batch processing and visualization of model outputs.

## 📁 Project Structure

```
.
├── cnn_data_preparation.py        # Prepares raster input data
├── cnn_train_model.py             # Trains the CNN model
├── cnn_predict_depth.py           # Generates predicted depth maps
├── cnn_validate_model.py          # Validates model performance
├── cnn_visualize_performance.py   # Creates plots to visualize accuracy and loss
├── cnn_batch_processing.py        # Batch processing for multiple events
├── cnn_cnn_model.py               # CNN model architecture
├── Depth/
│   └── depth.tif                  # Ground truth depth raster
├── Rasters/
│   ├── 24_hour.tif                # Input rasters used as features
│   ├── 2_hour.tif
│   ├── backwater.tif
│   └── dams.tif
```

## 🧠 Model

The model is a convolutional neural network implemented in `cnn_cnn_model.py`. Input features are geospatial raster layers, and the output is a predicted flood depth raster.

## ⚙️ Requirements

- Python 3.8+
- NumPy
- Rasterio
- PyTorch
- Scikit-learn
- Matplotlib
- GDAL

You can install dependencies via:

```bash
pip install -r requirements.txt
```

*(Note: Ensure GDAL is installed via your package manager before installing Rasterio)*

## 🚀 Usage

### 1. Data Preparation

```bash
python cnn_data_preparation.py
```

### 2. Train Model

```bash
python cnn_train_model.py
```

### 3. Validate Model

```bash
python cnn_validate_model.py
```

### 4. Predict Depth

```bash
python cnn_predict_depth.py
```

### 5. Visualize Performance

```bash
python cnn_visualize_performance.py
```

## 📦 Batch Processing

Use `cnn_batch_processing.py` to run predictions across multiple events.

## 📊 Outputs

- Predicted flood depth raster
- Validation metrics and loss plots
- Error maps (if implemented)

## 📌 Notes

- Ensure all raster layers are aligned and have the same resolution.
- The `Depth/` folder contains the ground truth used for training/validation.
- Adjust the model architecture or hyperparameters in `cnn_cnn_model.py`.

## 👨‍🔬 Authors

Developed by Yogesh Bhattarai(yogeshbhattarai.sb@gmail.com), for research on CNN-based flood depth prediction.
