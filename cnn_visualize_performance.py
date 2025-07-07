import os
import torch
import numpy as np
import matplotlib
# Use non-interactive Agg backend to avoid Qt threading issues
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Temporary workaround for OMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def count_model_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def nn_model_training_summary(model, losses, val_losses, maes, val_maes, epoch, 
                             fig_title='Model Performance (MSE and MAE)', fig_file=None):
    """
    Visualize training and validation losses (MSE and MAE) over epochs.
    
    Args:
        model: PyTorch model
        losses: List of training MSE losses
        val_losses: List of validation MSE losses
        maes: List of training MAE values
        val_maes: List of validation MAE values
        epoch: Number of training epochs
        fig_title: Title for the plot
        fig_file: File path to save the plot (if None, display plot)
    
    Returns:
        0: Indicates successful execution
    """
    ep = epoch  # Use epoch directly
    
    # Convert lists to numpy arrays for plotting
    losses = np.array(losses)
    val_losses = np.array(val_losses)
    maes = np.array(maes)
    val_maes = np.array(val_maes)
    
    # Check if validation data is sufficient for per-epoch averaging
    if len(val_losses) >= ep and len(val_maes) >= ep:
        # Average losses per epoch
        loss_by_epoch = torch.mean(torch.tensor(losses).view(ep, -1), dim=1).numpy()
        val_loss_by_epoch = torch.mean(torch.tensor(val_losses).view(ep, -1), dim=1).numpy()
        mae_by_epoch = torch.mean(torch.tensor(maes).view(ep, -1), dim=1).numpy()
        val_mae_by_epoch = torch.mean(torch.tensor(val_maes).view(ep, -1), dim=1).numpy()
    else:
        print(f"Warning: Insufficient validation data (val_losses: {len(val_losses)}, val_maes: {len(val_maes)}). Plotting raw values.")
        loss_by_epoch = losses  # Use raw values if not enough data
        val_loss_by_epoch = val_losses
        mae_by_epoch = maes
        val_mae_by_epoch = val_maes
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot MSE
    if len(losses) > 0:
        ax1.plot(np.arange(len(losses)), losses, 'black', alpha=0.3, label='Train MSE (batch)')
    if len(val_losses) > 0:
        ax1.plot(np.arange(len(val_losses)), val_losses, 'green', alpha=0.3, label='Val MSE (batch)')
    if len(loss_by_epoch) > 0:
        ax1.plot(np.arange(len(loss_by_epoch)), loss_by_epoch, 'r', label='Train MSE (avg)')
    if len(val_loss_by_epoch) > 0:
        ax1.plot(np.arange(len(val_loss_by_epoch)), val_loss_by_epoch, 'b', label='Val MSE (avg)')
    ax1.set_title(f'{fig_title} - MSE')
    ax1.set_xlabel('Batch' if len(val_losses) < ep else 'Epoch')
    ax1.set_ylabel('MSE')
    ax1.legend()
    ax1.set_ylim(0, max(max(losses, default=1), max(val_losses, default=1)) * 1.1)
    
    # Plot MAE
    if len(maes) > 0:
        ax2.plot(np.arange(len(maes)), maes, 'black', alpha=0.3, label='Train MAE (batch)')
    if len(val_maes) > 0:
        ax2.plot(np.arange(len(val_maes)), val_maes, 'green', alpha=0.3, label='Val MAE (batch)')
    if len(mae_by_epoch) > 0:
        ax2.plot(np.arange(len(mae_by_epoch)), mae_by_epoch, 'r', label='Train MAE (avg)')
    if len(val_mae_by_epoch) > 0:
        ax2.plot(np.arange(len(val_mae_by_epoch)), val_mae_by_epoch, 'b', label='Val MAE (avg)')
    ax2.set_title(f'{fig_title} - MAE')
    ax2.set_xlabel('Batch' if len(val_losses) < ep else 'Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.set_ylim(0, max(max(maes, default=1), max(val_maes, default=1)) * 1.1)
    
    plt.tight_layout()
    
    if fig_file is not None:
        plt.savefig(fig_file)
        plt.close(fig)
    else:
        plt.show()
    
    print('Number of Parameters:', count_model_param(model))
    return 0