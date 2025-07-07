import torch
import torch.nn as nn

def validate_model(model, val_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    total_mae = 0
    num_batches = 0
    val_losses = []
    val_maes = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            mae = torch.mean(torch.abs(outputs - targets))
            
            total_loss += loss.item()
            total_mae += mae.item()
            val_losses.append(loss.item())
            val_maes.append(mae.item())
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    print(f"Validation Loss: {avg_loss:.4f}, Validation MAE: {avg_mae:.4f}")
    return avg_loss, avg_mae, val_losses, val_maes