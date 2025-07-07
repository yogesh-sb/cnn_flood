import torch
import torch.nn as nn
import torch.optim as optim
from cnn_cnn_model import CNNModel

def train_model(train_loader, device, epochs=50, model_path='surrogate_model.pth'):
    model = CNNModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    losses = []
    maes = []
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_mae = 0.0
        num_batches = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mae = torch.mean(torch.abs(outputs - targets))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_mae += mae.item()
            losses.append(loss.item())
            maes.append(mae.item())
            num_batches += 1
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / num_batches:.4f}, MAE: {running_mae / num_batches:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    return model, losses, maes