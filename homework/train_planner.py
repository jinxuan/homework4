# train_planner.py

"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from homework.models import MLPPlanner, TransformerPlanner
from grader.datasets.road_dataset import load_data
import os

def train(
    model_name: str = "mlp_planner",
    transform_pipeline: str = "state_only",
    num_workers: int = 4,
    lr: float = 1e-3,
    batch_size: int = 128,
    num_epoch: int = 40,
):
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model selection
    if model_name == "mlp_planner":
        model = MLPPlanner().to(device)
    elif model_name == "transformer_planner":
        model = TransformerPlanner().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Create dataloaders
    train_loader = load_data(
        dataset_path='drive_data/train',
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = load_data(
        dataset_path='drive_data/val',
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    best_val_error = float('inf')
    
    for epoch in range(num_epoch):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            target_waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)
            
            pred_waypoints = model(track_left, track_right)
            loss = criterion(pred_waypoints[waypoints_mask], target_waypoints[waypoints_mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                target_waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                
                pred_waypoints = model(track_left, track_right)
                loss = criterion(pred_waypoints[waypoints_mask], target_waypoints[waypoints_mask])
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epoch}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model (now using validation loss instead of error)
        if val_loss < best_val_error:
            best_val_error = val_loss
            torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pt')
            print(f"New best model saved with loss: {best_val_error:.4f}")
        
        print("-" * 50)

if __name__ == "__main__":
    # Example usage
    for lr in [1e-2, 1e-3, 1e-4]:
        print(f"\nTraining with learning rate: {lr}")
        train(
            model_name="transformer_planner",
            transform_pipeline="state_only",
            num_workers=4,
            lr=lr,
            batch_size=128,
            num_epoch=40,
        )