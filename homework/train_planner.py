import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from homework.models import MLPPlanner, save_model
from grader.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
import numpy as np

# Enhanced training hyperparameters
BATCH_SIZE = 128  # Increased batch size
EPOCHS = 100  # More epochs
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization
PATIENCE = 10  # For early stopping

def train_planner():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model, optimizer, and loss
    model = MLPPlanner(hidden_size=256).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Custom weighted MSE loss
    def weighted_mse_loss(pred, target, mask):
        # Weight longitudinal error (x) and lateral error (y) differently
        error = (pred - target) ** 2
        weighted_error = torch.stack(
            [error[..., 0] * 1.0,  # longitudinal weight
             error[..., 1] * 2.0], # lateral weight (higher)
            dim=-1
        )
        return (weighted_error * mask[..., None]).mean()
    
    # Create dataloaders
    train_loader = load_data(
        dataset_path='drive_data/train',
        transform_pipeline='state_only',
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = load_data(
        dataset_path='drive_data/val',
        transform_pipeline='state_only',
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        
        for batch in train_loader:
            # Get batch data
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            target_waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)
            
            # Forward pass
            predicted_waypoints = model(track_left, track_right)
            
            # Compute loss
            loss = weighted_mse_loss(
                predicted_waypoints,
                target_waypoints,
                waypoints_mask
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                target_waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                
                predicted_waypoints = model(track_left, track_right)
                val_loss = weighted_mse_loss(
                    predicted_waypoints,
                    target_waypoints,
                    waypoints_mask
                )
                
                total_val_loss += val_loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{EPOCHS}] '
              f'Train Loss: {avg_train_loss:.4f} '
              f'Val Loss: {avg_val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f'New best validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model and save it
    model.load_state_dict(best_model_state)
    save_model(model)
    print(f'Final best validation loss: {best_val_loss:.4f}')

if __name__ == "__main__":
    train_planner()