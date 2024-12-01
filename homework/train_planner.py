# train_planner.py

"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from homework.models import MLPPlanner, TransformerPlanner, save_model, load_model
from grader.datasets.road_dataset import load_data
import torch.nn.functional as F

# Enhanced training hyperparameters
BATCH_SIZE = 128
EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 20

# Scheduler parameters
T_MAX = 50
ETA_MIN = 1e-5

def train_planner(model_name='mlp'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")

    # Model initialization based on model_name
    if model_name == 'mlp':
        model = MLPPlanner(hidden_size=512, num_hidden_layers=5, dropout_rate=0.3).to(device)
    elif model_name == 'transformer':
        model = TransformerPlanner().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    print(f"Training {model_name.upper()} model...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=T_MAX,
        eta_min=ETA_MIN,
        verbose=True
    )
    
    def combined_loss(pred, target, mask):
        mse = F.mse_loss(pred, target, reduction='none')
        smooth_l1 = F.smooth_l1_loss(pred, target, reduction='none')
        combined = mse + smooth_l1
        combined = combined * mask[..., None]
        return combined.mean()
    
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
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        
        for batch in train_loader:
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            target_waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)
            
            predicted_waypoints = model(track_left, track_right)
            
            loss = F.mse_loss(
                predicted_waypoints[..., 0],
                target_waypoints[..., 0],
                reduction='none'
            ) + 2.0 * F.mse_loss(
                predicted_waypoints[..., 1],
                target_waypoints[..., 1],
                reduction='none'
            )
            loss = (loss * waypoints_mask).mean()
            
            optimizer.zero_grad()
            loss.backward()
            
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
                val_loss = combined_loss(
                    predicted_waypoints,
                    target_waypoints,
                    waypoints_mask
                )
                
                total_val_loss += val_loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        
        print(f'Epoch [{epoch+1}/{EPOCHS}] '
              f'Train Loss: {avg_train_loss:.4f} '
              f'Val Loss: {avg_val_loss:.4f}')
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f'New best validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        save_model(model, model_name=model_name)
        print(f'Final best validation loss: {best_val_loss:.4f}')
    else:
        print('No improvement during training.')

if __name__ == "__main__":
    # Train both models
    # print("Starting MLP training...")
    # train_planner('mlp')
    print("\nStarting Transformer training...")
    train_planner('transformer')