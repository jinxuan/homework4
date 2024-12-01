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
import math

# Base hyperparameters (for MLP)
BASE_CONFIG = {
    'batch_size': 128,
    'epochs': 150,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 20,
    't_max': 50,
    'eta_min': 1e-5
}

# Significantly modified Transformer config
TRANSFORMER_CONFIG = {
    'batch_size': 32,            # Reduced for better stability
    'epochs': 150,
    'learning_rate': 2e-4,       # Reduced learning rate
    'weight_decay': 1e-4,        # Increased weight decay
    'patience': 25,
    't_max': 75,
    'eta_min': 1e-6,
    'warmup_epochs': 5,          # Reduced warmup
    'gradient_clip': 0.5         # Added gradient clipping
}

def train_planner(model_name='mlp'):
    config = TRANSFORMER_CONFIG if model_name == 'transformer' else BASE_CONFIG
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if model_name == 'mlp':
        model = MLPPlanner(hidden_size=512, num_hidden_layers=5, dropout_rate=0.3).to(device)
    elif model_name == 'transformer':
        model = TransformerPlanner().to(device)
        # Initialize transformer weights properly
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    print(f"Training {model_name.upper()} model...")

    # Custom learning rate schedule for transformer
    if model_name == 'transformer':
        def get_lr_multiplier(epoch):
            if epoch < config['warmup_epochs']:
                # Smoother warmup curve
                return epoch / config['warmup_epochs']
            return 1.0
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.98)    # Standard transformer betas
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['t_max'],
        eta_min=config['eta_min'],
        verbose=True
    )
    
    def get_loss_fn(model_type):
        def transformer_loss(pred, target, mask):
            # Separate longitudinal and lateral components
            long_loss = F.smooth_l1_loss(pred[..., 0], target[..., 0], reduction='none')
            lat_loss = F.smooth_l1_loss(pred[..., 1], target[..., 1], reduction='none')
            
            # Combine with higher weight on lateral error
            loss = (long_loss + 3.0 * lat_loss) * mask
            
            return loss.sum() / (mask.sum() + 1e-6)
        
        def mlp_loss(pred, target, mask):
            mse = F.mse_loss(pred, target, reduction='none')
            smooth_l1 = F.smooth_l1_loss(pred, target, reduction='none')
            combined = mse + smooth_l1
            return (combined * mask[..., None]).mean()
        
        return transformer_loss if model_type == 'transformer' else mlp_loss
    
    loss_fn = get_loss_fn(model_name)
    
    train_loader = load_data(
        dataset_path='drive_data/train',
        transform_pipeline='state_only',
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = load_data(
        dataset_path='drive_data/val',
        transform_pipeline='state_only',
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['epochs']):
        # Adjust learning rate for transformer warmup
        if model_name == 'transformer' and epoch < config['warmup_epochs']:
            lr_multiplier = get_lr_multiplier(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['learning_rate'] * lr_multiplier
        
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        
        for batch in train_loader:
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            target_waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)
            
            # Gradient accumulation for transformer
            accumulation_steps = 4 if model_name == 'transformer' else 1
            
            predicted_waypoints = model(track_left, track_right)
            loss = loss_fn(predicted_waypoints, target_waypoints, waypoints_mask) / accumulation_steps
            
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item() * accumulation_steps
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
                val_loss = loss_fn(predicted_waypoints, target_waypoints, waypoints_mask)
                
                total_val_loss += val_loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        
        print(f'Epoch [{epoch+1}/{config["epochs"]}] '
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
        
        if patience_counter >= config['patience']:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        save_model(model)
        print(f'Final best validation loss: {best_val_loss:.4f}')
    else:
        print('No improvement during training.')

if __name__ == "__main__":
    # Train both models
    # print("Starting MLP training...")
    # train_planner('mlp')
    print("\nStarting Transformer training...")
    train_planner('transformer')