# train_planner.py

"""
Usage:
    python3 -m homework.train_planner --model transformer --your_args_here
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from homework.models import MLPPlanner, TransformerPlanner, save_model, load_model
from grader.datasets.road_dataset import load_data
import torch.nn.functional as F
import math
import argparse


# Base hyperparameters (for MLP)
BASE_CONFIG = {
    'batch_size': 128,
    'epochs': 150,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 20,
    't_max': 50,
    'eta_min': 1e-5,
    'gradient_clip': 1.0
}

# Modified Transformer config to reduce size but maintain capacity
TRANSFORMER_CONFIG = {
    'batch_size': 128,            # Reduced for better stability
    'epochs': 150,
    'learning_rate': 5e-4,       # Reduced from 5e-3 for better stability
    'weight_decay': 1e-4,        # Adjusted weight decay
    'patience': 25,
    't_max': 100,                # Increased T_max for slower decay
    'eta_min': 1e-6,
    'warmup_epochs': 5,          # Increased warmup for better training stability
    'gradient_clip': 0.5         # Adjusted gradient clipping
}


def get_loss_fn(model_type):
    def transformer_loss(pred, target, mask):
        # Balanced L1 loss for better handling of lateral and longitudinal errors
        long_loss = F.l1_loss(pred[..., 0], target[..., 0], reduction='none')
        lat_loss = F.l1_loss(pred[..., 1], target[..., 1], reduction='none')
        
        # Balanced weights: adjust to prevent overshadowing
        loss = (long_loss * 1.0 + lat_loss * 2.0) * mask  # Reduced lateral weight from 4.0 to 2.0
        
        # Normalize by the number of valid waypoints
        return loss.sum() / (mask.sum() + 1e-6)
    
    def mlp_loss(pred, target, mask):
        # Original MLP loss
        mse = F.mse_loss(pred, target, reduction='none')
        smooth_l1 = F.smooth_l1_loss(pred, target, reduction='none')
        combined = mse + smooth_l1
        return (combined * mask[..., None]).mean()
    
    return transformer_loss if model_type == 'transformer' else mlp_loss


def train_planner(model_name='transformer'):
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
                # Linear warmup
                return (epoch + 1) / config['warmup_epochs']
            return 1.0
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.98)
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
        eta_min=config['eta_min']
    )
    
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
    
    try:
        for epoch in range(config['epochs']):
            # Adjust learning rate for transformer warmup
            if model_name == 'transformer' and epoch < config['warmup_epochs']:
                lr_multiplier = get_lr_multiplier(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['learning_rate'] * lr_multiplier
            
            model.train()
            total_train_loss = 0.0
            num_train_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                target_waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                
                optimizer.zero_grad()
                
                # Get predictions
                predicted_waypoints = model(track_left, track_right)
                
                # Get the appropriate loss function
                loss_fn = get_loss_fn(model_name)
                
                # Calculate loss
                loss = loss_fn(predicted_waypoints, target_waypoints, waypoints_mask)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                
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
                    
                    loss_fn = get_loss_fn(model_name)
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
                # Save best model immediately when we find it
                model.load_state_dict(best_model_state)
                save_model(model)
                print(f'Saved best model with validation loss: {best_val_loss:.4f}')
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
                
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    finally:
        print('\nSaving best model...')
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            save_model(model)
            print(f'Final best validation loss: {best_val_loss:.4f}')
        else:
            print('No improvement during training, saving current model state.')
            save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Planner Model")
    parser.add_argument('--model', type=str, default='transformer', choices=['mlp', 'transformer'], help='Model type to train')
    
    args = parser.parse_args()
    
    print(f"\nStarting {args.model.capitalize()} training...")
    train_planner(args.model)