# train_planner.py

"""
Usage:
    python3 -m homework.train_planner --model transformer --your_args_here
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from homework.models import CNNPlanner, MLPPlanner, TransformerPlanner, save_model, load_model
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
    'learning_rate': 1e-4,       # Reduced from 5e-3 for better stability
    'weight_decay': 1e-4,        # Adjusted weight decay
    'patience': 25,
    't_max': 100,                # Increased T_max for slower decay
    'eta_min': 1e-6,
    'warmup_epochs': 5,          # Increased warmup for better training stability
    'gradient_clip': 0.5         # Adjusted gradient clipping
}

# Add this to the existing configs
CNN_CONFIG = {
    'batch_size': 64,            # Smaller batch size for CNN
    'epochs': 100,
    'learning_rate': 1e-4,       # Lower learning rate for stability
    'weight_decay': 1e-4,
    'patience': 15,
    't_max': 50,
    'eta_min': 1e-6,
    'gradient_clip': 1.0
}


def get_loss_fn(model_type):
    def transformer_loss(pred, target, mask):
        # Balanced L1 loss for better handling of lateral and longitudinal errors
        long_loss = F.l1_loss(pred[..., 0], target[..., 0], reduction='none')
        lat_loss = F.l1_loss(pred[..., 1], target[..., 1], reduction='none')
        
        # Balanced weights: adjust to prevent overshadowing
        loss = (long_loss * 4.0 + lat_loss * 1.0) * mask  # Reduced lateral weight from 4.0 to 2.0
        
        # Normalize by the number of valid waypoints
        return loss.sum() / (mask.sum() + 1e-6)
    
    def mlp_loss(pred, target, mask):
        # Original MLP loss
        mse = F.mse_loss(pred, target, reduction='none')
        smooth_l1 = F.smooth_l1_loss(pred, target, reduction='none')
        combined = mse + smooth_l1
        return (combined * mask[..., None]).mean()
    
    def cnn_loss(pred, target, mask):
        # Similar to transformer loss but with different weights
        long_loss = F.l1_loss(pred[..., 0], target[..., 0], reduction='none')
        lat_loss = F.l1_loss(pred[..., 1], target[..., 1], reduction='none')
        
        # Balance longitudinal and lateral errors
        loss = (long_loss * 2.0 + lat_loss * 1.0) * mask
        return loss.sum() / (mask.sum() + 1e-6)
    
    if model_type == 'transformer':
        return transformer_loss
    elif model_type == 'cnn':
        return cnn_loss
    else:
        return mlp_loss


def train_planner(model_name='transformer'):
    # Update config selection
    if model_name == 'transformer':
        config = TRANSFORMER_CONFIG
    elif model_name == 'cnn':
        config = CNN_CONFIG
    else:
        config = BASE_CONFIG
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Update model selection
    if model_name == 'mlp':
        model = MLPPlanner(hidden_size=512, num_hidden_layers=5, dropout_rate=0.3).to(device)
    elif model_name == 'transformer':
        model = TransformerPlanner().to(device)
    elif model_name == 'cnn':
        model = CNNPlanner().to(device)
        
    print(f"Training {model_name.upper()} model...")
    
    # Optimizer setup
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
    
    # Data loading - use "default" instead of "image_only"
    transform_pipeline = 'default' if model_name == 'cnn' else 'state_only'
    
    train_loader = load_data(
        dataset_path='drive_data/train',
        transform_pipeline=transform_pipeline,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = load_data(
        dataset_path='drive_data/val',
        transform_pipeline=transform_pipeline,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    try:
        for epoch in range(config['epochs']):
            model.train()
            total_train_loss = 0.0
            num_train_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Handle different inputs for CNN vs other models
                if model_name == 'cnn':
                    inputs = batch['image'].to(device)
                    target_waypoints = batch['waypoints'].to(device)
                    waypoints_mask = batch['waypoints_mask'].to(device)
                    
                    optimizer.zero_grad()
                    predicted_waypoints = model(inputs)
                else:
                    track_left = batch['track_left'].to(device)
                    track_right = batch['track_right'].to(device)
                    target_waypoints = batch['waypoints'].to(device)
                    waypoints_mask = batch['waypoints_mask'].to(device)
                    
                    optimizer.zero_grad()
                    predicted_waypoints = model(track_left, track_right)
                
                # Get appropriate loss function
                loss_fn = get_loss_fn(model_name)
                loss = loss_fn(predicted_waypoints, target_waypoints, waypoints_mask)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
                
                total_train_loss += loss.item()
                num_train_batches += 1
                
                # Print progress every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{config["epochs"]}] '
                          f'Batch [{batch_idx+1}/{len(train_loader)}] '
                          f'Loss: {loss.item():.4f}')
            
            avg_train_loss = total_train_loss / num_train_batches
            
            # Validation phase
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if model_name == 'cnn':
                        inputs = batch['image'].to(device)
                        target_waypoints = batch['waypoints'].to(device)
                        waypoints_mask = batch['waypoints_mask'].to(device)
                        
                        predicted_waypoints = model(inputs)
                    else:
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
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f'New best validation loss: {best_val_loss:.4f}')
                model.load_state_dict(best_model_state)
                save_model(model)
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
                
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    finally:
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            save_model(model)
            print(f'Final best validation loss: {best_val_loss:.4f}')
        else:
            save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Planner Model")
    parser.add_argument('--model', type=str, default='transformer', 
                      choices=['mlp', 'transformer', 'cnn'], 
                      help='Model type to train')
    
    args = parser.parse_args()
    print(f"\nStarting {args.model.capitalize()} training...")
    train_planner(args.model)