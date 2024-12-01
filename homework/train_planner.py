"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import torch.nn as nn
from homework.models import MLPPlanner, save_model
from grader.datasets.road_dataset import load_data

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

def train_planner():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model, optimizer, and loss
    model = MLPPlanner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Create dataloader using the provided load_data function
    train_loader = load_data(
        dataset_path='drive_data/train',
        transform_pipeline='state_only',  # Since we only need track data, not images
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            # Get batch data
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            target_waypoints = batch['waypoints'].to(device)
            
            # Forward pass
            predicted_waypoints = model(track_left, track_right)
            
            # Compute loss
            loss = criterion(predicted_waypoints, target_waypoints)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}')
        
        # Save model if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model)
            print(f'New best loss: {best_loss:.4f}')

if __name__ == "__main__":
    train_planner()
