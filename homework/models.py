from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import at the top of the file
import math
HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_size: int = 512,
        num_hidden_layers: int = 3,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_size = 2 * n_track * 2
        output_size = n_waypoints * 2

        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        ])

        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = track_left.shape[0]

        track_left_flat = track_left.reshape(batch_size, -1)
        track_right_flat = track_right.reshape(batch_size, -1)
        x = torch.cat([track_left_flat, track_right_flat], dim=1)

        x = F.relu(self.input_layer(x))
        x = self.dropout(x)

        for layer in self.hidden_layers:
            residual = x
            x = F.relu(layer(x))
            x = self.dropout(x)
            x = x + residual

        out = self.output(x)
        return out.reshape(batch_size, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Model dimensions
        self.embedding_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.1
        
        # Input processing - Changed to handle 2D input properly
        self.input_projection = nn.Sequential(
            nn.Linear(2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=128,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output processing
        self.output_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, track_left, track_right):
        batch_size = track_left.shape[0]
        seq_len = track_left.shape[1]
        
        # Combine track boundaries [batch, seq_len, 2]
        track_input = torch.stack([track_left, track_right], dim=-1)
        
        # Project input to embedding dimension
        x = self.input_projection(track_input)
        
        # Add positional encoding using sin/cos directly
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=x.device) * 
                           (-math.log(10000.0) / self.embedding_dim))
        
        pe = torch.zeros(seq_len, self.embedding_dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe.unsqueeze(0)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Generate waypoints
        waypoints = self.output_projection(x)
        
        return waypoints
    
class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
