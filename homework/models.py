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
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        super().__init__()
        
        # Perceiver dimensions
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.input_dim = 2  # (x, y) coordinates
        self.latent_dim = 256
        self.num_latents = 8
        self.num_heads = 8
        self.num_layers = 4
        self.dropout = 0.1
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.latent_dim)
        
        # Learned latent array
        self.latents = nn.Parameter(torch.randn(1, self.num_latents, self.latent_dim))
        
        # Cross-attention blocks
        self.cross_attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.latent_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Self-attention blocks for latents
        self.self_attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.latent_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Layer norms
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(self.latent_dim) for _ in range(self.num_layers)
        ])
        self.self_norms = nn.ModuleList([
            nn.LayerNorm(self.latent_dim) for _ in range(self.num_layers)
        ])
        
        # MLPs after attention
        self.cross_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim * 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.latent_dim * 4, self.latent_dim)
            ) for _ in range(self.num_layers)
        ])
        
        self.self_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim * 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.latent_dim * 4, self.latent_dim)
            ) for _ in range(self.num_layers)
        ])
        
        # Output heads
        self.output_query = nn.Parameter(torch.randn(1, n_waypoints, self.latent_dim))
        self.output_attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)
        
        # Final output projections
        self.output_longitudinal = nn.Linear(self.latent_dim, 1)
        self.output_lateral = nn.Linear(self.latent_dim, 1)
    
    def forward(self, track_left, track_right):
        batch_size = track_left.shape[0]
        
        # Reshape track points
        track_left = track_left.reshape(batch_size, self.n_track, 2)
        track_right = track_right.reshape(batch_size, self.n_track, 2)
        track_points = torch.cat([track_left, track_right], dim=1)  # [B, 2*n_track, 2]
        
        # Project input to latent dimension
        inputs = self.input_projection(track_points)  # [B, 2*n_track, latent_dim]
        
        # Expand latents for batch
        latents = self.latents.expand(batch_size, -1, -1)  # [B, num_latents, latent_dim]
        
        # Process through Perceiver layers
        for i in range(self.num_layers):
            # Cross-attention between latents and inputs
            cross_attn, _ = self.cross_attention_blocks[i](
                query=latents,
                key=inputs,
                value=inputs
            )
            latents = latents + cross_attn
            latents = self.cross_norms[i](latents)
            latents = latents + self.cross_mlps[i](latents)
            
            # Self-attention between latents
            self_attn, _ = self.self_attention_blocks[i](
                query=latents,
                key=latents,
                value=latents
            )
            latents = latents + self_attn
            latents = self.self_norms[i](latents)
            latents = latents + self.self_mlps[i](latents)
        
        # Output processing
        output_query = self.output_query.expand(batch_size, -1, -1)
        output, _ = self.output_attention(
            query=output_query,
            key=latents,
            value=latents
        )
        output = self.output_norm(output)
        
        # Generate predictions
        longitudinal = self.output_longitudinal(output)  # [B, n_waypoints, 1]
        lateral = self.output_lateral(output)  # [B, n_waypoints, 1]
        
        # Combine lateral and longitudinal predictions
        waypoints = torch.cat([longitudinal, lateral], dim=-1)  # [B, n_waypoints, 2]
        
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
            print(e)
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
