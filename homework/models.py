from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Architecture parameters
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = 256  # Transformer dimension
        self.nhead = 8      # Number of attention heads
        self.num_layers = 4
        self.dropout = 0.1
        
        # Input embedding
        self.input_embedding = nn.Linear(2, self.d_model)  # From (x,y) to d_model
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=1024,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=1024,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers
        )
        
        # Learnable output queries
        self.output_queries = nn.Parameter(
            torch.randn(1, n_waypoints, self.d_model)
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, 2)

    def forward(self, track_left, track_right):
        batch_size = track_left.shape[0]
        
        # Prepare input sequence
        track_left = track_left.reshape(batch_size, self.n_track, 2)
        track_right = track_right.reshape(batch_size, self.n_track, 2)
        track_points = torch.cat([track_left, track_right], dim=1)  # [B, 2*n_track, 2]
        
        # Embed input
        x = self.input_embedding(track_points)  # [B, 2*n_track, d_model]
        x = self.pos_encoder(x)
        
        # Create attention mask (optional, if needed)
        # src_mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # Encode track points
        memory = self.transformer_encoder(x)
        
        # Prepare decoder queries
        queries = self.output_queries.expand(batch_size, -1, -1)
        
        # Decode
        output = self.transformer_decoder(
            queries,
            memory
        )
        
        # Project to waypoints
        waypoints = self.output_projection(output)  # [B, n_waypoints, 2]
        
        return waypoints

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints

        # Register normalization constants
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 256 because of skip connection
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 128 because of skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Global Average Pooling and final prediction
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.final = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # Normalize input
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Decoder with skip connections
        d3 = self.dec3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.dec1(d2)

        # Global Average Pooling and reshape to waypoints
        x = self.gap(d1)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        
        # Reshape to (batch_size, n_waypoints, 2)
        return x.view(-1, self.n_waypoints, 2)


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
            m.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        except RuntimeError as e:
            print(e)
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
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

    return str(output_path)


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024