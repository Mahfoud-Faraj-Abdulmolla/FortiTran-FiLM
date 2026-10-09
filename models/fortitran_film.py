# ==============================================================================
# FortiTranM-FiLM: Adaptive Transformer with Feature-wise Linear Modulation
# Authors: Mahfoud Faraj Abdulmolla, Osama Abusaid
# Institution: University of Tripoli, Electrical & Electronic Engineering Dept.
# Repository: https://github.com/Mahfoud-Faraj-Abdulmolla/FortiTran-FiLM
# ==============================================================================

import torch
import torch.nn as nn


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Module.
    
    Dynamically conditions the intermediate spectro-temporal feature maps using 
    physical channel metadata:
        c = [SNR (dB), RMS Delay Spread (ns), Max Doppler Shift (Hz)]
        
    Generates affine transformation parameters (gamma, beta) to scale and shift
    features without modifying the underlying Transformer backbone weights:
        F_modulated = (1 + gamma) * F + beta
    """
    def __init__(self, in_features: int = 3, hidden_dim: int = 32):
        super().__init__()
        # 3-layer compact MLP: maps 3 physical parameters to affine pair (gamma, beta)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )
        
        # Zero-initialize the output projection:
        # Guarantees identity mapping at the start of transfer learning (gamma=0, beta=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feature_map: torch.Tensor, condition_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: Tensor of shape [batch, 120, 14] or [batch, channels, 120, 14]
            condition_vector: Tensor of shape [batch, 3] representing [SNR, DS, Doppler]
        """
        affine_params = self.mlp(condition_vector)  # [batch, 2]
        gamma = affine_params[:, 0]
        beta = affine_params[:, 1]

        # Broadcast scalar scaling and shift across grid dimensions
        while gamma.dim() < feature_map.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        return (1.0 + gamma) * feature_map + beta


class SpectroTemporalEnhancer(nn.Module):
    """
    Shallow 2D Convolutional Enhancer capturing local time-frequency correlation.
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 120, 14]
        return self.net(x)


class FortiTranM_FiLM(nn.Module):
    """
    FortiTranM-FiLM: Deep Hybrid CNN-Transformer with FiLM Transfer Conditioning.
    
    Architecture Overview:
    ----------------------
    1. Learned 2D Linear Upsampler: projects 80 pilot estimates directly to (120 x 14) grid.
    2. Shallow CNN Enhancer: extracts local spectro-temporal features.
    3. Patch Partitioning: decomposes grid into 280 non-overlapping (3 x 2) spatial patches.
    4. Multi-Head Self-Attention: captures global multi-path delay and temporal Doppler fading.
    5. Residual Feature Fusion: combines shallow spatial features with deep attention tokens.
    6. FiLM Transfer Modulation: applies physical metadata conditioning (SNR, DS, Doppler).
    7. Final Reconstructor: outputs high-fidelity complex channel matrix.
    """
    def __init__(
        self,
        num_scs: int = 120,
        num_symbols: int = 14,
        num_pilots: int = 80,
        patch_size: tuple = (3, 2),
        d_model: int = 32,
        nhead: int = 4,
        num_transformer_layers: int = 3,
        use_film: bool = True
    ):
        super().__init__()
        self.num_scs = num_scs
        self.num_symbols = num_symbols
        self.num_pilots = num_pilots
        self.patch_size = patch_size
        self.use_film = use_film
        
        total_elements = num_scs * num_symbols  # 1680
        patch_len = patch_size[0] * patch_size[1]  # 6
        num_patches = total_elements // patch_len  # 280

        # 1. Learned Upsampler (Data-driven 2D Wiener interpolation)
        self.upsampler = nn.Linear(num_pilots, total_elements)

        # 2. Shallow Convolutional Enhancer
        self.shallow_enhancer = SpectroTemporalEnhancer(in_channels=1, hidden_channels=32, out_channels=1)

        # 3. Patch Projection & Positional Embedding
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        # 4. Pure Spatial Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.unpatch_projection = nn.Linear(d_model, patch_len)

        # 5. FiLM Conditioning Block
        if self.use_film:
            self.film_block = FiLMBlock(in_features=3, hidden_dim=32)

        # 6. Final Reconstructor
        self.reconstructor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, pilot_input: torch.Tensor, condition_vector: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pilot_input: Tensor of shape [batch, 80] (vectorized LS estimates at pilots)
            condition_vector: Optional tensor of shape [batch, 3] (SNR, DS, Doppler)
            
        Returns:
            Reconstructed channel frequency response of shape [batch, 120, 14]
        """
        batch_size = pilot_input.shape[0]

        # 1. Linear Upsampling
        up = self.upsampler(pilot_input).view(batch_size, 1, self.num_scs, self.num_symbols)

        # 2. Shallow CNN feature extraction
        f_shallow = self.shallow_enhancer(up)

        # 3. Patching
        # Reshape to [batch, 280, 6]
        p_h, p_w = self.patch_size
        patches = (
            up.squeeze(1)
            .view(batch_size, self.num_scs // p_h, p_h, self.num_symbols // p_w, p_w)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(batch_size, -1, p_h * p_w)
        )

        # 4. Transformer Attention
        tokens = self.patch_projection(patches) + self.pos_embedding
        tokens = self.transformer(tokens)
        deep_patches = self.unpatch_projection(tokens)

        # Unpatching back to 2D spectro-temporal map
        deep_2d = (
            deep_patches.view(batch_size, self.num_scs // p_h, self.num_symbols // p_w, p_h, p_w)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(batch_size, 1, self.num_scs, self.num_symbols)
        )

        # 5. Residual Fusion
        f_fused = f_shallow + deep_2d

        # 6. FiLM Transfer Conditioning
        if self.use_film and condition_vector is not None:
            f_modulated = self.film_block(f_fused, condition_vector)
        else:
            f_modulated = f_fused

        # 7. Final CNN Reconstruction
        out = self.reconstructor(f_modulated).squeeze(1)
        return out
