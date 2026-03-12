# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class PositionGetter2D:
    """Generates and caches 2D spatial positions for patches in a grid.
    
    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.
    
    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
        normalize_coords: Normalization mode ('separate', 'max', 'min')
    """
    
    def __init__(self, normalize_coords: str = "separate"):
        """Initializes the position generator with an empty cache.
        
        Args:
            normalize_coords: How to normalize coordinates. Options:
                - 'separate': normalize h and w independently to [-1, 1]
                - 'max': normalize by max(H, W)
                - 'min': normalize by min(H, W)
        """
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self.normalize_coords = normalize_coords
    
    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.
        
        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.
            
        Returns:
            Tensor of shape (batch_size, height*width, 2) containing normalized y,x coordinates
            in range [-1, +1] for each position in the grid.
        """
        cache_key = (height, width, self.normalize_coords)
        if cache_key not in self.position_cache:
            # Generate normalized coordinates in range [-1, +1]
            if self.normalize_coords == "max":
                max_HW = max(height, width)
                coords_h = (torch.arange(0.5, height, device=device) / max_HW)
                coords_w = (torch.arange(0.5, width, device=device) / max_HW)
            elif self.normalize_coords == "min":
                min_HW = min(height, width)
                coords_h = (torch.arange(0.5, height, device=device) / min_HW)
                coords_w = (torch.arange(0.5, width, device=device) / min_HW)
            elif self.normalize_coords == "separate":
                coords_h = (torch.arange(0.5, height, device=device) / height)
                coords_w = (torch.arange(0.5, width, device=device) / width)
            else:
                raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
            
            positions = torch.cartesian_prod(coords_h, coords_w)
            # Shift from [0, 1] to [-1, +1]
            positions = 2.0 * positions - 1.0
            self.position_cache[cache_key] = positions
        
        cached_positions = self.position_cache[cache_key]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

# class PositionGetter2D:
#     """Generates and caches 2D spatial positions for patches in a grid.

#     This class efficiently manages the generation of spatial coordinates for patches
#     in a 2D grid, caching results to avoid redundant computations.

#     Attributes:
#         position_cache: Dictionary storing precomputed position tensors for different
#             grid dimensions.
#     """

#     def __init__(self):
#         """Initializes the position generator with an empty cache."""
#         self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

#     def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
#         """Generates spatial positions for a batch of patches.

#         Args:
#             batch_size: Number of samples in the batch.
#             height: Height of the grid in patches.
#             width: Width of the grid in patches.
#             device: Target device for the position tensor.

#         Returns:
#             Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
#             for each position in the grid, repeated for each batch item.
#         """
#         if (height, width) not in self.position_cache:
#             y_coords = torch.arange(height, device=device)
#             x_coords = torch.arange(width, device=device)
#             positions = torch.cartesian_prod(y_coords, x_coords)
#             self.position_cache[height, width] = positions

#         cached_positions = self.position_cache[height, width]
#         return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation with normalized coordinates.
    
    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions using normalized coordinates in range [-1, +1].
    
    Args:
        frequency: Base frequency (period) for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0
        use_periods: If True, use period-based computation like RopePositionEmbedding.
                    If False, use inverse frequency. Default: True
    """
    
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0, use_periods: bool = True):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.use_periods = use_periods
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def _compute_frequency_components(
        self, dim: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes period components for rotary embeddings.
        
        Args:
            dim: Feature dimension per head (D_head).
            device: Target device for computations.
            dtype: Data type for the computed tensors.
            
        Returns:
            periods tensor for angle computation.
        """
        cache_key = (dim, device, dtype, self.use_periods)
        if cache_key not in self.frequency_cache:
            if self.use_periods:
                # Use period-based computation like RopePositionEmbedding
                # For D_head dimensions, we need D_head // 4 periods
                # periods = base ^ (2 * i / (D_head // 2)) for i in [0, D_head//4)
                exponents = 2 * torch.arange(dim // 4, device=device, dtype=dtype) / (dim // 2)
                periods = self.base_frequency ** exponents
            else:
                # Original inverse frequency approach
                exponents = torch.arange(0, dim, 2, device=device).float() / dim
                periods = self.base_frequency ** exponents
            
            self.frequency_cache[cache_key] = periods
        
        return self.frequency_cache[cache_key]
    
    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.
        
        Args:
            x: Input tensor to rotate.
            
        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, tokens: torch.Tensor, positions: torch.Tensor, prefix_per_frame: int = 0) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.
        
        This implementation matches DINOv3's RopePositionEmbedding.
        The feature arrangement is: [y_0, ..., y_D//4-1, x_0, ..., x_D//4-1] repeated twice.
        
        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, D_head).
                   The feature dimension (D_head) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_patches, 2) containing
                      normalized y and x coordinates in range [-1, +1] for each patch token 
                      (excluding prefix tokens).
            prefix: Number of prefix tokens (e.g., CLS, register tokens) to skip.
                   These tokens will not be modified by RoPE. Default: 0.
                      
        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.
            Prefix tokens remain unchanged.
            
        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
            
        Example:
            >>> # tokens: [B, heads, 19, D_head] = [B, heads, 3_special + 16_patches, D_head]
            >>> # positions: [B, 16, 2] for the 16 patch tokens only (normalized coords)
            >>> output = rope(tokens, positions, prefix=3)
            >>> # First 3 tokens unchanged, remaining 16 have RoPE applied
        """
        import math
        
        # Validate inputs
        B, n_heads, n_tokens, D_head = tokens.shape
        assert D_head % 4 == 0, "Feature dimension must be divisible by 4"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_patches, 2)"
        assert prefix_per_frame >= 0, f"Prefix must be non-negative, got {prefix_per_frame}"
        assert prefix_per_frame + positions.shape[1] == n_tokens, \
            f"Mismatch: prefix ({prefix_per_frame}) + positions ({positions.shape[1]}) != n_tokens ({n_tokens})"
        
        # If no prefix, apply RoPE to all tokens
        if prefix_per_frame == 0:
            tokens_to_process = tokens
            positions_to_use = positions
        else:
            # Split prefix and patch tokens
            tokens_to_process = tokens[:, :, prefix_per_frame:, :]  # [B, heads, n_patches, D_head]
            positions_to_use = positions  # [B, n_patches, 2]
        
        # Get period components for D_head dimensions
        periods = self._compute_frequency_components(D_head, tokens.device, tokens.dtype)
        # periods shape: [D_head // 4]
        
        # Compute angles for both y and x coordinates
        # positions_to_use: [B, n_patches, 2], periods: [D_head//4]
        # angles: [B, n_patches, 2, D_head//4]
        angles = 2 * math.pi * positions_to_use[:, :, :, None] / periods[None, None, None, :]
        
        # Flatten the coordinate and frequency dimensions
        # [B, n_patches, 2, D_head//4] -> [B, n_patches, D_head//2]
        angles = angles.flatten(2, 3)
        
        # Tile to get full D_head dimensions
        # [B, n_patches, D_head//2] -> [B, n_patches, D_head]
        angles = angles.tile(1, 1, 2)
        
        # Add head dimension for broadcasting
        # [B, n_patches, D_head] -> [B, 1, n_patches, D_head]
        angles = angles.unsqueeze(1)
        
        # Compute cos and sin
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        processed_tokens = tokens_to_process * cos + self._rotate_features(tokens_to_process) * sin
        
        # Concatenate back with prefix if needed
        if prefix_per_frame == 0:
            return processed_tokens
        else:
            prefix_tokens = tokens[:, :, :prefix_per_frame, :]  # Unchanged prefix tokens
            return torch.cat([prefix_tokens, processed_tokens], dim=2)
# class RotaryPositionEmbedding2D(nn.Module):
#     """2D Rotary Position Embedding implementation.

#     This module applies rotary position embeddings to input tokens based on their
#     2D spatial positions. It handles the position-dependent rotation of features
#     separately for vertical and horizontal dimensions.

#     Args:
#         frequency: Base frequency for the position embeddings. Default: 100.0
#         scaling_factor: Scaling factor for frequency computation. Default: 1.0

#     Attributes:
#         base_frequency: Base frequency for computing position embeddings.
#         scaling_factor: Factor to scale the computed frequencies.
#         frequency_cache: Cache for storing precomputed frequency components.
#     """

#     def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
#         """Initializes the 2D RoPE module."""
#         super().__init__()
#         self.base_frequency = frequency
#         self.scaling_factor = scaling_factor
#         self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

#     def _compute_frequency_components(
#         self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Computes frequency components for rotary embeddings.

#         Args:
#             dim: Feature dimension (must be even).
#             seq_len: Maximum sequence length.
#             device: Target device for computations.
#             dtype: Data type for the computed tensors.

#         Returns:
#             Tuple of (cosine, sine) tensors for frequency components.
#         """
#         cache_key = (dim, seq_len, device, dtype)
#         if cache_key not in self.frequency_cache:
#             # Compute frequency bands
#             exponents = torch.arange(0, dim, 2, device=device).float() / dim
#             inv_freq = 1.0 / (self.base_frequency**exponents)

#             # Generate position-dependent frequencies
#             positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
#             angles = torch.einsum("i,j->ij", positions, inv_freq)

#             # Compute and cache frequency components
#             angles = angles.to(dtype)
#             angles = torch.cat((angles, angles), dim=-1)
#             cos_components = angles.cos().to(dtype)
#             sin_components = angles.sin().to(dtype)
#             self.frequency_cache[cache_key] = (cos_components, sin_components)

#         return self.frequency_cache[cache_key]

#     @staticmethod
#     def _rotate_features(x: torch.Tensor) -> torch.Tensor:
#         """Performs feature rotation by splitting and recombining feature dimensions.

#         Args:
#             x: Input tensor to rotate.

#         Returns:
#             Rotated feature tensor.
#         """
#         feature_dim = x.shape[-1]
#         x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
#         return torch.cat((-x2, x1), dim=-1)

#     def _apply_1d_rope(
#         self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
#     ) -> torch.Tensor:
#         """Applies 1D rotary position embeddings along one dimension.

#         Args:
#             tokens: Input token features.
#             positions: Position indices.
#             cos_comp: Cosine components for rotation.
#             sin_comp: Sine components for rotation.

#         Returns:
#             Tokens with applied rotary position embeddings.
#         """
#         # Embed positions with frequency components
#         cos = F.embedding(positions, cos_comp)[:, None, :, :]
#         sin = F.embedding(positions, sin_comp)[:, None, :, :]

#         # Apply rotation
#         return (tokens * cos) + (self._rotate_features(tokens) * sin)

#     def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
#         """Applies 2D rotary position embeddings to input tokens.

#         Args:
#             tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
#                    The feature dimension (dim) must be divisible by 4.
#             positions: Position tensor of shape (batch_size, n_tokens, 2) containing
#                       the y and x coordinates for each token.

#         Returns:
#             Tensor of same shape as input with applied 2D rotary position embeddings.

#         Raises:
#             AssertionError: If input dimensions are invalid or positions are malformed.
#         """
#         # Validate inputs
#         assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
#         assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

#         # Compute feature dimension for each spatial direction
#         feature_dim = tokens.size(-1) // 2

#         # Get frequency components
#         max_position = int(positions.max()) + 1
#         cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

#         # Split features for vertical and horizontal processing
#         vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

#         # Apply RoPE separately for each dimension
#         vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
#         horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

#         # Combine processed features
#         return torch.cat((vertical_features, horizontal_features), dim=-1)
