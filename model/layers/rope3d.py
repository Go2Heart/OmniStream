import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

MAX_FRAMES = 128

class PositionGetter3D:
    """Generates and caches 3D spatial-temporal positions for patches in a 3D grid.
    
    This class efficiently manages the generation of spatial-temporal coordinates for patches
    in a 3D grid (T, H, W), caching results to avoid redundant computations.
    
    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
        normalize_coords: Normalization mode for coordinates.
    """
    
    def __init__(self, normalize_coords: str = "separate", rescale_coords = 2.0):
        """Initializes the 3D position generator with an empty cache.
        
        Args:
            normalize_coords: How to normalize coordinates. Options:
                - 'separate': normalize t, h, and w independently to [-1, 1]
                - 'max': normalize by max(T, H, W)
                - 'min': normalize by min(T, H, W)
        """
        self.position_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self.normalize_coords = normalize_coords
        self.rescale_coords = rescale_coords
    def __call__(self, batch_size: int, time_steps: int, height: int, width: int, 
             device: torch.device, time_shift: int = 0, total_length: int = 16, rescale: bool = False) -> torch.Tensor:
        """Generates 3D spatial-temporal positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            time_steps: Number of time steps (T dimension).
            height: Height of the spatial grid (Y dimension).
            width: Width of the spatial grid (X dimension).
            device: Target device for the position tensor.
            time_shift: Time offset for incremental processing (KV cache mode).

        Returns:
            Tensor of shape (batch_size, T*H*W, 3) containing normalized t,y,x coordinates
            in range [-1, +1] for each position in the 3D grid.
        """
        # 关键修复：cache_key必须包含time_shift！
        cache_key = (time_steps, height, width, self.normalize_coords, time_shift, total_length)
        
        if cache_key not in self.position_cache:
            # Generate normalized coordinates in range [-1, +1]
            if self.normalize_coords == "max":
                max_THW = max(time_steps, height, width)
                coords_t = (torch.arange(0.5, time_steps, device=device) / max_THW)
                coords_h = (torch.arange(0.5, height, device=device) / max_THW)
                coords_w = (torch.arange(0.5, width, device=device) / max_THW)
            elif self.normalize_coords == "min":
                min_THW = min(time_steps, height, width)
                coords_t = (torch.arange(0.5, time_steps, device=device) / min_THW)
                coords_h = (torch.arange(0.5, height, device=device) / min_THW)
                coords_w = (torch.arange(0.5, width, device=device) / min_THW)
            elif self.normalize_coords == "separate":
                # time indices: [time_shift, time_shift+1, ..., time_shift+time_steps-1]
                coords_t = ((torch.arange(time_shift, time_shift + time_steps, device=device) / total_length) + 1) / 2 # new rope
                # coords_t = ((torch.arange(time_shift, time_shift+time_steps, device=device) / total_length + 1/32) + 1) / 2 # for old ckpts
                coords_h = (torch.arange(0.5, height, device=device) / height)
                coords_w = (torch.arange(0.5, width, device=device) / width)
            else:
                raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
            
            positions = torch.cartesian_prod(coords_t, coords_h, coords_w)
            # Shift from [0, 1] to [-1, +1]
            positions = 2.0 * positions - 1.0
            self.position_cache[cache_key] = positions
        
        cached_positions = self.position_cache[cache_key].clone()
        if rescale:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, device=device, dtype=cached_positions.dtype).uniform_(rescale_min, rescale_max).exp() # use same scale for y and x
            rescale_t = torch.empty(1, device=device, dtype=cached_positions.dtype).uniform_(rescale_min, rescale_max).exp()
            rescale_thw = torch.cat([rescale_t, rescale_hw, rescale_hw], dim=0)
            cached_positions *= rescale_thw
        total_tokens = time_steps * height * width
        return cached_positions.view(1, total_tokens, 3).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding3D(nn.Module):
    """3D Rotary Position Embedding with interleaved temporal encoding.
    
    This module extends 2D RoPE by interleaving temporal (t) encodings at positions
    where index % 4 == 3, while preserving the original y and x encodings at other positions.
    Like y and x, t is also repeated twice for consistency.
    
    Pattern: Keep [y_0...y_D/4-1, x_0...x_D/4-1] repeated twice, but replace positions
    at indices 3, 7, 11, 15, ... (i.e., i % 4 == 3) with [t_0...t_D/8-1] repeated twice.
    
    Args:
        frequency: Base frequency (period) for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0
        use_periods: If True, use period-based computation like RopePositionEmbedding.
                    If False, use inverse frequency. Default: True
    """
    
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0, use_periods: bool = True):
        """Initializes the 3D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.use_periods = use_periods
        self.frequency_cache: Dict[Tuple, torch.Tensor] = {}
    
    def _compute_frequency_components(
        self, dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Computes period components for rotary embeddings.
        
        Args:
            dim: Feature dimension for this coordinate.
            device: Target device for computations.
            dtype: Data type for the computed tensors.
            
        Returns:
            periods tensor for angle computation.
        """
        cache_key = (dim, device, dtype, self.use_periods)
        if cache_key not in self.frequency_cache:
            if self.use_periods:
                # Use period-based computation like RopePositionEmbedding
                # For a dimension of size dim, we need dim//2 periods
                # periods = base ^ (2 * i / dim) for i in [0, dim//2)
                exponents = 2 * torch.arange(dim // 2, device=device, dtype=dtype) / dim
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
    
    def forward(self, tokens: torch.Tensor, positions: torch.Tensor, prefix: int = 0, 
                prefix_per_frame: int = None) -> torch.Tensor:
        """Applies 3D rotary position embeddings to input tokens.
        
        This implementation preserves the 2D RoPE structure but interleaves temporal
        encodings at positions where index % 4 == 3. Like y and x, t is repeated twice.
        
        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, D_head).
                   The feature dimension (D_head) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_patches, 3) containing
                      normalized [t, y, x] coordinates in range [-1, +1] for each patch token 
                      (excluding prefix tokens).
            prefix: Number of prefix tokens at the beginning to skip (for single-frame mode).
                   These tokens will not be modified by RoPE. Default: 0.
            prefix_per_frame: If not None, indicates the number of prefix tokens per frame
                            (for multi-frame video mode). In this case, the token sequence is:
                            [frame0_prefix, frame0_patches, frame1_prefix, frame1_patches, ...]
                            If None, uses the original mode with all prefixes at the beginning.
                      
        Returns:
            Tensor of same shape as input with applied 3D rotary position embeddings.
            Prefix tokens remain unchanged.
            
        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
            
        Example:
            >>> # For D_head=64:
            >>> # Original 2D: [y_0...y_15, x_0...x_15] each repeated twice
            >>> # New 3D: [t_0...t_7] repeated twice, interleaved at positions % 4 == 3
            >>> # Index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 ... 32 33 34 35 ...
            >>> # Result: y0 y1 y2 t0 y4 y5 y6 t1 y8 y9 y10 t2 ... ... x0 x1 x2 t0 ...
        """
        import math
        
        # Validate inputs
        B, n_heads, n_tokens, D_head = tokens.shape
        assert D_head % 4 == 0, "Feature dimension must be divisible by 4"
        assert positions.ndim == 3 and positions.shape[-1] == 3, \
            "Positions must have shape (batch_size, n_patches, 3)"
        assert prefix >= 0, f"Prefix must be non-negative, got {prefix}"
        
        # Handle two different token layout modes
        if prefix_per_frame is None:
            # Original mode: all prefixes at the beginning
            assert prefix + positions.shape[1] == n_tokens, \
                f"Mismatch: prefix ({prefix}) + positions ({positions.shape[1]}) != n_tokens ({n_tokens})"
            
            if prefix == 0:
                tokens_to_process = tokens
                positions_to_use = positions
            else:
                # Split prefix and patch tokens
                tokens_to_process = tokens[:, :, prefix:, :]  # [B, heads, n_patches, D_head]
                positions_to_use = positions  # [B, n_patches, 3]
            
            # Get period components - use same reference as 2D (D_head // 2)
            # This ensures compatibility with pretrained 2D weights
            periods = self._compute_frequency_components(D_head // 2, tokens.device, tokens.dtype)
            # periods shape: [D_head // 4]
            
            # Extract coordinates: positions_to_use is [B, n_patches, 3] with [t, y, x]
            coord_t = positions_to_use[:, :, 0:1]  # [B, n_patches, 1]
            coord_y = positions_to_use[:, :, 1:2]  # [B, n_patches, 1]
            coord_x = positions_to_use[:, :, 2:3]  # [B, n_patches, 1]
            
            # Compute angles for each coordinate
            # Compute angles_y
            angles_y = 2 * math.pi * coord_y[:, :, :, None] / periods[None, None, None, :]
            angles_y = angles_y.flatten(2, 3)  # [B, n_patches, D_head//4]
            angles_y = angles_y.repeat(1, 1, 2)  # [B, n_patches, D_head//2]
            
            # Compute angles_x
            angles_x = 2 * math.pi * coord_x[:, :, :, None] / periods[None, None, None, :]
            angles_x = angles_x.flatten(2, 3)  # [B, n_patches, D_head//4]
            angles_x = angles_x.repeat(1, 1, 2)  # [B, n_patches, D_head//2]
            
            # For t: only compute D_head//8 unique values (will be repeated twice)
            periods_t = periods[:D_head//8]  # [D_head // 8]
            angles_t = 2 * math.pi * coord_t[:, :, :, None] / periods_t[None, None, None, :]
            angles_t = angles_t.flatten(2, 3)  # [B, n_patches, D_head//8]
            angles_t = angles_t.repeat(1, 1, 2)  # [B, n_patches, D_head//4] - repeated twice
            
            # Concatenate [y, x] as in 2D version
            angles_yx = torch.cat([angles_y, angles_x], dim=2)  # [B, n_patches, D_head]
            
            # Create interleaved pattern
            # Memory optimization: use in-place operation instead of clone
            angles = angles_yx  # Reuse the tensor instead of cloning
            
            # Create mask for positions to replace (index % 4 == 3)
            t_target_indices = torch.arange(D_head, device=tokens.device)[torch.arange(D_head, device=tokens.device) % 4 == 3]
            
            # Replace with t encodings
            angles[:, :, t_target_indices] = angles_t
            
            # Add head dimension for broadcasting
            # [B, n_patches, D_head] -> [B, 1, n_patches, D_head]
            angles = angles.unsqueeze(1)
            
            # Compute cos and sin
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            
            # Apply rotation: x * cos + rotate_half(x) * sin
            processed_tokens = tokens_to_process * cos + self._rotate_features(tokens_to_process) * sin
            
            # Concatenate back with prefix if needed
            if prefix == 0:
                return processed_tokens
            else:
                prefix_tokens = tokens[:, :, :prefix, :]  # Unchanged prefix tokens
                return torch.cat([prefix_tokens, processed_tokens], dim=2)
        
        else:
            # Video mode: prefixes are interleaved with patches
            # Token layout: [frame0_prefix, frame0_patches, frame1_prefix, frame1_patches, ...]
            assert prefix_per_frame > 0, f"prefix_per_frame must be positive, got {prefix_per_frame}"
            
            n_patches = positions.shape[1]  # Total number of patch tokens (T*H*W)
            # Calculate number of frames
            # From the constraint: n_tokens = T * (prefix_per_frame + patches_per_frame)
            # and n_patches = T * patches_per_frame
            # We get: n_tokens = T * prefix_per_frame + n_patches
            # Therefore: T = (n_tokens - n_patches) / prefix_per_frame
            
            assert (n_tokens - n_patches) % prefix_per_frame == 0, \
                f"Invalid token layout: (n_tokens={n_tokens} - n_patches={n_patches}) must be divisible by prefix_per_frame={prefix_per_frame}"
            
            num_frames = (n_tokens - n_patches) // prefix_per_frame
            assert num_frames > 0, f"Calculated num_frames={num_frames} must be positive"
            
            patches_per_frame = n_patches // num_frames
            assert patches_per_frame * num_frames == n_patches, \
                f"Patch count mismatch: patches_per_frame={patches_per_frame} * num_frames={num_frames} != n_patches={n_patches}"
            
            # Get period components
            periods = self._compute_frequency_components(D_head // 2, tokens.device, tokens.dtype)
            
            # Extract coordinates
            coord_t = positions[:, :, 0:1]  # [B, T*H*W, 1]
            coord_y = positions[:, :, 1:2]  # [B, T*H*W, 1]
            coord_x = positions[:, :, 2:3]  # [B, T*H*W, 1]
            
            # Compute angles for each coordinate
            angles_y = 2 * math.pi * coord_y[:, :, :, None] / periods[None, None, None, :]
            angles_y = angles_y.flatten(2, 3).repeat(1, 1, 2)  # [B, T*H*W, D_head//2]
            
            angles_x = 2 * math.pi * coord_x[:, :, :, None] / periods[None, None, None, :]
            angles_x = angles_x.flatten(2, 3).repeat(1, 1, 2)  # [B, T*H*W, D_head//2]
            
            periods_t = periods[:D_head//8]
            angles_t = 2 * math.pi * coord_t[:, :, :, None] / periods_t[None, None, None, :]
            angles_t = angles_t.flatten(2, 3).repeat(1, 1, 2)  # [B, T*H*W, D_head//4]
            
            # Create angles
            angles_yx = torch.cat([angles_y, angles_x], dim=2)  # [B, T*H*W, D_head]
            angles = angles_yx
            t_target_indices = torch.arange(D_head, device=tokens.device)[torch.arange(D_head, device=tokens.device) % 4 == 3]
            angles[:, :, t_target_indices] = angles_t
            
            # Now we need to apply RoPE only to patch tokens, skipping prefix tokens
            # Build output tensor by processing frame by frame
            tokens_reshaped = tokens.view(B, n_heads, num_frames, prefix_per_frame + patches_per_frame, D_head)

            prefix_tokens = tokens_reshaped[:, :, :, :prefix_per_frame, :]  # [B, heads, T, prefix, D]
            patch_tokens = tokens_reshaped[:, :, :, prefix_per_frame:, :]   # [B, heads, T, patches, D]

            angles_reshaped = angles.view(B, num_frames, patches_per_frame, D_head).unsqueeze(1)
            original_dtype = patch_tokens.dtype
            cos = torch.cos(angles_reshaped).to(original_dtype)
            sin = torch.sin(angles_reshaped).to(original_dtype)
            processed = patch_tokens * cos + self._rotate_features(patch_tokens) * sin

            output_reshaped = torch.cat([prefix_tokens, processed], dim=3)  # [B, heads, T, prefix+patches, D]

            output_tokens = output_reshaped.view(B, n_heads, n_tokens, D_head)
            return output_tokens