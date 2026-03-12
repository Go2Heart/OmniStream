import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List

from .layers.block import Block
from .layers.rope3d import RotaryPositionEmbedding3D, PositionGetter3D
from .dinov3.layers import Mlp, PatchEmbed, SwiGLUFFN

from .configuration_omnistream import OmnistreamConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

ffn_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
}

class DinoVideoTransformer(nn.Module):
    """
    The Aggregator applies causal temporal attention over input frames,


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=["global"],
        rope_freq=100,
        init_values=0.01,
        enable_causal_temporal=True,
        attention_type="joint_space_time",
        ffn_layer="mlp",
        window_size=None,  # Sliding window size (in frames). None = no limit
        attention_sink_size=1,  # Number of initial frames to keep as attention sink
        upcycle_to_moe=False,
        num_experts=8,
        top_k=2,
    ):
        super().__init__()

        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding3D(frequency=100) if rope_freq > 0 else None # hard code to freq 100 to align with dinov3
        self.position_getter = PositionGetter3D() if self.rope is not None else None


        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=False,
                    ffn_layer=ffn_dict[ffn_layer],
                    rope=self.rope if self.rope is not None else None,
                )
                for _ in range(depth)
            ]
        )
        
        self.post_layernorm = nn.LayerNorm(embed_dim)
        self.ar_embedder = nn.Linear(1, embed_dim)

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.enable_causal_temporal = enable_causal_temporal
        self.attention_type = attention_type
        self.window_size = window_size
        self.attention_sink_size = attention_sink_size


        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))

        # The patch tokens start after the register tokens
        self.patch_start_idx = 1 + num_register_tokens # CLS, Registers
        # self.cls_projection = nn.Identity() # nn.Linear(embed_dim * 2, embed_dim)

        self.upcycle_to_moe = upcycle_to_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.embed_dim = embed_dim
    
    def _trim_kv_cache(self, past_key_values, num_patches_per_frame):
        """
        Trim KV cache to keep attention sink + sliding window.
        
        Args:
            past_key_values: List of (past_k, past_v) tuples
            num_patches_per_frame: Number of patches (tokens) per frame
        
        Returns:
            Trimmed past_key_values
        """
        if past_key_values is None or len(past_key_values) == 0:
            return past_key_values
        
        if self.window_size is None:
            return past_key_values
        
        # Get current sequence length from KV cache
        past_seq_len = past_key_values[0][0].shape[2]  # [B, num_heads, seq_len, head_dim]
        past_num_frames = past_seq_len // num_patches_per_frame
        
        # If not exceeding window size, no trimming needed
        if past_num_frames <= self.window_size:
            return past_key_values
        
        # Calculate indices to keep
        # Keep: [sink frames] + [recent window frames]
        sink_tokens = self.attention_sink_size * num_patches_per_frame
        
        # Start of window (excluding sink)
        window_start_frame = past_num_frames - self.window_size + self.attention_sink_size
        window_start_token = window_start_frame * num_patches_per_frame
        
        # Trim each layer's KV cache
        trimmed_kv = []
        for past_k, past_v in past_key_values:
            # Keep sink + window
            if sink_tokens > 0:
                k_sink = past_k[:, :, :sink_tokens, :]
                v_sink = past_v[:, :, :sink_tokens, :]
                
                k_window = past_k[:, :, window_start_token:, :]
                v_window = past_v[:, :, window_start_token:, :]
                
                trimmed_k = torch.cat([k_sink, k_window], dim=2)
                trimmed_v = torch.cat([v_sink, v_window], dim=2)
            else:
                trimmed_k = past_k[:, :, window_start_token:, :]
                trimmed_v = past_v[:, :, window_start_token:, :]
            
            trimmed_kv.append((trimmed_k, trimmed_v))
        
        return tuple(trimmed_kv)
    

    def forward(
        self,
        images,
        images_shape,
        attention_mask=None,
        masks=None,
        return_register_tokens=False,
        use_cache=False,
        past_key_values=None,
        logical_frame_idx=None,  # Track the logical frame index when using sliding window
        total_length=16,
        add_camera_token=False,
        aspect_ratio=None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B*T, 3, H, W], in range [0, 1].
                B: batch size, T: sequence length, 3: RGB channels, H: height, W: width
            logical_frame_idx (int): The logical frame index (total frames processed so far).
                Used with sliding window to correctly generate attention masks.

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, T, C_in, H, W = images_shape
        patch_start_idx = self.patch_start_idx
        if add_camera_token:
            patch_start_idx += 1
        num_patches_per_frame = (H // self.patch_size) * (W // self.patch_size) + patch_start_idx
        
        # Calculate past_length from physical cache
        past_length = 0 
        if past_key_values is not None and len(past_key_values) > 0:
            # past_key_values[0] = (past_k, past_v)
            # past_k shape: [B, num_heads, past_seq_len, head_dim]
            past_seq_len = past_key_values[0][0].shape[2]
            past_length = past_seq_len // num_patches_per_frame
        
        # Use logical_frame_idx for attention mask generation if provided
        # Otherwise use physical cache length
        if logical_frame_idx is None:
            logical_frame_idx = past_length
        patch_tokens = self.patch_embed(images).flatten(1, 2) # B*T, N D
        if masks is not None:
            # masks [B, T*N]
            # patch_tokens [B*T, N, D]
            patch_tokens = torch.where(masks.unsqueeze(-1).view(patch_tokens.shape[0], -1, 1), self.mask_token.to(patch_tokens.dtype).unsqueeze(0), patch_tokens)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        
        # cls_tokens = slice_expand_and_flatten(self.cls_token, B, T) # [BxT, D] => [BxT, 1, D]
        cls_tokens = cls_token.expand(B*T, 1, -1)
        register_token = slice_expand_and_flatten(self.register_token, B, T, use_first_frame_token=past_length == 0)
        if add_camera_token:
            camera_token = slice_expand_and_flatten(self.camera_token, B, T, use_first_frame_token=past_length == 0)
            if aspect_ratio is not None:
                log_aspect_ratio = torch.log(aspect_ratio).to(self.camera_token.device).to(self.camera_token.dtype) # log(height / width)
                ar_embedding = self.ar_embedder(log_aspect_ratio.unsqueeze(-1))  # (B, D)
                ar_embedding = ar_embedding.unsqueeze(1).expand(B, T, -1).reshape(B*T, 1, -1)  # (B*T, 1, D)
                camera_token = camera_token + ar_embedding
            tokens = torch.cat([cls_tokens, camera_token, register_token, patch_tokens], dim=1)
        else:
            tokens = torch.cat([cls_tokens, register_token, patch_tokens], dim=1)

        if attention_mask is None:
            if self.enable_causal_temporal and self.attention_type == "joint_space_time":
                current_patches = T * num_patches_per_frame
                total_patches = (past_length + T) * num_patches_per_frame 
                
                ladder_mask = torch.zeros(current_patches, total_patches, device=tokens.device)
                
                for t in range(T):
                    current_start = t * num_patches_per_frame
                    current_end = (t + 1) * num_patches_per_frame
                    
                    # Logical frame index for current frame
                    current_logical_frame = logical_frame_idx + t + 1
                    
                    # Apply sliding window and attention sink
                    if self.window_size is not None and current_logical_frame > self.window_size:
                        # Calculate which frames should be visible in the physical cache
                        # Physical cache contains: [sink frames] + [recent window frames]
                        
                        # Sink frames are always at the beginning of cache
                        sink_end = self.attention_sink_size * num_patches_per_frame
                        ladder_mask[current_start:current_end, :sink_end] = 1
                        
                        # Recent frames are at the end of cache
                        # Calculate how many recent frames (excluding sink) should be visible
                        recent_frames_to_keep = self.window_size - self.attention_sink_size
                        
                        # In physical cache: sink frames come first, then recent frames
                        # Map logical frame range to physical cache positions
                        window_start_in_cache = self.attention_sink_size * num_patches_per_frame
                        
                        # Calculate which recent frames (in logical space) should be visible
                        logical_window_start = current_logical_frame - recent_frames_to_keep
                        logical_window_end = current_logical_frame
                        
                        # In physical cache, recent frames start after sink
                        # and include frames from window_start up to current position
                        cache_window_start = window_start_in_cache
                        cache_window_end = past_length * num_patches_per_frame + (t + 1) * num_patches_per_frame
                        
                        ladder_mask[current_start:current_end, cache_window_start:cache_window_end] = 1
                    else:
                        # No window limit: see all previous frames
                        visible_end = (past_length + t + 1) * num_patches_per_frame
                        ladder_mask[current_start:current_end, :visible_end] = 1
                
                attention_mask = ladder_mask
            else:
                raise ValueError(f"Unknown attention type: {self.attention_type}")
        attention_mask = attention_mask.bool().to(patch_tokens.device)
        _, P, C = patch_tokens.shape

        # tokens = patch_tokens

        pos = None
        if self.rope is not None:
            if use_cache and past_length > 0:
                pos = self.position_getter(B, T, H // self.patch_size, W // self.patch_size, 
                                      device=patch_tokens.device, time_shift=past_length, total_length=total_length)
                # pos[:, :, 0] += past_length
            else:
                pos = self.position_getter(B, T, H // self.patch_size, W // self.patch_size, device=patch_tokens.device, total_length=total_length, rescale=self.training)

        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        present_key_values = [] if use_cache else None

        for _ in range(self.depth):
            layer_past = past_key_values[global_idx] if past_key_values is not None else None

            tokens, global_idx, global_intermediates, layer_present = self._process_global_attention(
                tokens, B, T, P, C, global_idx, pos=pos, attention_mask=attention_mask, patch_start_idx=patch_start_idx, use_cache=use_cache, past_key_value=layer_past
            )
            output_list.append(global_intermediates[0])
            if use_cache:
                present_key_values.append(layer_present)
        del global_intermediates
        norm_output = self.post_layernorm(output_list[-1])
        last_hidden_state = norm_output[:, :, patch_start_idx:, :]
        reg_start_idx = 2 if add_camera_token else 1
        register_tokens = norm_output[:, :, reg_start_idx:patch_start_idx, :] # cls x 1, camera x 1, register tokens x 4
        pooler_output = norm_output[:, :, 0, :] # normal cls output without mean pooling
        camera_token = norm_output[:, :, 1, :] if add_camera_token else None
        if use_cache and present_key_values is not None:
            present_key_values = list(self._trim_kv_cache(tuple(present_key_values), num_patches_per_frame))
        
        result = [output_list, camera_token, last_hidden_state, pooler_output, patch_start_idx]
        if return_register_tokens:
            result.append(register_tokens)
        if use_cache:
            result.append(tuple(present_key_values))
            # Return updated logical frame index
            result.append(logical_frame_idx + T)
        
        return tuple(result)

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, attention_mask=None, patch_start_idx=0, use_cache=False, past_key_value=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        intermediates = []
        present_key_values = []
        if self.training:
            tokens = checkpoint(self.blocks[global_idx], tokens, pos, attention_mask, patch_start_idx, use_reentrant=False)
            block_present = None
            # tokens = self.blocks[global_idx](tokens, pos=pos, attention_mask=attention_mask, patch_start_idx=patch_start_idx)
        else:
            result = self.blocks[global_idx](tokens, pos=pos, attention_mask=attention_mask, patch_start_idx=patch_start_idx, use_cache=use_cache, past_key_value=past_key_value)
            if use_cache:
                tokens, block_present = result
            else:
                tokens = result
                block_present = None
        global_idx += 1
        intermediates.append(tokens.view(B, S, P, C))
        if use_cache:
            present_key_values.append(block_present)
        final_present = present_key_values[-1] if use_cache and len(present_key_values) > 0 else None

        return tokens, global_idx, intermediates, final_present
    
def slice_expand_and_flatten(token_tensor, B, S, use_first_frame_token=True):
    if use_first_frame_token or S > 1:
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
        others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
        combined = torch.cat([query, others], dim=1)
    else:
        combined = token_tensor[:, 1:2, ...].expand(B, S, *token_tensor.shape[2:])
    
    combined = combined.view(B * S, *combined.shape[2:])
    return combined




class OmnistreamPretrainedModel(PreTrainedModel):
    config_class = OmnistreamConfig
    supports_gradient_checkpointing = True
    base_model_prefix = "vfm"
    main_input_name = "pixel_values"
    
    def _init_weights(self, module):
        if hasattr(module, '_is_pretrained') and module._is_pretrained:
            return
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=.02)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, val=1.0)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        else:
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.dim() > 1:
                    nn.init.kaiming_uniform_(module.weight)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)



class OmnistreamMultiFrameTransformer(OmnistreamPretrainedModel):
    def __init__(self, config, window_size=None, attention_sink_size=1):
        super().__init__(config)
        self.config = config

        self.patch_embed = DinoVideoTransformer(
            img_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.hidden_size,
            depth=config.video_aggregator_layer,
            num_heads=config.num_attention_heads,
            window_size=window_size,
            attention_sink_size=attention_sink_size,
            mlp_ratio=config.mlp_ratio,
            ffn_layer=config.ffn_layer,
            upcycle_to_moe=config.upcycle_to_moe,
            num_experts=config.num_experts,
            top_k=config.top_k,
        )

    def forward(self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multi_task_input: Optional[dict] = None,
        output_aggregators = None, # for downstream feature extraction
        return_features = None, # only for segmentation head
        use_cache=False,
        past_key_values=None,
        logical_frame_idx=None,
        total_length=16,
        add_camera_token=False,
        aspect_ratio=None,
        **kwargs,
    ):
        B, S, C_in, H, W = pixel_values.shape
        # import ipdb; ipdb.set_trace()
        if S > total_length:
            total_length = S # ensure temporal rope works
        images = pixel_values.view(B * S, C_in, H, W)

        if use_cache:
            video_hidden_states, camera_token, last_hidden_state, pooler_output, video_patch_start_idx, present_key_values, logical_frame_idx = self.patch_embed(
                images, 
                (B, S, C_in, H, W),
                use_cache=use_cache,
                past_key_values=past_key_values,
                logical_frame_idx=logical_frame_idx,
                total_length=total_length,
                add_camera_token=add_camera_token,
                aspect_ratio=aspect_ratio,
            )
        else:
            video_hidden_states, camera_token, last_hidden_state, pooler_output, video_patch_start_idx = self.patch_embed(
                images, 
                (B, S, C_in, H, W),
                use_cache=use_cache,
                past_key_values=past_key_values,
                add_camera_token=add_camera_token,
                total_length=total_length,
                aspect_ratio=aspect_ratio,
                )
        video_backbone_outputs = ModelOutput(
            last_hidden_state=last_hidden_state, # (B, T, N, D)
            pooler_output=pooler_output, # (B, T, D)
            camera_token=camera_token, # (B, T, D)
            hidden_states=video_hidden_states, # Lx (B, T, N, D)
            patch_start_idx=video_patch_start_idx,
        )

        if not return_dict: 
            return tuple(output for output in [video_backbone_outputs] if output is not None)
        return video_backbone_outputs