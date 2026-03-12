# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TimeSformer model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class OmnistreamConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VFMModel`]. 

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_frames (`int`, *optional*, defaults to 8):
            The number of frames in each video.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        attention_type (`str`, *optional*, defaults to `"divided_space_time"`):
            The attention type to use. Must be one of `"divided_space_time"`, `"space_only"`, `"joint_space_time"`.
        drop_path_rate (`float`, *optional*, defaults to 0):
            The dropout ratio for stochastic depth.
    ```"""

    model_type = "vfm"

    def __init__(
        self,
        image_size=512,
        patch_size=16,
        num_channels=3,
        num_frames=16,
        hidden_size=1024,
        
        mlp_ratio=4.0,
        ffn_layer="mlp",
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        attention_type="divided_space_time",
        drop_path_rate=0,
        enable_causal_temporal=False,
        position_embedding="sinusoidal",
        init_values=0.01,
        patch_embed_name="dinov2_vitb14_reg",
        chosen_layers=[1, 2, 4, 5],
        geometry_aggregator=True,
        video_aggregator=True,
        image_aggregator=True,
        geometry_aggregator_layer=6,
        video_aggregator_layer=6,
        image_aggregator_layer=6,
        grounding_ratio=0.5,
        upcycle_to_moe=False,
        num_experts=8,
        top_k=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.patch_embed_name = patch_embed_name
        self.init_values = init_values
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.chosen_layers = chosen_layers
        self.mlp_ratio = mlp_ratio
        self.ffn_layer = ffn_layer
        self.geometry_aggregator = geometry_aggregator
        self.geometry_aggregator_layer = geometry_aggregator_layer
        
        self.video_aggregator = video_aggregator
        self.video_aggregator_layer = video_aggregator_layer
        
        self.image_aggregator = image_aggregator
        self.image_aggregator_layer = image_aggregator_layer
        
        self.grounding_ratio = grounding_ratio
        self.upcycle_to_moe = upcycle_to_moe
        self.num_experts = num_experts
        self.top_k = top_k
        # self.num_hidden_layers = num_hidden_layers

        # self.hidden_dropout_prob = hidden_dropout_prob
        # self.attention_probs_dropout_prob = attention_probs_dropout_prob
        
        # self.qkv_bias = qkv_bias

        # self.attention_type = attention_type
        # self.drop_path_rate = drop_path_rate
        # self.enable_causal_temporal = enable_causal_temporal
        # self.position_embedding = position_embedding
        self.image_ssl = {
            "crops": {
                "local_crops_number": 2,
            },
            "ibot": {
                "separate_head": True,
                "loss_weight": 1.0,
                "mask_sample_probability": 0.5,
                "mask_ratio_min_max": [0.1, 0.5],
                "mask_random_circular_shift": False,
                "force_masking_even_with_zero_weight": False,
                "head_n_prototypes": 65536,
                "head_bottleneck_dim": 256,
                "head_norm_last_layer": False,
                "head_nlayers": 3,
                "head_hidden_dim": 2048,    
            },
            "dino": {
                "loss_weight": 1.0,
                "global_ignore_diagonal": True,
                "head_n_prototypes": 65536,
                "head_bottleneck_dim": 256,
                "head_norm_last_layer": False,
                "head_nlayers": 3,
                "head_hidden_dim": hidden_size*2, #2048,
                "koleo_loss_weight": 0.1,
                "koleo_loss_distributed": False,
                "koleo_topk": 1,
                "koleo_distributed_replicas": 0,
                "force_weight_norm": False,
                
                "reweight_dino_local_loss": False,
                "local_loss_weight_schedule": {
                    "start": 0.5,
                    "peak": 0.5,
                    "end": 0.5,
                    "warmup_epochs": 0,
                },
            },
            "gram": {
                "use_loss": True,
                "compute_stats": False,
                "loss_weight": 1.0,
                "ema_teacher": False,
                "ckpt": None,
                "it_load_ema_teacher": -1,
                "rep_update": True,
                "update_frequency": 50000,
                "it_first_update": 0,
                "max_updates": None,
                "normalized": True,
                "img_level": True,
                "remove_neg": False,
                "remove_only_teacher_neg": False,
                "tokens_used": "all",
                "global_teacher_resize_method": "bicubic",
                "global_teacher_resize_antialias": False,
                "loss_weight_schedule": None,
                # {
                #     "start": 0,
                #     "peak": 0,
                #     "end": 2.0,
                #     "warmup_epochs": 1000,
                #     "cosine_epochs": 1,
                # },
            },
            "train": {
                "centering": "sinkhorn_knopp",
            },
            "compute_precision": {
                "sharding_strategy": "SHARD_GRAD_OP",
            },
            "distillation": {
                "enabled": False,
                "full_cfg_path": "",
                "checkpoint_path": "",
            },
            "multidistillation": {
                "enabled": False,
            }
        }
        