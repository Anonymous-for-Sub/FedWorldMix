#!/usr/bin/env python

# Copyright 2024 Lirui Wang and The HuggingFace Inc. team. All rights reserved.
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
"""Heterogeneous Pre-trained Transformer Policy

As per Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers (https://liruiw.github.io/hpt/).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

from collections import deque
from functools import partial
from typing import Callable, List, Optional, Tuple

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, einsum, nn

from .configuration_hpt import HPTConfig
from .normalize import Normalize, Unnormalize
from .utils import get_device_from_parameters, populate_queues

LOSS = partial(F.smooth_l1_loss, beta=0.05)
INIT_CONST = 0.02
_LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)
_CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
_IOSP_CLIP_GLOBAL_CACHE: dict[tuple[str, str], object] = {}


class HPTPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "hpt"],
):
    """
    Heterogeneous Pre-trained Transformer Policy as per Scaling Proprioceptive-Visual Learning
    with Heterogeneous Pre-trained Transformers  (paper: https://liruiw.github.io/hpt/, code: https://github.com/liruiw/HPT-Transfer)
    """

    name = "hpt"

    def __init__(
        self,
        config: HPTConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = HPTConfig()

        self.config: HPTConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.model = HPT(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {}
        for key in self.config.output_shapes:
            self._queues[key] = deque(maxlen=self.config.n_action_steps)
        for key in self.config.input_shapes:
            self._queues[key] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        batch = self.normalize_inputs(batch)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.model.generate_actions(batch)[:, : self.config.n_action_steps]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss = self.model.compute_loss(batch)
        loss_dict = {"loss": loss}
        return loss_dict


class HPT(nn.Module):
    """Heterogeneous Pre-trained Transformer: The underlying neural network for HPTPolicy.

    Note: In this code we use the terms `stem`, 'trunk', `head`. The meanings are as follows.
        -  The stem, consisting of a proprioception tokenizer and a vision tokenizer, maps the
          vision and proprioception observations of different embodiments to a fixed number (e.g. 16) of tokens.
        -  The shared trunk, which is a Transformer, maps the concatenated tokens into shared representations.
        -  The head then maps the processed tokens to actions in different downstream tasks.
        For a specific embodiment, one stem/head pair is activated.

    ┌─────────────────────────────────────────────────────┐
    |                         Outputs                     |
    |                            ▲                        |
    |     ┌───────────┐   ┌───────────┐   ┌───────────┐   |
    |     |   Head 1  |   |   Head 2  |   |   Head 3  |   |
    |     └───────▲───┘   └───────▲───┘   └─────▲─────┘   |
    |             │               │             │         |
    |            ┌─────────────────────────────────┐      |
    |            |          Trunk Transformer.     │      |
    |            └────────▲──────────▲─────────────┘      |
    |                     │          │                    |
    |           ┌─────────┴──────┬───┴────────────┐       |
    |           │                │                │       |
    |   ┌───────┴────┐   ┌─────┴─────┐   ┌────┴──────┐    |
    |   │  Stem.     │   │  Stem.    │   │  Stem.    │    |
    |   │  Encoder 1 │   │  Encoder 2│   │  Encoder 3│    |
    |   │  ┌─────────┐   │ ┌─────────┐   │ ┌─────────┐    |
    |   │  │image emb│   │ │image emb│   │ │image emb│    |
    |   │  │state emb│   │ │state emb│   │ │state emb│    |
    |   └──┴─────────┘   └─┴─────────┘   └─┴─────────┘    |
    |                                                     |
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, config: HPTConfig):
        super().__init__()
        self.config = config
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.embed_dim = config.embed_dim

        self.trunk = self._create_policy_trunk(config.embed_dim, config.num_blocks, config.num_heads)
        self.stems = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        self.encoders = nn.ModuleDict()
        self.domains = []
        self.n_obs_steps = config.n_obs_steps
        self.action_chunk_size = config.action_chunk_size
        self.token_postprocessing = config.token_postprocessing

        # initialize modules.
        if "image" in config.modalities and not self.config.freeze_encoders:
            self.init_encoders("image", ResNet(resnet_model=self.config.vision_backbone))
        self.init_domain_stem(self.config.domain_name)
        self.init_domain_head(self.config.domain_name)

        self.iosp_enabled = bool(getattr(self.config, "iosp_enabled", False))
        self.iosp_mode = str(getattr(self.config, "iosp_mode", "proxy") or "proxy").strip().lower()
        self.iosp_top_k = int(getattr(self.config, "iosp_top_k", 8) or 8)
        self.iosp_embed_dim = int(getattr(self.config, "iosp_embed_dim", 0) or 0)
        self.iosp_clip_model_name = str(getattr(self.config, "iosp_clip_model_name", "") or "").strip()
        self.iosp_clip_image_size = int(getattr(self.config, "iosp_clip_image_size", 224) or 224)
        self.iosp_clip_strategy = str(getattr(self.config, "iosp_clip_strategy", "direct") or "direct").strip().lower()
        self.iosp_clip_remaining_max = int(getattr(self.config, "iosp_clip_remaining_max", 0) or 0)
        self.iosp_proj = None
        self.iosp_type_embeddings = None
        self.iosp_type_embeddings_clip = None
        self.iosp_group_moes = None
        if self.iosp_enabled:
            if self.iosp_embed_dim <= 0:
                raise ValueError("iosp_embed_dim must be > 0 when iosp_enabled=True.")
            self.iosp_proj = nn.Linear(self.iosp_embed_dim, self.embed_dim, bias=False)
            self.iosp_type_embeddings = nn.Parameter(torch.zeros(3, self.embed_dim))
            nn.init.trunc_normal_(self.iosp_type_embeddings, std=INIT_CONST)
            if self.iosp_mode == "clip" and int(self.iosp_embed_dim) != 512:
                raise ValueError("iosp_mode=clip requires iosp_embed_dim=512 (CLIP projection dim).")
            if self.iosp_mode == "clip" and self.iosp_clip_strategy in ("stem_latent", "hpt_stem_latent"):
                self.iosp_type_embeddings_clip = nn.Parameter(torch.zeros(3, int(self.iosp_embed_dim)))
                nn.init.trunc_normal_(self.iosp_type_embeddings_clip, std=INIT_CONST)
            if self.iosp_mode == "clip" and bool(getattr(self.config, "iosp_group_moe_enabled", False)):
                self.iosp_group_moes = nn.ModuleList(
                    [
                        DGMoEAdapter(
                            int(self.iosp_embed_dim),
                            num_experts=int(getattr(self.config, "iosp_group_moe_num_experts", 4) or 4),
                            expert_hidden_dim=int(getattr(self.config, "iosp_group_moe_hidden_dim", 128) or 128),
                            lambda_threshold=float(getattr(self.config, "iosp_group_moe_lambda", 0.5) or 0.5),
                            temperature=float(getattr(self.config, "iosp_group_moe_temperature", 1.0) or 1.0),
                            renormalize_gates=bool(getattr(self.config, "iosp_group_moe_renormalize_gates", False)),
                            gate_residual=bool(getattr(self.config, "iosp_group_moe_gate_residual", True)),
                        )
                        for _ in range(3)
                    ]
                )

    def _init_weights(self, m: nn.Module):
        """
        Weight initialization for transformer
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_encoders(self, modality: str, encoder: nn.Module):
        """
        Add image/language encoders into the policy parameters in the case of joint finetuning
        """
        self.encoders[modality] = encoder
        self.encoders = nn.ModuleDict(self.encoders)

    def init_domain_stem(self, domain_name: str):
        """
        Initialize an observation stem for each domain
        """
        self.stem_spec = self.config
        self.modalities = self.config.modalities
        for modality in self.modalities:
            self.stems[domain_name + "_" + modality] = MLPStem(
                input_dim=getattr(self.config, modality + "_input_dim"),
                output_dim=getattr(self.config, modality + "_output_dim"),
                widths=getattr(self.config, modality + "_widths"),
                num_of_copy=getattr(self.config, modality + "_num_of_copy"),
            )
            self.stems[domain_name + "_" + modality].init_cross_attn(self.config, modality)

    def init_domain_head(self, domain_name: str):
        """initialize an action head for each domain, along with normalizer"""
        self.head_spec = self.config
        self.action_chunk_size = self.config.action_chunk_size
        self.domains.append(domain_name)
        if self.config.head_architecture == "diffusion":
            self.heads[domain_name] = DiffusionHead(
                config=self.config,
                embed_dim=self.config.embed_dim,
                action_chunk_size=self.config.action_chunk_size,
                action_dim=self.config.head_action_dim,
            )

        elif self.config.head_architecture == "regression":
            self.heads[domain_name] = RegressionHead(
                config=self.config,
                embed_dim=self.config.embed_dim,
                action_chunk_size=self.config.action_chunk_size,
                action_dim=self.config.head_action_dim,
            )

        elif self.config.head_architecture == "transformer":
            self.heads[domain_name] = TransformerHead(
                config=self.config,
                action_chunk_size=self.config.action_chunk_size,
            )


    def _create_policy_trunk(
        self,
        embed_dim: int = 1024,
        num_blocks: int = 24,
        num_heads: int = 16,
        drop_path: float = 0.0,
        weight_init_style: str = "pytorch",
    ):
        """create the shared representation for pretraining"""

        def instantiate_trunk(embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
                weight_init_style=weight_init_style,
                dgmoe_enabled=bool(self.config.dgmoe_enabled),
                dgmoe_num_experts=int(self.config.dgmoe_num_experts),
                dgmoe_expert_hidden_dim=int(self.config.dgmoe_expert_hidden_dim),
                dgmoe_impl=str(getattr(self.config, "dgmoe_impl", "adapter") or "adapter").strip().lower(),
                dgmoe_lambda=float(self.config.dgmoe_lambda),
                dgmoe_temperature=float(self.config.dgmoe_temperature),
                dgmoe_renormalize_gates=bool(self.config.dgmoe_renormalize_gates),
                dgmoe_gate_residual=bool(self.config.dgmoe_gate_residual),
                dgmoe_force_active=bool(getattr(self.config, "dgmoe_force_active", True)),
            )

        trunk = {}
        trunk["trunk"] = instantiate_trunk(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=drop_path,
        )
        self.trunk = nn.ModuleDict(trunk)

        if len(self.config.load_pretrained) > 0:
            self.load_trunk(self.config.load_pretrained)

        return self.trunk

    def _reset_parameters(self):
        self.apply(self._init_weights)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the HPT to generate actions at test time. Assume inputs have been normalized.

        `batch` should have the following structure:
        {
            "observation.state" (optional): (B, state_dim) batch of robot states.

            "observation.images": (B, n_cameras, C, H, W) batch of images.
                AND/OR
            "observation.environment_state": (B, env_dim) batch of environment states.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
        """
        domain = self.config.domain_name
        features = self.forward_features(batch)
        action = self.heads[domain](features)
        return action

    def generate_actions(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the HPT to generate actions at test time. Assume inputs have been normalized."""
        domain = self.config.domain_name
        features = self.forward_features(batch)
        action = self.heads[domain](features)
        start = self.config.n_obs_steps - 1
        action = action[:, start:]

        return action

    def preprocess_tokens(self, domain: str, features: List[Tensor]) -> Tensor:
        """
        Shared modality layers and add modality tokens. Add positional and time embeddings.
        """
        tokens = torch.cat(features, dim=-2)
        position_tokens = self.get_position_embedding(tokens)
        tokens = tokens + position_tokens
        return tokens

    def get_position_embedding(self, feature: Tensor) -> Tensor:
        """
        Add positional embedding to the features
        """
        pos_embedding = get_sinusoid_encoding_table(0, feature.shape[1], self.embed_dim)
        pos_embedding = pos_embedding.repeat((1, 1, 1)).to(feature.device)
        return pos_embedding

    def postprocess_tokens(self, trunk_tokens: Tensor) -> Tensor:
        """
        Postprocesses the trunk tokens to obtain the final features.

        Args:
            trunk_tokens (Tensor): The trunk tokens of shape (N, L, D), where N is the batch size,
                                        L is the sequence length, and D is the token dimension.

        Returns:
            Tensor: The postprocessed tokens of shape (N, D), where N is the batch size and D is the
                          final feature dimension.
        """
        if self.token_postprocessing == "mean":
            return trunk_tokens.mean(dim=1)
        elif self.token_postprocessing == "no-op":
            return trunk_tokens

    def mapped_modality_keys(self, modality: str, data: dict[str, Tensor]) -> str | Tensor:
        """Select the data for the given modality"""
        selected_keys = []
        selected_data = []

        for k in sorted(data):  # maintain order
            if modality in k and k in self.config.input_shapes:
                selected_keys.append(k)
                selected_data.append(data[k])

        if len(selected_keys) == 0:
            raise ValueError(f"{modality=} not found in data keys")

        if modality == "image":
            sample = selected_data[0]
            if sample.ndim == 5:
                # Per-view tensors are (B, T, C, H, W); stack them into an instance/view axis.
                data = torch.stack(selected_data, dim=2)  # (B, T, V, C, H, W)
            elif sample.ndim == 4:
                # Per-view tensors are (B, C, H, W); add a time axis (T=1) and view axis (V=len(keys)).
                data = torch.stack(selected_data, dim=1)  # (B, V, C, H, W)
                data = rearrange(data, "b v c h w -> b 1 v c h w")
            else:
                raise ValueError(
                    f"Unsupported image tensor rank {int(sample.ndim)} for {selected_keys[0]} "
                    "(expected 4D or 5D tensors)."
                )

        if modality in ("state", "language"):
            data = torch.cat(selected_data, dim=-1)
            if data.ndim < 4:
                data = rearrange(data, "b t d -> b t 1 d" if data.ndim == 3 else "b d -> b 1 1 d")
        return modality, data

    def stem_process(self, domain: str, data: dict):
        """
        Pass through the stem to a fixed number of tokens.
        Args:
            data: dictionary of tensors of different modalities
        """
        feats = []

        for policy_modality in self.modalities:
            stem = self.stems[domain + "_" + policy_modality]
            modality, modality_data = self.mapped_modality_keys(policy_modality, data)
            skip_image_encoder = False

            if (
                modality == "image"
                and self.iosp_enabled
                and self.iosp_mode == "clip"
                and "observation.iosp" in data
            ):
                if self.iosp_clip_strategy == "direct":
                    feats.append(self._iosp_clip_tokens(modality_data, data["observation.iosp"]))
                    continue
                if self.iosp_clip_strategy == "stem_latent":
                    clip_context = self._iosp_clip_context_tokens(modality_data, data["observation.iosp"])
                    modality_data = rearrange(clip_context, "b l d -> b 1 1 l d")
                    skip_image_encoder = True
                elif self.iosp_clip_strategy == "hpt_stem_latent":
                    hpt_context = self._iosp_clip_selected_hpt_tokens(
                        modality_data,
                        data["observation.iosp"],
                        iosp_perm=data.get("observation.iosp_perm", None),
                    )
                    modality_data = rearrange(hpt_context, "b l d -> b 1 1 l d")
                    skip_image_encoder = True
                else:
                    raise ValueError(f"Unsupported iosp_clip_strategy={self.iosp_clip_strategy!r}.")

            # add time horizon and instance number
            if "image" in modality and "image" in self.encoders and not bool(skip_image_encoder):
                modality_data = self.encoders["image"](modality_data)

            # positional embedding for observations
            data_shape = modality_data.shape
            data_horizon = data_shape[1]
            horizon = data_horizon

            if self.training and self.stem_spec.random_horizon_masking and data_horizon > 1:
                horizon = np.random.randint(1, data_horizon + 1)
                modality_data = modality_data[:, data_horizon - horizon : data_horizon]

            # data is N x T x M x ... x D where M is the # of instances for that sensor
            positional_embedding = get_sinusoid_encoding_table(
                0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]
            ).to(modality_data)
            positional_embedding = repeat(
                positional_embedding, "b h w -> (repeat b) h w", repeat=data_shape[0]
            )

            modality_data = modality_data + positional_embedding.view(modality_data.shape)
            stem_token = stem.compute_latent(modality_data)
            feats.append(stem_token)

        return feats

    def _get_iosp_clip_model(self, device: torch.device):
        model_name = str(self.iosp_clip_model_name or "openai/clip-vit-base-patch32").strip()
        key = (model_name, str(device))
        cached = _IOSP_CLIP_GLOBAL_CACHE.get(key, None)
        if cached is not None:
            return cached  # type: ignore[return-value]

        try:
            from transformers import CLIPModel  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on external env
            raise RuntimeError("transformers is required for iosp_mode=clip.") from exc
        model = CLIPModel.from_pretrained(model_name)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.to(device)
        _IOSP_CLIP_GLOBAL_CACHE[key] = model
        return model

    def _iosp_clip_tokens(self, modality_data: Tensor, iosp_flat: Tensor) -> Tensor:
        """Paper-aligned IOSP: compute CLIP ViT patch tokens, then select TO/SO/BO top-k patches."""
        if self.iosp_proj is None or self.iosp_type_embeddings is None:
            raise RuntimeError("IOSP clip mode requires iosp_proj and iosp_type_embeddings.")

        if modality_data.ndim == 6:
            # (B,T,V,C,H,W) -> use last observation step.
            images = modality_data[:, -1]  # (B,V,C,H,W)
        elif modality_data.ndim == 5:
            images = modality_data  # (B,V,C,H,W)
        else:
            raise ValueError(f"Unsupported image tensor rank {int(modality_data.ndim)} for iosp clip mode.")

        batch_size, views, channels, height, width = images.shape
        if int(channels) != 3:
            raise ValueError(f"Expected RGB images with C=3, got C={int(channels)}.")

        if iosp_flat.ndim == 1:
            iosp_flat = iosp_flat.view(1, -1)
        if iosp_flat.ndim == 3 and int(iosp_flat.shape[1]) == 1:
            iosp_flat = iosp_flat[:, 0, :]
        if iosp_flat.ndim != 2:
            raise ValueError(f"Expected observation.iosp shape (B, 3*E), got {tuple(iosp_flat.shape)}.")
        expected = 3 * int(self.iosp_embed_dim)
        if int(iosp_flat.shape[-1]) != expected:
            raise ValueError(
                "observation.iosp feature width mismatch: "
                f"got {int(iosp_flat.shape[-1])}, expected {expected}."
            )
        if int(iosp_flat.shape[0]) != int(batch_size):
            raise ValueError(
                "observation.iosp batch mismatch: "
                f"got B={int(iosp_flat.shape[0])}, expected {int(batch_size)}."
            )

        clip_model = self._get_iosp_clip_model(device=images.device)
        images_flat = images.reshape(int(batch_size) * int(views), int(channels), int(height), int(width))
        target = int(self.iosp_clip_image_size)
        if target > 0 and (int(height) != target or int(width) != target):
            images_flat = F.interpolate(images_flat, size=(target, target), mode="bilinear", align_corners=False)
        mean = torch.tensor(_CLIP_IMAGE_MEAN, device=images_flat.device, dtype=images_flat.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_IMAGE_STD, device=images_flat.device, dtype=images_flat.dtype).view(1, 3, 1, 1)
        pixel_values = (images_flat - mean) / std

        with torch.no_grad():
            vision_out = clip_model.vision_model(pixel_values=pixel_values)
            vision_tokens = vision_out.last_hidden_state  # (B*V, 1+P, Dvision)
            patch_tokens = clip_model.visual_projection(vision_tokens)  # (B*V, 1+P, 512)
        patch_tokens = patch_tokens[:, 1:, :].contiguous()  # drop CLS, keep patches
        patch_tokens = patch_tokens.reshape(
            int(batch_size),
            int(views) * int(patch_tokens.shape[1]),
            -1,
        )  # (B, L, 512)

        group_embeds = iosp_flat.view(int(batch_size), 3, int(self.iosp_embed_dim))  # (B,3,512)
        tokens_norm = F.normalize(patch_tokens, dim=-1)
        groups_norm = F.normalize(group_embeds, dim=-1)
        sim = torch.einsum("bld,bgd->blg", tokens_norm, groups_norm)  # (B,L,3)

        seq_len = int(patch_tokens.shape[1])
        k = min(int(self.iosp_top_k), max(seq_len // 3, 1))
        selected_mask = torch.zeros((int(batch_size), seq_len), device=patch_tokens.device, dtype=torch.bool)
        selected_chunks: list[Tensor] = []
        for group_idx in range(3):
            scores = sim[:, :, group_idx].clone()
            scores[selected_mask] = -1e9
            _, idx = torch.topk(scores, k=int(k), dim=1)
            selected_chunks.append(idx)
            selected_mask.scatter_(1, idx, True)

        # Gather CLIP tokens for group-level MoE refinement.
        group_tokens: list[Tensor] = []
        for group_idx, idx in enumerate(selected_chunks):
            toks = torch.gather(
                patch_tokens,
                1,
                idx.unsqueeze(-1).expand(int(batch_size), int(k), int(patch_tokens.shape[-1])),
            )
            if self.iosp_group_moes is not None:
                moe_out, _, _ = self.iosp_group_moes[group_idx](toks)
                toks = toks + moe_out
            group_tokens.append(toks)

        patch_tokens_embed = self.iosp_proj(patch_tokens)  # (B,L,embed_dim)
        group_tokens_embed = [self.iosp_proj(toks) for toks in group_tokens]  # 3x(B,k,embed_dim)

        idx_all = torch.arange(seq_len, device=patch_tokens.device).unsqueeze(0).expand(int(batch_size), seq_len)
        tmp = idx_all.clone()
        tmp[selected_mask] = tmp[selected_mask] + int(seq_len)
        remaining_sorted = torch.argsort(tmp, dim=1)
        remaining = remaining_sorted[:, : seq_len - (3 * int(k))]
        remaining_embed = torch.gather(
            patch_tokens_embed,
            1,
            remaining.unsqueeze(-1).expand(int(batch_size), int(remaining.shape[1]), int(self.embed_dim)),
        )

        out = torch.cat([group_tokens_embed[0], group_tokens_embed[1], group_tokens_embed[2], remaining_embed], dim=1)
        out[:, 0:k, :] = out[:, 0:k, :] + self.iosp_type_embeddings[0]
        out[:, k : 2 * k, :] = out[:, k : 2 * k, :] + self.iosp_type_embeddings[1]
        out[:, 2 * k : 3 * k, :] = out[:, 2 * k : 3 * k, :] + self.iosp_type_embeddings[2]
        return out

    def _iosp_clip_context_tokens(self, modality_data: Tensor, iosp_flat: Tensor) -> Tensor:
        """IOSP (CLIP) context tokens in CLIP space for stem_latent strategy.

        Returns a token sequence in CLIP projection space (dim=iosp_embed_dim, typically 512) that can be
        passed through the existing HPT image stem to produce fixed latent tokens.
        """
        if self.iosp_mode != "clip":
            raise RuntimeError("_iosp_clip_context_tokens called but iosp_mode != 'clip'.")

        if modality_data.ndim == 6:
            images = modality_data[:, -1]  # (B,V,C,H,W)
        elif modality_data.ndim == 5:
            images = modality_data
        else:
            raise ValueError(f"Unsupported image tensor rank {int(modality_data.ndim)} for iosp clip mode.")

        batch_size, views, channels, height, width = images.shape
        if int(channels) != 3:
            raise ValueError(f"Expected RGB images with C=3, got C={int(channels)}.")

        if iosp_flat.ndim == 1:
            iosp_flat = iosp_flat.view(1, -1)
        if iosp_flat.ndim == 3 and int(iosp_flat.shape[1]) == 1:
            iosp_flat = iosp_flat[:, 0, :]
        if iosp_flat.ndim != 2:
            raise ValueError(f"Expected observation.iosp shape (B, 3*E), got {tuple(iosp_flat.shape)}.")
        expected = 3 * int(self.iosp_embed_dim)
        if int(iosp_flat.shape[-1]) != expected:
            raise ValueError(
                "observation.iosp feature width mismatch: "
                f"got {int(iosp_flat.shape[-1])}, expected {expected}."
            )
        if int(iosp_flat.shape[0]) != int(batch_size):
            raise ValueError(
                "observation.iosp batch mismatch: "
                f"got B={int(iosp_flat.shape[0])}, expected {int(batch_size)}."
            )

        clip_model = self._get_iosp_clip_model(device=images.device)
        images_flat = images.reshape(int(batch_size) * int(views), int(channels), int(height), int(width))
        target = int(self.iosp_clip_image_size)
        if target > 0 and (int(height) != target or int(width) != target):
            images_flat = F.interpolate(images_flat, size=(target, target), mode="bilinear", align_corners=False)
        mean = torch.tensor(_CLIP_IMAGE_MEAN, device=images_flat.device, dtype=images_flat.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_IMAGE_STD, device=images_flat.device, dtype=images_flat.dtype).view(1, 3, 1, 1)
        pixel_values = (images_flat - mean) / std

        with torch.no_grad():
            vision_out = clip_model.vision_model(pixel_values=pixel_values)
            vision_tokens = vision_out.last_hidden_state  # (B*V, 1+P, Dvision)
            patch_tokens = clip_model.visual_projection(vision_tokens)  # (B*V, 1+P, 512)
        patch_tokens = patch_tokens[:, 1:, :].contiguous()  # drop CLS, keep patches
        patch_tokens = patch_tokens.reshape(
            int(batch_size),
            int(views) * int(patch_tokens.shape[1]),
            -1,
        )  # (B, L, 512)

        group_embeds = iosp_flat.view(int(batch_size), 3, int(self.iosp_embed_dim))  # (B,3,512)
        tokens_norm = F.normalize(patch_tokens, dim=-1)
        groups_norm = F.normalize(group_embeds, dim=-1)
        sim = torch.einsum("bld,bgd->blg", tokens_norm, groups_norm)  # (B,L,3)

        seq_len = int(patch_tokens.shape[1])
        k = min(int(self.iosp_top_k), max(seq_len // 3, 1))
        selected_mask = torch.zeros((int(batch_size), seq_len), device=patch_tokens.device, dtype=torch.bool)
        selected_chunks: list[Tensor] = []
        for group_idx in range(3):
            scores = sim[:, :, group_idx].clone()
            scores[selected_mask] = -1e9
            _, idx = torch.topk(scores, k=int(k), dim=1)
            selected_chunks.append(idx)
            selected_mask.scatter_(1, idx, True)

        group_tokens: list[Tensor] = []
        for group_idx, idx in enumerate(selected_chunks):
            toks = torch.gather(
                patch_tokens,
                1,
                idx.unsqueeze(-1).expand(int(batch_size), int(k), int(patch_tokens.shape[-1])),
            )
            if self.iosp_group_moes is not None:
                moe_out, _, _ = self.iosp_group_moes[group_idx](toks)
                toks = toks + moe_out
            group_tokens.append(toks)

        idx_all = torch.arange(seq_len, device=patch_tokens.device).unsqueeze(0).expand(int(batch_size), seq_len)
        tmp = idx_all.clone()
        tmp[selected_mask] = tmp[selected_mask] + int(seq_len)
        remaining_sorted = torch.argsort(tmp, dim=1)
        remaining = remaining_sorted[:, : seq_len - (3 * int(k))]
        if int(self.iosp_clip_remaining_max) > 0:
            remaining = remaining[:, : int(self.iosp_clip_remaining_max)]
        remaining_tokens = torch.gather(
            patch_tokens,
            1,
            remaining.unsqueeze(-1).expand(int(batch_size), int(remaining.shape[1]), int(patch_tokens.shape[-1])),
        )

        out = torch.cat([group_tokens[0], group_tokens[1], group_tokens[2], remaining_tokens], dim=1)
        if self.iosp_type_embeddings_clip is not None:
            out[:, 0:k, :] = out[:, 0:k, :] + self.iosp_type_embeddings_clip[0]
            out[:, k : 2 * k, :] = out[:, k : 2 * k, :] + self.iosp_type_embeddings_clip[1]
            out[:, 2 * k : 3 * k, :] = out[:, 2 * k : 3 * k, :] + self.iosp_type_embeddings_clip[2]
        return out

    def _iosp_clip_selected_hpt_tokens(
        self,
        modality_data: Tensor,
        iosp_flat: Tensor,
        *,
        iosp_perm: Tensor | None = None,
    ) -> Tensor:
        """Hybrid IOSP-CLIP: use CLIP patch similarity to select indices, but return HPT image-encoder tokens.

        This keeps CLIP-space patch selection while feeding the downstream HPT stem/trunk with the existing
        HPT image-encoder (ResNet) spatial token distribution.
        """
        if self.iosp_mode != "clip":
            raise RuntimeError("_iosp_clip_selected_hpt_tokens called but iosp_mode != 'clip'.")
        if "image" not in self.encoders:
            raise RuntimeError("iosp_clip_strategy=hpt_stem_latent requires the HPT image encoder to be initialized.")

        if modality_data.ndim == 6:
            images = modality_data[:, -1]  # (B,V,C,H,W)
        elif modality_data.ndim == 5:
            images = modality_data
        else:
            raise ValueError(f"Unsupported image tensor rank {int(modality_data.ndim)} for iosp clip mode.")

        batch_size, views, channels, height, width = images.shape
        if int(channels) != 3:
            raise ValueError(f"Expected RGB images with C=3, got C={int(channels)}.")

        # Fast path: reuse a precomputed permutation over spatial tokens, if provided by the dataset.
        if iosp_perm is not None:
            if iosp_perm.ndim == 1:
                iosp_perm = iosp_perm.view(1, -1)
            if iosp_perm.ndim != 2:
                raise ValueError(f"Expected observation.iosp_perm shape (B, L), got {tuple(iosp_perm.shape)}.")
            if int(iosp_perm.shape[0]) != int(batch_size):
                raise ValueError(
                    "observation.iosp_perm batch mismatch: "
                    f"got B={int(iosp_perm.shape[0])}, expected {int(batch_size)}."
                )

            encoder_in = images.unsqueeze(1)  # (B,1,V,3,H,W)
            hpt_tokens = self.encoders["image"](encoder_in)
            if hpt_tokens.ndim != 3:
                raise RuntimeError(f"Unexpected HPT image encoder token shape {tuple(hpt_tokens.shape)}.")
            seq_len = int(hpt_tokens.shape[1])
            perm_len = int(iosp_perm.shape[1])
            if perm_len <= 0:
                raise ValueError("observation.iosp_perm must have non-empty length.")
            if int(iosp_perm.max().item()) >= seq_len or int(iosp_perm.min().item()) < 0:
                raise ValueError(
                    "observation.iosp_perm contains out-of-range indices: "
                    f"min={int(iosp_perm.min().item())}, max={int(iosp_perm.max().item())}, seq_len={seq_len}."
                )

            out = torch.gather(
                hpt_tokens,
                1,
                iosp_perm.unsqueeze(-1).expand(int(batch_size), perm_len, int(hpt_tokens.shape[-1])),
            )
            k = min(int(self.iosp_top_k), max(seq_len // 3, 1))
            if self.iosp_type_embeddings_clip is not None and perm_len >= 3 * int(k):
                out[:, 0:k, :] = out[:, 0:k, :] + self.iosp_type_embeddings_clip[0]
                out[:, k : 2 * k, :] = out[:, k : 2 * k, :] + self.iosp_type_embeddings_clip[1]
                out[:, 2 * k : 3 * k, :] = out[:, 2 * k : 3 * k, :] + self.iosp_type_embeddings_clip[2]
            return out

        if iosp_flat.ndim == 1:
            iosp_flat = iosp_flat.view(1, -1)
        if iosp_flat.ndim == 3 and int(iosp_flat.shape[1]) == 1:
            iosp_flat = iosp_flat[:, 0, :]
        if iosp_flat.ndim != 2:
            raise ValueError(f"Expected observation.iosp shape (B, 3*E), got {tuple(iosp_flat.shape)}.")
        expected = 3 * int(self.iosp_embed_dim)
        if int(iosp_flat.shape[-1]) != expected:
            raise ValueError(
                "observation.iosp feature width mismatch: "
                f"got {int(iosp_flat.shape[-1])}, expected {expected}."
            )
        if int(iosp_flat.shape[0]) != int(batch_size):
            raise ValueError(
                "observation.iosp batch mismatch: "
                f"got B={int(iosp_flat.shape[0])}, expected {int(batch_size)}."
            )

        # 1) CLIP tokens for selection indices.
        clip_model = self._get_iosp_clip_model(device=images.device)
        images_flat = images.reshape(int(batch_size) * int(views), int(channels), int(height), int(width))
        target = int(self.iosp_clip_image_size)
        if target > 0 and (int(height) != target or int(width) != target):
            images_flat_clip = F.interpolate(images_flat, size=(target, target), mode="bilinear", align_corners=False)
        else:
            images_flat_clip = images_flat
        mean = torch.tensor(_CLIP_IMAGE_MEAN, device=images_flat_clip.device, dtype=images_flat_clip.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_IMAGE_STD, device=images_flat_clip.device, dtype=images_flat_clip.dtype).view(1, 3, 1, 1)
        pixel_values = (images_flat_clip - mean) / std

        with torch.no_grad():
            vision_out = clip_model.vision_model(pixel_values=pixel_values)
            vision_tokens = vision_out.last_hidden_state  # (B*V, 1+P, Dvision)
            clip_patch_tokens = clip_model.visual_projection(vision_tokens)  # (B*V, 1+P, 512)
        clip_patch_tokens = clip_patch_tokens[:, 1:, :].contiguous()  # drop CLS, keep patches
        clip_patch_tokens = clip_patch_tokens.reshape(
            int(batch_size),
            int(views) * int(clip_patch_tokens.shape[1]),
            -1,
        )  # (B, L, 512)

        group_embeds = iosp_flat.view(int(batch_size), 3, int(self.iosp_embed_dim))  # (B,3,512)
        tokens_norm = F.normalize(clip_patch_tokens, dim=-1)
        groups_norm = F.normalize(group_embeds, dim=-1)
        sim = torch.einsum("bld,bgd->blg", tokens_norm, groups_norm)  # (B,L,3)

        seq_len = int(clip_patch_tokens.shape[1])
        k = min(int(self.iosp_top_k), max(seq_len // 3, 1))
        selected_mask = torch.zeros((int(batch_size), seq_len), device=clip_patch_tokens.device, dtype=torch.bool)
        selected_chunks: list[Tensor] = []
        for group_idx in range(3):
            scores = sim[:, :, group_idx].clone()
            scores[selected_mask] = -1e9
            _, idx = torch.topk(scores, k=int(k), dim=1)
            selected_chunks.append(idx)
            selected_mask.scatter_(1, idx, True)

        idx_all = torch.arange(seq_len, device=clip_patch_tokens.device).unsqueeze(0).expand(int(batch_size), seq_len)
        tmp = idx_all.clone()
        tmp[selected_mask] = tmp[selected_mask] + int(seq_len)
        remaining_sorted = torch.argsort(tmp, dim=1)
        remaining = remaining_sorted[:, : seq_len - (3 * int(k))]
        if int(self.iosp_clip_remaining_max) > 0:
            remaining = remaining[:, : int(self.iosp_clip_remaining_max)]

        # 2) HPT image encoder tokens (ResNet 7x7 grid per view).
        encoder_in = images.unsqueeze(1)  # (B,1,V,3,H,W)
        hpt_tokens = self.encoders["image"](encoder_in)  # (B, 49*V, 512) when T=1
        if hpt_tokens.ndim != 3:
            raise RuntimeError(f"Unexpected HPT image encoder token shape {tuple(hpt_tokens.shape)}.")
        if int(hpt_tokens.shape[1]) != int(seq_len):
            raise RuntimeError(
                "HPT image token length mismatch vs CLIP patch length: "
                f"got {int(hpt_tokens.shape[1])}, expected {int(seq_len)}."
            )

        group_tokens: list[Tensor] = []
        for group_idx, idx in enumerate(selected_chunks):
            toks = torch.gather(
                hpt_tokens,
                1,
                idx.unsqueeze(-1).expand(int(batch_size), int(k), int(hpt_tokens.shape[-1])),
            )
            if self.iosp_group_moes is not None:
                moe_out, _, _ = self.iosp_group_moes[group_idx](toks)
                toks = toks + moe_out
            group_tokens.append(toks)

        remaining_tokens = torch.gather(
            hpt_tokens,
            1,
            remaining.unsqueeze(-1).expand(int(batch_size), int(remaining.shape[1]), int(hpt_tokens.shape[-1])),
        )
        out = torch.cat([group_tokens[0], group_tokens[1], group_tokens[2], remaining_tokens], dim=1)
        if self.iosp_type_embeddings_clip is not None:
            out[:, 0:k, :] = out[:, 0:k, :] + self.iosp_type_embeddings_clip[0]
            out[:, k : 2 * k, :] = out[:, k : 2 * k, :] + self.iosp_type_embeddings_clip[1]
            out[:, 2 * k : 3 * k, :] = out[:, 2 * k : 3 * k, :] + self.iosp_type_embeddings_clip[2]
        return out

    def _apply_iosp(self, image_tokens: Tensor, iosp_flat: Tensor) -> Tensor:
        """IOSP: reorder image tokens based on TO/SO/BO text embeddings.

        Expected `iosp_flat` shape: (B, 3 * iosp_embed_dim).
        """
        if not self.iosp_enabled or self.iosp_proj is None:
            return image_tokens

        if image_tokens.ndim != 3:
            raise ValueError(f"Expected image_tokens shape (B, L, D), got {tuple(image_tokens.shape)}.")
        batch_size, seq_len, dim = image_tokens.shape
        if dim != int(self.embed_dim):
            raise ValueError(f"Expected image_tokens dim={int(self.embed_dim)}, got {int(dim)}.")

        if iosp_flat.ndim == 1:
            iosp_flat = iosp_flat.view(1, -1)
        if iosp_flat.ndim == 3 and iosp_flat.shape[1] == 1:
            iosp_flat = iosp_flat[:, 0, :]
        if iosp_flat.ndim != 2:
            raise ValueError(f"Expected observation.iosp shape (B, 3*E), got {tuple(iosp_flat.shape)}.")

        expected = 3 * int(self.iosp_embed_dim)
        if int(iosp_flat.shape[-1]) != expected:
            raise ValueError(
                "observation.iosp feature width mismatch: "
                f"got {int(iosp_flat.shape[-1])}, expected {expected}."
            )
        if int(iosp_flat.shape[0]) != int(batch_size):
            raise ValueError(
                "observation.iosp batch mismatch: "
                f"got B={int(iosp_flat.shape[0])}, expected {int(batch_size)}."
            )

        group_embeds = iosp_flat.view(batch_size, 3, int(self.iosp_embed_dim))
        group_embeds = self.iosp_proj(group_embeds)

        tokens_norm = F.normalize(image_tokens, dim=-1)
        groups_norm = F.normalize(group_embeds, dim=-1)
        sim = torch.einsum("bld,bgd->blg", tokens_norm, groups_norm)  # (B,L,3)

        k = min(int(self.iosp_top_k), max(int(seq_len) // 3, 1))
        selected_mask = torch.zeros((batch_size, seq_len), device=image_tokens.device, dtype=torch.bool)
        selected_chunks: list[Tensor] = []
        for group_idx in range(3):
            scores = sim[:, :, group_idx].clone()
            scores[selected_mask] = -1e9
            _, idx = torch.topk(scores, k=k, dim=1)
            selected_chunks.append(idx)
            selected_mask.scatter_(1, idx, True)
        selected = torch.cat(selected_chunks, dim=1)  # (B,3k)

        idx_all = torch.arange(seq_len, device=image_tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        tmp = idx_all.clone()
        tmp[selected_mask] = tmp[selected_mask] + int(seq_len)
        remaining_sorted = torch.argsort(tmp, dim=1)
        remaining = remaining_sorted[:, : int(seq_len) - int(selected.shape[1])]
        perm = torch.cat([selected, remaining], dim=1)  # (B,L)

        out = torch.gather(image_tokens, 1, perm.unsqueeze(-1).expand(batch_size, seq_len, dim))
        if self.iosp_type_embeddings is not None:
            out[:, 0:k, :] = out[:, 0:k, :] + self.iosp_type_embeddings[0]
            out[:, k : 2 * k, :] = out[:, k : 2 * k, :] + self.iosp_type_embeddings[1]
            out[:, 2 * k : 3 * k, :] = out[:, 2 * k : 3 * k, :] + self.iosp_type_embeddings[2]
        return out

    def forward_features(self, data: dict) -> Tensor:
        """
        Compute the features for the given domain and data.
        Args:
            domain (str): The domain of the data.
            data (Tensor): The input data.
        """
        domain = self.config.domain_name

        # stem pass
        self.stem_tokens = self.stem_process(domain, data)
        if (
            self.iosp_enabled
            and self.iosp_mode != "clip"
            and "observation.iosp" in data
            and "image" in self.modalities
        ):
            try:
                image_index = list(self.modalities).index("image")
            except ValueError:
                image_index = -1
            if image_index >= 0 and image_index < len(self.stem_tokens):
                self.stem_tokens[image_index] = self._apply_iosp(
                    self.stem_tokens[image_index], data["observation.iosp"]
                )

        # combine tokens
        self.trunk_tokens = self.preprocess_tokens(domain, self.stem_tokens)

        # trunk pass
        self.trunk_tokens = self.trunk["trunk"](self.trunk_tokens)

        # pooling the features
        return self.postprocess_tokens(self.trunk_tokens)

    def load_trunk(self, path: str):
        """load the trunk part of the model"""

        def module_mean_param(module):
            def maybe_mean(x):
                return float(torch.abs(x).mean()) if x is not None else 0

            max_data = np.mean([(maybe_mean(param.data)) for name, param in module.named_parameters()])
            return max_data

        local_override = os.environ.get("HPT_LOCAL_DIR", "").strip()
        if local_override:
            local_dir = Path(local_override)
            trunk_path = local_dir / "trunk.pth"
            if trunk_path.exists():
                print(f"Loading trunk from local dir: {trunk_path}")
                payload = torch.load(trunk_path, map_location="cpu")
                keys = self.trunk.load_state_dict(payload, strict=False)
                if getattr(keys, "unexpected_keys", None):
                    raise RuntimeError(f"Unexpected keys when loading trunk: {keys.unexpected_keys}")
                return

        candidate_dir = Path(path)
        if candidate_dir.exists() and candidate_dir.is_dir():
            trunk_path = candidate_dir / "trunk.pth"
            if trunk_path.exists():
                print(f"Loading trunk from local dir: {trunk_path}")
                payload = torch.load(trunk_path, map_location="cpu")
                keys = self.trunk.load_state_dict(payload, strict=False)
                if getattr(keys, "unexpected_keys", None):
                    raise RuntimeError(f"Unexpected keys when loading trunk: {keys.unexpected_keys}")
                return

        repo_id = "liruiw/hpt-" + path
        print(f"Loading trunk from {repo_id}")
        import huggingface_hub

        download_path = huggingface_hub.snapshot_download(repo_id)
        payload = torch.load(str(Path(download_path) / "trunk.pth"), map_location="cpu")
        keys = self.trunk.load_state_dict(payload, strict=False)
        if getattr(keys, "unexpected_keys", None):
            raise RuntimeError(f"Unexpected keys when loading trunk: {keys.unexpected_keys}")

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute the loss for the training loop forward pass."""
        domain = self.config.domain_name
        features = self.forward_features(batch)
        loss = self.heads[domain].compute_loss(features, batch)
        return loss



class RegressionHead(nn.Module):
    """Simple action regression head.

    This head is used in Phase B to keep HPT backbone experiments self-contained,
    without introducing diffusion/ACT dependencies.
    """

    def __init__(
        self,
        config: HPTConfig,
        embed_dim: int,
        action_chunk_size: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_chunk_size = int(action_chunk_size)
        self.action_dim = int(action_dim)
        hidden_sizes = tuple(getattr(config, "head_widths", ()) or ())
        layers: list[nn.Module] = []
        in_dim = int(embed_dim)
        if hidden_sizes:
            for h in hidden_sizes:
                layers.append(nn.Linear(in_dim, int(h)))
                layers.append(nn.ReLU())
                in_dim = int(h)
        layers.append(nn.Linear(in_dim, self.action_chunk_size * self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, context: Tensor) -> Tensor:
        if context.ndim != 2:
            context = context.view(int(context.shape[0]), -1)
        out = self.net(context)
        out = out.view(int(out.shape[0]), self.action_chunk_size, self.action_dim)
        if bool(getattr(self.config, "head_tanh_end", False)):
            out = torch.tanh(out)
        return out

    def compute_loss(self, global_cond: Tensor, data: dict[str, Tensor]) -> Tensor:
        pred = self(global_cond)
        target = data["action"]
        if target.ndim == 2 and int(target.shape[1]) == int(self.action_dim):
            target = target.unsqueeze(1)
        if target.ndim == 3 and int(target.shape[1]) == 1 and int(self.action_chunk_size) > 1:
            target = target.expand(-1, int(self.action_chunk_size), -1)
        if target.ndim == 2:
            target = target.reshape(int(target.shape[0]), int(self.action_chunk_size), int(self.action_dim))
        return LOSS(pred, target)

class DiffusionHead(nn.Module):
    """Diffusion based policy head based on the diffusion implementation"""

    def __init__(self, config, embed_dim: int, action_chunk_size: int, action_dim: int) -> None:
        super().__init__()
        from ..diffusion.modeling_diffusion import DiffusionConditionalUnet1d, _make_noise_scheduler

        self.model = DiffusionConditionalUnet1d(config=config, global_cond_dim=embed_dim)
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def conditional_sample(
        self,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """
        Perform conditional sampling using the diffusion process.
        """
        model = self.model
        scheduler = self.noise_scheduler
        device = get_device_from_parameters(model)
        trajectory = torch.randn(
            size=(len(global_cond), self.action_chunk_size, self.action_dim),
            dtype=global_cond.dtype,
            device=device,
            generator=generator,
        )

        global_cond = global_cond.to(device)
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = model(
                trajectory,
                torch.full(trajectory.shape[:1], t, dtype=torch.long, device=trajectory.device),
                global_cond=global_cond,
            )
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator).prev_sample
        return trajectory.transpose(0,1)

    def forward(self, global_cond: Tensor):
        return self.conditional_sample(global_cond)

    def compute_loss(self, global_cond: Tensor, data: Tensor) -> Tensor:
        trajectory = data["action"].reshape(
            (len(global_cond), self.action_chunk_size, self.action_dim)
        )  # Reshape the action tensor
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (len(trajectory),), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)
        target = noise
        return F.mse_loss(pred, target)


class TransformerHead(nn.Module):
    def __init__(
        self,
        config,
        action_chunk_size: int = 4,
    ) -> None:
        """
        Transformer decoder similar to ACT head.
        """
        super().__init__()
        from ..act.modeling_act import ACTDecoder

        self.config = config
        self.action_chunk_size = action_chunk_size
        self.decoder = ACTDecoder(config)
        self.tokens = nn.Parameter(torch.randn(action_chunk_size, self.config.embed_dim) * INIT_CONST)
        self.head_mlp = nn.Linear(self.config.embed_dim, self.config.head_action_dim)

    def forward(self, context: Tensor) -> Tensor:
        """
        context: (B, input_dim)
        """
        decoder_in = torch.zeros(
            (self.tokens.shape[0], len(context), self.config.embed_dim),
            dtype=context.dtype,
            device=context.device,
        )
        if len(context.shape) == 2:
            context = context.unsqueeze(1)

        decoder_out = self.decoder(
            decoder_in,  # [100, 8, 512]
            context.transpose(0, 1),  #  [1, 8, 512]
            decoder_pos_embed=self.tokens.unsqueeze(1),  #  [100, 1, 512]
        )
        out = self.head_mlp(decoder_out).transpose(0, 1).contiguous()
        return out

    def compute_loss(self, x: Tensor, target: dict) -> Tensor:
        target_action = target["action"]
        pred_action = self(x).view(target_action.shape)
        return F.l1_loss(pred_action, target_action)


class PolicyStem(nn.Module):
    """policy stem"""

    def __init__(self, **kwargs):
        super().__init__()

    def init_cross_attn(self, stem_spec, modality: str):
        """initialize cross attention module and the learnable tokens"""
        token_num = getattr(stem_spec, modality + "_crossattn_latent")
        self.tokens = nn.Parameter(torch.randn(1, token_num, stem_spec.embed_dim) * INIT_CONST)

        self.cross_attention = CrossAttention(
            stem_spec.embed_dim,
            heads=stem_spec.crossattn_heads,
            dim_head=stem_spec.crossattn_dim_head,
            dropout=stem_spec.crossattn_modality_dropout,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_latent(self, x: Tensor) -> Tensor:
        """
        Computes the latent representations of input data by attention.

        Args:
            Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.

        Returns:
            Output tensor with latent tokens, shape [32, 16, 128], where 16 is the number
            of tokens and 128 is the dimensionality of each token.

        Examples for vision features from ResNet:
        >>> x = np.random.randn(32, 3, 1, 49, 512)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)

        Examples for proprioceptive features:
        >>> x = np.random.randn(32, 3, 1, 7)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)
        """
        # Initial reshape to adapt to token dimensions
        stem_feat = self(x)  # (32, 3, 1, 49, 128)
        stem_feat = stem_feat.reshape(stem_feat.shape[0], -1, stem_feat.shape[-1])  # (32, 147, 128)

        # Replicating tokens for each item in the batch and computing cross-attention
        stem_tokens = self.tokens.repeat(len(stem_feat), 1, 1)  # (32, 16, 128)
        stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (32, 16, 128)
        return stem_tokens


class MLPStem(PolicyStem):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: Tuple[int] = (512, 512),
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
    ) -> None:
        """MLP Stem class"""
        super().__init__()
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList([nn.Sequential(*modules) for _ in range(num_of_copy)])

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W]
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        if self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            y = torch.stack(out, dim=1)
        else:
            y = self.net(x)
        return y


class CrossAttention(nn.Module):
    """
    CrossAttention module used in the Perceiver IO model.

    Args:
        query_dim (int): The dimension of the query input.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the CrossAttention module.

        Args:
            x (Tensor): The query input tensor.
            context (Tensor): The context input tensor.
            mask (Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.
        """
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> (b h) n d", h=h)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=h)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=h)

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            # fill in the masks with negative values
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _DGMoEExpert(nn.Module):
    """Lightweight expert used by the FedVLA DGMoE adapter.

    This intentionally uses a bottleneck (adapter-style) MLP to keep the
    parameter growth manageable when inserted into each trunk block.
    """

    def __init__(self, dim: int, hidden_dim: int, *, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(dim), int(hidden_dim))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(hidden_dim), int(dim))
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DGMoEAdapter(nn.Module):
    """Dual-Gating MoE layer for FedVLA-style trunk personalization.

    Two insertion styles are supported:
    - adapter: extra residual branch after the pretrained FFN path.
    - paper: the DGMoE branch itself serves as the FFN path.

    Inputs are accepted as (..., D) tensors (e.g., L×B×D in the HPT trunk).
    """

    def __init__(
        self,
        dim: int,
        *,
        num_experts: int,
        expert_hidden_dim: int,
        implementation: str = "adapter",
        lambda_threshold: float = 0.5,
        temperature: float = 1.0,
        renormalize_gates: bool = False,
        gate_residual: bool = True,
        force_active: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.implementation = str(implementation or "adapter").strip().lower()
        self.lambda_threshold = float(lambda_threshold)
        self.temperature = float(temperature)
        self.renormalize_gates = bool(renormalize_gates)
        self.force_active = bool(force_active)

        self.token_gate = nn.Linear(self.dim, self.num_experts, bias=True)
        self.gate_residual = nn.Linear(self.num_experts, self.num_experts, bias=False) if gate_residual else None
        expert_gate_init = torch.ones(self.num_experts) if self.implementation == "paper" else torch.zeros(self.num_experts)
        self.expert_gate = nn.Parameter(expert_gate_init)
        self.experts = nn.ModuleList(
            [_DGMoEExpert(self.dim, int(expert_hidden_dim), drop=0.0) for _ in range(self.num_experts)]
        )

    def forward(self, x: Tensor, *, prev_gate_logits: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"DGMoEAdapter expected last dim={self.dim}, got {int(x.shape[-1])}.")

        x_flat = x.reshape(-1, self.dim)
        logits = self.token_gate(x_flat)
        if prev_gate_logits is not None and self.gate_residual is not None:
            if prev_gate_logits.shape != logits.shape:
                raise ValueError(
                    "prev_gate_logits shape mismatch: "
                    f"prev={tuple(prev_gate_logits.shape)} curr={tuple(logits.shape)}."
                )
            logits = logits + self.gate_residual(prev_gate_logits)

        probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)

        if self.implementation == "paper":
            thresholds = self.lambda_threshold * torch.relu(self.expert_gate).unsqueeze(0)
        else:
            thresholds = self.lambda_threshold * torch.sigmoid(self.expert_gate).unsqueeze(0)
        mask = probs > thresholds
        if self.force_active:
            inactive = mask.sum(dim=-1, keepdim=True) == 0
            top_idx = torch.argmax(probs, dim=-1, keepdim=True)
            if bool(inactive.any()):
                mask = mask.scatter(-1, top_idx, True)

        gated = probs * mask.to(dtype=probs.dtype)
        if self.renormalize_gates:
            gated = gated / gated.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        out_flat = torch.zeros_like(x_flat)
        for expert_index, expert in enumerate(self.experts):
            out_flat = out_flat + gated[:, expert_index : expert_index + 1] * expert(x_flat)

        expert_counts = mask.to(dtype=torch.float32).sum(dim=0)  # (K,)
        return out_flat.reshape_as(x), logits.detach(), expert_counts.detach()


class BlockWithMasking(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        ffn_dropout_rate: float = 0.0,
        *,
        dgmoe_enabled: bool = False,
        dgmoe_impl: str = "adapter",
        dgmoe_num_experts: int = 0,
        dgmoe_expert_hidden_dim: int = 128,
        dgmoe_lambda: float = 0.5,
        dgmoe_temperature: float = 1.0,
        dgmoe_renormalize_gates: bool = False,
        dgmoe_gate_residual: bool = True,
        dgmoe_force_active: bool = True,
    ):
        super().__init__()
        self.attn = attn_target()
        self.norm_1 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
        self.norm_2 = norm_layer(dim)
        self.dgmoe = None
        self.dgmoe_impl = str(dgmoe_impl or "adapter").strip().lower()
        self.norm_3 = None if self.dgmoe_impl == "paper" else norm_layer(dim) if bool(dgmoe_enabled) else None
        if bool(dgmoe_enabled):
            if int(dgmoe_num_experts) <= 0:
                raise ValueError("dgmoe_num_experts must be > 0 when dgmoe_enabled=True.")
            self.dgmoe = DGMoEAdapter(
                int(dim),
                implementation=self.dgmoe_impl,
                num_experts=int(dgmoe_num_experts),
                expert_hidden_dim=int(dgmoe_expert_hidden_dim),
                lambda_threshold=float(dgmoe_lambda),
                temperature=float(dgmoe_temperature),
                renormalize_gates=bool(dgmoe_renormalize_gates),
                gate_residual=bool(dgmoe_gate_residual),
                force_active=bool(dgmoe_force_active),
            )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        *,
        prev_gate_logits: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        x = x + self.attn(self.norm_1(x), attn_mask)
        if self.dgmoe is None:
            x = x + self.mlp(self.norm_2(x))
            return x, None, None
        if self.dgmoe_impl == "paper":
            moe_out, gate_logits, expert_counts = self.dgmoe(self.norm_2(x), prev_gate_logits=prev_gate_logits)
            x = x + moe_out
            return x, gate_logits, expert_counts
        x = x + self.mlp(self.norm_2(x))
        if self.norm_3 is None:
            return x, None, None
        moe_out, gate_logits, expert_counts = self.dgmoe(self.norm_3(x), prev_gate_logits=prev_gate_logits)
        x = x + moe_out
        return x, gate_logits, expert_counts


class MultiheadAttention(nn.MultiheadAttention):
    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        return super().forward(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        attn_target: Callable,
        embed_dim: int,
        num_blocks: int,
        block: Callable = BlockWithMasking,
        pre_transformer_layer: Optional[Callable] = None,
        post_transformer_layer: Optional[Callable] = None,
        norm_layer: Callable = _LAYER_NORM,
        mlp_ratio: int = 4,
        ffn_dropout_rate: float = 0.0,
        weight_init_style: str = "pytorch",
        *,
        dgmoe_enabled: bool = False,
        dgmoe_impl: str = "adapter",
        dgmoe_num_experts: int = 0,
        dgmoe_expert_hidden_dim: int = 128,
        dgmoe_lambda: float = 0.5,
        dgmoe_temperature: float = 1.0,
        dgmoe_renormalize_gates: bool = False,
        dgmoe_gate_residual: bool = True,
        dgmoe_force_active: bool = True,
    ):
        """
        Simple Transformer with the following features
        1. Supports masked attention
        2. Supports DropPath
        3. Supports LayerScale
        4. Supports Dropout in Attention and FFN
        5. Makes few assumptions about the input except that it is a Tensor
        """
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer
        self.dgmoe_enabled = bool(dgmoe_enabled)
        self.dgmoe_num_experts = int(dgmoe_num_experts)
        self.blocks = nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    norm_layer=norm_layer,
                    dgmoe_enabled=bool(dgmoe_enabled),
                    dgmoe_impl=str(dgmoe_impl),
                    dgmoe_num_experts=int(dgmoe_num_experts),
                    dgmoe_expert_hidden_dim=int(dgmoe_expert_hidden_dim),
                    dgmoe_lambda=float(dgmoe_lambda),
                    dgmoe_temperature=float(dgmoe_temperature),
                    dgmoe_renormalize_gates=bool(dgmoe_renormalize_gates),
                    dgmoe_gate_residual=bool(dgmoe_gate_residual),
                    dgmoe_force_active=bool(dgmoe_force_active),
                )
                for i in range(num_blocks)
            ]
        )
        self.post_transformer_layer = post_transformer_layer
        self.weight_init_style = weight_init_style
        self.latest_expert_counts: Tensor | None = None
        self.latest_token_count: int = 0
        self.apply(self._init_weights)

    def forward(self, tokens: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Inputs
        - tokens: data of shape N x L x D (or L x N x D depending on the attention implementation)
        - attn: mask of shape L x L

        Output
        - x: data of shape N x L x D (or L x N x D depending on the attention implementation)
        """
        if self.pre_transformer_layer:
            tokens = self.pre_transformer_layer(tokens)
        prev_gate_logits: Tensor | None = None
        expert_counts: list[Tensor] = []
        self.latest_expert_counts = None
        self.latest_token_count = 0
        for _, blk in enumerate(self.blocks):
            tokens, gate_logits, counts = blk(tokens, attn_mask=attn_mask, prev_gate_logits=prev_gate_logits)
            if gate_logits is not None and counts is not None:
                prev_gate_logits = gate_logits
                expert_counts.append(counts)
        if expert_counts:
            self.latest_expert_counts = torch.stack(expert_counts, dim=0)
            # Note: tokens are in L×B×D inside the trunk.
            self.latest_token_count = int(tokens.shape[0] * tokens.shape[1])
        if self.post_transformer_layer:
            tokens = self.post_transformer_layer(tokens)
        return tokens

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor)
        return rearrange(x, self.rearrange_expr, **self.kwargs)


def get_sinusoid_encoding_table(position_start: int, position_end: int, d_hid: int) -> Tensor:
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position: int):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(position_start, position_end)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ResNet(PolicyStem):
    def __init__(
        self,
        output_dim: int = 10,
        weights: str = "DEFAULT",
        resnet_model: str = "resnet18",
        num_of_copy: int = 1,
    ) -> None:
        """ResNet Encoder for Images"""
        super().__init__()
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)
        self.num_of_copy = num_of_copy
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.input = input
        self.out_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size,
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        b, *_, h, w = x.shape
        x = x.view(len(x), -1, 3, h, w)
        x = x.view(-1, 3, h, w)
        # fixed image size
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.net(x).view(b, 512, -1).transpose(1, 2).contiguous()
        return feat
