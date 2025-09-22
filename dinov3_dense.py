from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer


logger = logging.getLogger(__name__)


class SDPMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def _shape(self, tensor: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        kv_states = key_value_states if key_value_states is not None else hidden_states
        kv_batch, kv_len, _ = kv_states.shape

        if kv_batch != batch_size:
            raise ValueError("Query and key/value batches must match")

        query = self._shape(self.q_proj(hidden_states), batch_size, seq_len)
        key = self._shape(self.k_proj(kv_states), batch_size, kv_len)
        value = self._shape(self.v_proj(kv_states), batch_size, kv_len)

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attn_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 4:
                attn_mask = attention_mask
            else:
                raise ValueError("Unsupported attention mask rank")
            attn_mask = attn_mask.to(query.dtype)
            attn_mask = torch.where(
                attn_mask > 0.5,
                torch.zeros_like(attn_mask),
                torch.full_like(attn_mask, float("-inf")),
            )

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.o_proj(attn_output)


class ParallelAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.spatial_attn = SDPMultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.temporal_attn = SDPMultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.cross_attn = SDPMultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=True),
            nn.Dropout3d(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        video_states: torch.Tensor,
        *,
        text_states: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, frames, height, width, channels = video_states.shape
        normed = self.input_norm(video_states)

        spatial_in = normed.view(bsz * frames, height * width, channels)
        spatial_out = self.spatial_attn(spatial_in)
        spatial_out = spatial_out.view(bsz, frames, height, width, channels)

        temporal_in = normed.permute(0, 2, 3, 1, 4).contiguous().view(bsz * height * width, frames, channels)
        temporal_out = self.temporal_attn(temporal_in)
        temporal_out = temporal_out.view(bsz, height, width, frames, channels).permute(0, 3, 1, 2, 4)

        if text_attention_mask is not None:
            text_mask = text_attention_mask.unsqueeze(1).expand(-1, frames, -1)
            text_mask = text_mask.reshape(bsz * frames, -1)
        else:
            text_mask = None

        cross_in = normed.view(bsz * frames, height * width, channels)
        text_kv = text_states.unsqueeze(1).expand(-1, frames, -1, -1).reshape(bsz * frames, -1, channels)
        cross_out = self.cross_attn(cross_in, key_value_states=text_kv, attention_mask=text_mask)
        cross_out = cross_out.view(bsz, frames, height, width, channels)

        fused = torch.cat([spatial_out, temporal_out, cross_out], dim=-1)
        fused = fused.permute(0, 4, 1, 2, 3)
        fused = self.fuse_conv(fused)
        fused = fused.permute(0, 2, 3, 4, 1)

        updated = video_states + fused
        updated = updated + self.mlp(self.output_norm(updated))
        return updated


@dataclass
class DinoV3ParallelOutput:
    loss: Optional[torch.Tensor] = None
    contrastive_loss: Optional[torch.Tensor] = None
    text_alignment_loss: Optional[torch.Tensor] = None
    video_text_matching_loss: Optional[torch.Tensor] = None
    mlm_loss: Optional[torch.Tensor] = None
    negative_loss: Optional[torch.Tensor] = None
    logits_per_video: Optional[torch.Tensor] = None
    logits_per_text: Optional[torch.Tensor] = None
    video_embeddings: Optional[torch.Tensor] = None
    text_embeddings: Optional[torch.Tensor] = None
    fused_video_states: Optional[torch.Tensor] = None


class DinoV3ParallelVideoTextMatcher(nn.Module):
    def __init__(
        self,
        *,
        vision_model_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        text_model_name: str = "google/umt5-xxxl",
        hidden_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 16,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        freeze_text_encoder: bool = True,
        lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        vision_lora_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.lora_target_modules = tuple(lora_target_modules)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.freeze_text_encoder = freeze_text_encoder

        vision_config = AutoConfig.from_pretrained(vision_model_name)
        vision_config._attn_implementation = "sdpa"
        self.vision_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        base_vision_model = AutoModel.from_pretrained(vision_model_name, config=vision_config)

        if vision_lora_path is not None:
            adapter_path = Path(vision_lora_path)
            if adapter_path.is_dir():
                self.vision_model = PeftModel.from_pretrained(
                    base_vision_model,
                    adapter_path,
                    is_trainable=True,
                )
            else:
                raise FileNotFoundError(f"LoRA adapter directory not found at {adapter_path}")
        else:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=list(self.lora_target_modules),
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.vision_model = get_peft_model(base_vision_model, lora_config)

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        if self.text_tokenizer.pad_token is None and self.text_tokenizer.eos_token is not None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.text_model = AutoModel.from_pretrained(text_model_name)

        if freeze_text_encoder:
            for param in self.text_model.parameters():
                param.requires_grad = False

        vision_hidden = self.vision_model.config.hidden_size
        text_hidden = getattr(self.text_model.config, "hidden_size", None) or getattr(self.text_model.config, "d_model")

        self.hidden_dim = hidden_dim or vision_hidden
        self.patch_norm = nn.LayerNorm(vision_hidden)
        self.video_proj = nn.Linear(vision_hidden, self.hidden_dim)
        self.text_proj = nn.Linear(text_hidden, self.hidden_dim)

        self.parallel_layers = nn.ModuleList(
            [
                ParallelAttentionBlock(
                    self.hidden_dim,
                    num_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        proj_dim = projection_dim or self.hidden_dim
        self.video_norm = nn.LayerNorm(self.hidden_dim)
        self.video_projection = nn.Linear(self.hidden_dim, proj_dim)
        self.text_norm = nn.LayerNorm(self.hidden_dim)
        self.text_projection = nn.Linear(self.hidden_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio

    @property
    def image_processor(self) -> AutoImageProcessor:
        return self.vision_processor

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self.text_tokenizer

    def encode_video(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch_size, frames = pixel_values.shape[:2]
        flat_pixels = pixel_values.view(batch_size * frames, *pixel_values.shape[2:])
        outputs = self.vision_model(pixel_values=flat_pixels, output_hidden_states=True, return_dict=True)
        patch_states = outputs.last_hidden_state
        register_tokens = getattr(self.vision_model.config, "num_register_tokens", 0)
        patch_states = patch_states[:, 1 + register_tokens :, :]
        patch_states = self.patch_norm(patch_states)

        num_patches = patch_states.shape[1]
        side = int(math.sqrt(num_patches))
        if side * side != num_patches:
            raise ValueError("Input resolution does not yield square patch grid")

        patch_states = patch_states.view(batch_size, frames, side, side, -1)
        return patch_states, (side, side)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, "encoder_last_hidden_state"):
            hidden = outputs.encoder_last_hidden_state
        else:
            raise ValueError("Text model did not return hidden states")
        return hidden

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        negative_input_ids: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        **_: torch.Tensor,
    ) -> DinoV3ParallelOutput:
        batch_size = pixel_values.shape[0]
        video_states, _ = self.encode_video(pixel_values)
        video_states = self.video_proj(video_states)

        text_states_raw = self.encode_text(input_ids, attention_mask)
        text_states = self.text_proj(text_states_raw)

        fused_video = video_states
        for layer in self.parallel_layers:
            fused_video = layer(
                fused_video,
                text_states=text_states,
                text_attention_mask=attention_mask,
            )

        fused_flat = fused_video.permute(0, 4, 1, 2, 3)
        pooled_video = fused_flat.mean(dim=(2, 3, 4))
        video_embeddings = self.video_projection(self.video_norm(pooled_video))
        video_embeddings = F.normalize(video_embeddings, dim=-1)

        mask = attention_mask.unsqueeze(-1)
        masked_text = text_states * mask
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_text = masked_text.sum(dim=1) / denom.squeeze(1)
        text_embeddings = self.text_projection(self.text_norm(pooled_text))
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * video_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_video.t()

        contrastive_loss = None
        text_alignment_loss = None
        video_text_matching_loss = None
        mlm_loss = None
        negative_loss = None
        total_loss = None

        if return_loss:
            targets = torch.arange(batch_size, device=logits_per_video.device)
            contrastive_loss = (
                F.cross_entropy(logits_per_video, targets) + F.cross_entropy(logits_per_text, targets)
            ) * 0.5
            text_alignment_loss = contrastive_loss
            total_loss = contrastive_loss

            if negative_input_ids is not None and negative_attention_mask is not None:
                bsz, k_neg, seq = negative_input_ids.shape
                neg_ids = negative_input_ids.view(bsz * k_neg, seq)
                neg_mask = negative_attention_mask.view(bsz * k_neg, seq)
                neg_hidden = self.encode_text(neg_ids, neg_mask)
                neg_hidden = self.text_proj(neg_hidden)
                neg_mask_exp = neg_mask.unsqueeze(-1)
                neg_masked = neg_hidden * neg_mask_exp
                neg_denom = neg_mask_exp.sum(dim=1, keepdim=True).clamp(min=1)
                neg_pooled = neg_masked.sum(dim=1) / neg_denom.squeeze(1)
                neg_embeddings = self.text_projection(self.text_norm(neg_pooled))
                neg_embeddings = F.normalize(neg_embeddings, dim=-1)
                neg_embeddings = neg_embeddings.view(bsz, k_neg, -1)

                pos_logits = logits_per_video.diag().unsqueeze(1)
                neg_logits = logit_scale * torch.einsum("bd,bkd->bk", video_embeddings, neg_embeddings)
                combined = torch.cat([pos_logits, neg_logits], dim=1)
                video_text_matching_loss = F.cross_entropy(
                    combined,
                    torch.zeros(bsz, dtype=torch.long, device=combined.device),
                )
                negative_loss = video_text_matching_loss
                total_loss = total_loss + video_text_matching_loss if total_loss is not None else video_text_matching_loss

        return DinoV3ParallelOutput(
            loss=total_loss,
            contrastive_loss=contrastive_loss,
            text_alignment_loss=text_alignment_loss,
            video_text_matching_loss=video_text_matching_loss,
            mlm_loss=mlm_loss,
            negative_loss=negative_loss,
            logits_per_video=logits_per_video,
            logits_per_text=logits_per_text,
            video_embeddings=video_embeddings,
            text_embeddings=text_embeddings,
            fused_video_states=fused_video,
        )

    def _export_config(self) -> Dict[str, Any]:
        return {
            "vision_model_name": self.vision_model_name,
            "text_model_name": self.text_model_name,
            "hidden_dim": self.hidden_dim,
            "projection_dim": self.video_projection.out_features,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "mlp_ratio": self.mlp_ratio,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "freeze_text_encoder": self.freeze_text_encoder,
            "lora_target_modules": list(self.lora_target_modules),
        }

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        config_path = save_path / "config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(self._export_config(), handle, indent=2)

        model_state = {}
        for key, value in self.state_dict().items():
            if key.startswith("vision_model.") or key.startswith("text_model."):
                continue
            model_state[key] = value.detach().cpu()
        torch.save(model_state, save_path / "model_state.pt")

        if isinstance(self.vision_model, PeftModel):
            adapter_dir = save_path / "vision_adapter"
            self.vision_model.save_pretrained(adapter_dir)
        else:
            logger.warning("Vision model is not a PEFT model; skipping adapter save.")

    @classmethod
    def from_pretrained(
        cls,
        directory: Union[str, Path],
        **overrides: Any,
    ) -> "DinoV3ParallelVideoTextMatcher":
        load_path = Path(directory)
        config_path = load_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Expected configuration at {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config.update(overrides)

        adapter_path = load_path / "vision_adapter"
        if adapter_path.exists():
            config.setdefault("vision_lora_path", str(adapter_path))

        model = cls(**config)

        state_path = load_path / "model_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            missing = [key for key in missing if not key.startswith(("vision_model.", "text_model."))]
            if missing:
                logger.warning("Missing keys when loading parallel head: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys when loading parallel head: %s", unexpected)
        else:
            logger.warning("Model weights not found at %s; loaded base configuration only.", state_path)

        return model


__all__ = [
    "DinoV3ParallelVideoTextMatcher",
    "DinoV3ParallelOutput",
]
