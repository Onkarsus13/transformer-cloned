from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Siglip2TextModel, Siglip2VisionModel
except ImportError as exc:  # pragma: no cover - transformers is a runtime dependency.
    raise ImportError("SiglipVideoTextMatcher requires transformers>=4.39 with Siglip2 support.") from exc


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (tensor * cos) + (rotate_half(tensor) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _build_cache(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.max_position_embeddings}."
            )

        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._cos_cached.size(1) < seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(dtype=dtype)[None, :, :]
            sin = emb.sin().to(dtype=dtype)[None, :, :]
            self._cos_cached = cos
            self._sin_cached = sin
        return self._cos_cached[:, :seq_len, :], self._sin_cached[:, :seq_len, :]

    def forward(
        self,
        seq_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._build_cache(seq_len, dtype=dtype, device=device)

    def gather_from_position_ids(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_position = int(position_ids.max().item()) + 1
        cos, sin = self._build_cache(max_position, dtype=dtype, device=device)
        cos = cos[0].index_select(0, position_ids.view(-1)).view(*position_ids.shape, -1)
        sin = sin[0].index_select(0, position_ids.view(-1)).view(*position_ids.shape, -1)
        return cos.unsqueeze(1), sin.unsqueeze(1)


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=max_position_embeddings)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if position_ids is not None:
            cos, sin = self.rotary_emb.gather_from_position_ids(
                position_ids,
                dtype=query.dtype,
                device=query.device,
            )
        else:
            cos, sin = self.rotary_emb(
                seq_len,
                dtype=query.dtype,
                device=query.device,
            )
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        context = self.out_proj(context)
        return self.output_dropout(context)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        self.attention = RotarySelfAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_input = self.norm1(hidden_states)
        hidden_states = hidden_states + self.attention(
            attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@dataclass
class SiglipVideoTextMatcherOutput:
    loss: Optional[torch.Tensor] = None
    contrastive_loss: Optional[torch.Tensor] = None
    mlm_loss: Optional[torch.Tensor] = None
    negative_loss: Optional[torch.Tensor] = None
    logits_per_video: Optional[torch.Tensor] = None
    logits_per_text: Optional[torch.Tensor] = None
    mlm_logits: Optional[torch.Tensor] = None
    video_embeddings: Optional[torch.Tensor] = None
    text_embeddings: Optional[torch.Tensor] = None
    sequence_hidden_states: Optional[torch.Tensor] = None


class SiglipVideoTextMatcher(nn.Module):
    def __init__(
        self,
        vision_model: Siglip2VisionModel,
        text_model: Siglip2TextModel,
        *,
        num_layers: int = 3,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
        projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model

        hidden_size = text_model.config.hidden_size
        vision_hidden = vision_model.config.hidden_size
        if hidden_size != vision_hidden:
            raise ValueError(
                "SiglipVideoTextMatcher requires text and vision encoders with matching hidden sizes,"
                f" got {hidden_size} and {vision_hidden}."
            )

        self.hidden_size = hidden_size
        if num_heads is None:
            num_heads = text_model.config.num_attention_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.video_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.text_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.video_cls_token, std=0.02)
        nn.init.normal_(self.text_cls_token, std=0.02)

        self.type_embeddings = nn.Embedding(2, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    max_position_embeddings=max_position_embeddings,
                )
                for _ in range(num_layers)
            ]
        )

        projection_dim = projection_dim or hidden_size
        self.video_projection = nn.Linear(hidden_size, projection_dim)
        self.text_projection = nn.Linear(hidden_size, projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        vocab_size = text_model.config.vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.text_model.get_input_embeddings().weight

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        negative_input_ids: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> SiglipVideoTextMatcherOutput:
        batch_size, num_frames = pixel_values.shape[:2]
        patch_seq_len = pixel_values.shape[2]
        frame_dim = pixel_values.shape[3]

        flat_pixel_values = pixel_values.view(batch_size * num_frames, patch_seq_len, frame_dim)
        flat_pixel_mask = pixel_attention_mask.view(batch_size * num_frames, patch_seq_len)
        flat_spatial_shapes = spatial_shapes.view(batch_size * num_frames, 2)

        vision_outputs = self.vision_model(
            pixel_values=flat_pixel_values,
            pixel_attention_mask=flat_pixel_mask,
            spatial_shapes=flat_spatial_shapes,
        )
        frame_embeddings = vision_outputs.pooler_output.view(batch_size, num_frames, self.hidden_size)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_token_states = text_outputs.last_hidden_state  # (B, L, H)

        video_cls = self.video_cls_token.expand(batch_size, -1, -1)
        text_cls = self.text_cls_token.expand(batch_size, -1, -1)

        video_tokens = torch.cat([video_cls, frame_embeddings], dim=1)
        text_tokens = torch.cat([text_cls, text_token_states], dim=1)

        sequence = torch.cat([video_tokens, text_tokens], dim=1)

        video_mask = torch.ones(batch_size, video_tokens.size(1), device=sequence.device, dtype=attention_mask.dtype)
        text_mask = torch.cat(
            [torch.ones(batch_size, 1, device=sequence.device, dtype=attention_mask.dtype), attention_mask], dim=1
        )
        combined_mask = torch.cat([video_mask, text_mask], dim=1)
        modality_ids = torch.cat(
            [
                torch.zeros(batch_size, video_tokens.size(1), device=sequence.device, dtype=torch.long),
                torch.ones(batch_size, text_tokens.size(1), device=sequence.device, dtype=torch.long),
            ],
            dim=1,
        )

        sequence = sequence + self.type_embeddings(modality_ids)
        sequence = self.embed_dropout(sequence)

        seq_len = sequence.size(1)
        position_ids = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch_size, -1)

        for layer in self.transformer_layers:
            sequence = layer(sequence, attention_mask=combined_mask, position_ids=position_ids)

        video_cls_index = 0
        text_cls_index = video_tokens.size(1)

        video_hidden = sequence[:, video_cls_index, :]
        text_hidden = sequence[:, text_cls_index, :]

        video_embeddings = F.normalize(self.video_projection(video_hidden), dim=-1)
        text_embeddings = F.normalize(self.text_projection(text_hidden), dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * torch.matmul(video_embeddings, text_embeddings.t())
        logits_per_text = logits_per_video.t()

        text_token_hidden = sequence[:, text_cls_index + 1 :, :]
        mlm_logits = self.lm_head(text_token_hidden)

        contrastive_loss = None
        mlm_loss_value = None
        negative_loss = None
        total_loss = None

        if return_loss:
            targets = torch.arange(batch_size, device=logits_per_video.device)
            contrastive_loss = (
                F.cross_entropy(logits_per_video, targets) + F.cross_entropy(logits_per_text, targets)
            ) * 0.5

            if labels is not None:
                mlm_loss_value = F.cross_entropy(
                    mlm_logits.view(-1, mlm_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

            if negative_input_ids is not None and negative_attention_mask is not None:
                bsz, k_neg, seq_neg = negative_input_ids.shape
                neg_inputs = negative_input_ids.view(bsz * k_neg, seq_neg)
                neg_masks = negative_attention_mask.view(bsz * k_neg, seq_neg)
                negative_outputs = self.text_model(
                    input_ids=neg_inputs,
                    attention_mask=neg_masks,
                )
                negative_embeddings = negative_outputs.pooler_output.view(bsz, k_neg, self.hidden_size)
                negative_embeddings = F.normalize(self.text_projection(negative_embeddings), dim=-1)

                pos_logits = logits_per_video.diag().unsqueeze(1)
                neg_logits = logit_scale * torch.einsum("bd,bkd->bk", video_embeddings, negative_embeddings)
                combined = torch.cat([pos_logits, neg_logits], dim=1)
                negative_loss = F.cross_entropy(
                    combined,
                    torch.zeros(bsz, dtype=torch.long, device=combined.device),
                )

            total_loss = contrastive_loss
            if mlm_loss_value is not None:
                total_loss = total_loss + mlm_loss_value if total_loss is not None else mlm_loss_value
            if negative_loss is not None:
                total_loss = total_loss + negative_loss if total_loss is not None else negative_loss

        return SiglipVideoTextMatcherOutput(
            loss=total_loss,
            contrastive_loss=contrastive_loss,
            mlm_loss=mlm_loss_value,
            negative_loss=negative_loss,
            logits_per_video=logits_per_video,
            logits_per_text=logits_per_text,
            mlm_logits=mlm_logits,
            video_embeddings=video_embeddings,
            text_embeddings=text_embeddings,
            sequence_hidden_states=sequence,
        )


def compute_retrieval_metrics(
    logits_per_video: torch.Tensor,
    *,
    topk: Tuple[int, ...] = (1, 5, 10),
) -> dict[str, torch.Tensor]:
    batch_size = logits_per_video.size(0)
    targets = torch.arange(batch_size, device=logits_per_video.device)
    metrics: dict[str, torch.Tensor] = {}
    rankings = logits_per_video.argsort(dim=-1, descending=True)
    for k in topk:
        if k > logits_per_video.size(-1):
            continue
        correct = (rankings[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean()
        metrics[f"video_to_text_top{k}"] = correct
    return metrics


__all__ = [
    "SiglipVideoTextMatcher",
    "SiglipVideoTextMatcherOutput",
    "compute_retrieval_metrics",
]
