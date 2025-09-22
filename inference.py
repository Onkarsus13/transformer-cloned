from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import Siglip2Processor, Siglip2TextModel, Siglip2VisionModel, set_seed

from .data import ContrastiveVideoTextCollator, VideoTextDataset
from .sg_match_model import SiglipVideoTextMatcher, compute_retrieval_metrics
from .dinov3_dense import DinoV3ParallelVideoTextMatcher


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video-text retrieval inference with a trained SigLIP matcher.")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata file pointing to videos and captions.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint or final weights.")
    parser.add_argument("--video-root", type=Path, help="Optional root to prepend to relative video paths.")
    parser.add_argument("--pretrained", type=str, default="google/siglip2-large-patch16-512")
    parser.add_argument("--model-type", choices=["siglip", "dinov3"], default="siglip")
    parser.add_argument("--vision-model-name", type=str, default="facebook/dinov3-vit7b16-pretrain-lvd1689m")
    parser.add_argument("--text-model-name", type=str, default="google/umt5-xxxl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--random-start", action="store_true", help="Sample a random clip rather than the first clip.")
    parser.add_argument("--max-text-length", type=int, default=77)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--fusion-heads", type=int, default=16)
    parser.add_argument("--fusion-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, help="Torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if CUDA is available.")
    parser.add_argument("--fp16", action="store_true", help="Run inference with float16 autocast on CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=str, default="1,5,10", help="Comma separated retrieval cutoffs.")
    parser.add_argument("--output-json", type=Path, help="Optional path to dump retrieval results and metrics.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def parse_topk(value: str) -> Tuple[int, ...]:
    parts = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not parts:
        raise ValueError("At least one integer must be provided to --topk.")
    positive = [v for v in parts if v > 0]
    if not positive:
        raise ValueError("--topk cutoffs must be positive integers.")
    return tuple(sorted(set(positive)))


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    pixel_values = out.get("pixel_values")
    if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 4:
        if "pixel_attention_mask" not in out:
            mask = torch.ones_like(pixel_values[..., 0], dtype=torch.long)
            out["pixel_attention_mask"] = mask
        if "spatial_shapes" not in out:
            batch_size, num_frames, patch_count, _ = pixel_values.shape
            side = int(patch_count ** 0.5)
            spatial = torch.full((batch_size, num_frames, 2), side, device=device, dtype=torch.long)
            out["spatial_shapes"] = spatial
    return out


def load_components(
    args: argparse.Namespace,
) -> Tuple[Any, Any, nn.Module]:
    if args.model_type == "siglip":
        processor = Siglip2Processor.from_pretrained(args.pretrained)
        vision = Siglip2VisionModel.from_pretrained(args.pretrained)
        text = Siglip2TextModel.from_pretrained(args.pretrained)
        model = SiglipVideoTextMatcher(
            vision_model=vision,
            text_model=text,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_position_embeddings=args.num_frames + args.max_text_length + 2,
        )
        return processor.image_processor, processor.tokenizer, model

    if args.model_type == "dinov3":
        checkpoint_path = args.checkpoint
        if checkpoint_path.is_dir() and (checkpoint_path / "config.json").exists():
            model = DinoV3ParallelVideoTextMatcher.from_pretrained(
                checkpoint_path,
                vision_model_name=args.vision_model_name,
                text_model_name=args.text_model_name,
            )
            setattr(model, "_loaded_from_pretrained", True)
        else:
            model = DinoV3ParallelVideoTextMatcher(
                vision_model_name=args.vision_model_name,
                text_model_name=args.text_model_name,
                num_layers=args.num_layers,
                num_heads=args.fusion_heads,
                dropout=args.dropout,
                mlp_ratio=args.fusion_mlp_ratio,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                freeze_text_encoder=True,
            )
        return model.image_processor, model.tokenizer, model

    raise ValueError(f"Unsupported model_type '{args.model_type}'")


def load_checkpoint(model: nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {checkpoint} does not contain a state dict.")
    model.load_state_dict(state, strict=True)


def build_dataloader(
    image_processor: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> DataLoader:
    dataset = VideoTextDataset(
        metadata=args.metadata,
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_root=args.video_root,
        num_frames=args.num_frames,
        frame_skip=args.frame_skip,
        random_start=args.random_start,
        max_text_length=args.max_text_length,
        num_negatives=0,
        pad_to_max_length=True,
    )
    collator = ContrastiveVideoTextCollator(tokenizer, mlm_probability=0.0)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
    )
    return dataloader


def collect_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    model.eval()
    video_embeddings: List[torch.Tensor] = []
    text_embeddings: List[torch.Tensor] = []
    all_video_paths: List[str] = []
    all_texts: List[str] = []

    use_autocast = use_fp16 and device.type != "cpu"

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Encoding", leave=False)
        for batch in progress:
            texts = batch.get("texts")
            if isinstance(texts, list):
                all_texts.extend(texts)
            video_paths = batch.get("video_path")
            if isinstance(video_paths, list):
                all_video_paths.extend(video_paths)

            model_inputs = {
                key: batch[key]
                for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes", "input_ids", "attention_mask")
                if key in batch
            }
            model_inputs = move_batch_to_device(model_inputs, device)

            autocast_context = (
                torch.autocast(device_type=device.type, dtype=torch.float16)
                if use_autocast
                else nullcontext()
            )
            with autocast_context:
                forward_inputs = {
                    "pixel_values": model_inputs["pixel_values"],
                    "input_ids": model_inputs["input_ids"],
                    "attention_mask": model_inputs["attention_mask"],
                    "return_loss": False,
                }
                for key in ("pixel_attention_mask", "spatial_shapes"):
                    if key in model_inputs:
                        forward_inputs[key] = model_inputs[key]

                outputs = model(**forward_inputs)

            if outputs.video_embeddings is None or outputs.text_embeddings is None:
                raise RuntimeError("Model did not return video/text embeddings during inference.")

            video_embeddings.append(outputs.video_embeddings.detach().to("cpu"))
            text_embeddings.append(outputs.text_embeddings.detach().to("cpu"))

    if not video_embeddings:
        raise RuntimeError("No batches were processed; check the metadata path and dataset configuration.")

    return (
        torch.cat(video_embeddings, dim=0),
        torch.cat(text_embeddings, dim=0),
        all_video_paths,
        all_texts,
    )


def build_rankings(
    similarity: torch.Tensor,
    queries: Sequence[str],
    candidates: Sequence[str],
    topk: Tuple[int, ...],
    *,
    prefix: str,
) -> List[Dict[str, object]]:
    if similarity.size(0) != len(queries):
        raise ValueError(f"Expected {similarity.size(0)} queries but received {len(queries)} labels.")
    if similarity.size(1) == 0:
        raise ValueError("Similarity matrix has zero candidates.")

    max_k = min(similarity.size(1), max(topk))
    top_scores, top_indices = similarity.topk(max_k, dim=-1)

    results: List[Dict[str, object]] = []
    for idx in range(similarity.size(0)):
        entry: Dict[str, object] = {
            "query_index": idx,
            prefix: queries[idx] if idx < len(queries) else None,
        }
        matches = [
            {
                "rank": rank + 1,
                "candidate_index": int(candidate_idx),
                "candidate": candidates[int(candidate_idx)] if int(candidate_idx) < len(candidates) else None,
                "score": float(top_scores[idx, rank].item()),
            }
            for rank, candidate_idx in enumerate(top_indices[idx])
        ]
        for k in topk:
            entry[f"top{k}"] = matches[: min(k, len(matches))]
        results.append(entry)
    return results


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    topk = parse_topk(args.topk)

    if args.fp16 and args.cpu:
        raise ValueError("--fp16 cannot be combined with --cpu.")

    image_processor, tokenizer, model = load_components(args)
    if not getattr(model, "_loaded_from_pretrained", False):
        load_checkpoint(model, args.checkpoint)

    device_str = args.device or ("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)
    if args.fp16 and device.type != "cuda":
        raise ValueError("--fp16 requires a CUDA device.")

    model.to(device)

    dataloader = build_dataloader(image_processor, tokenizer, args)

    use_fp16 = args.fp16 and device.type == "cuda"
    video_embeddings, text_embeddings, video_paths, texts = collect_embeddings(model, dataloader, device, use_fp16)

    logit_scale = model.logit_scale.detach().exp().cpu()
    similarity = torch.matmul(video_embeddings, text_embeddings.t()) * logit_scale

    metrics_v2t = compute_retrieval_metrics(similarity, topk=topk)
    metrics_t2v_raw = compute_retrieval_metrics(similarity.t(), topk=topk)
    metrics_t2v = {
        key.replace("video_to_text", "text_to_video"): value for key, value in metrics_t2v_raw.items()
    }
    metrics = {**{k: float(v) for k, v in metrics_v2t.items()}, **{k: float(v) for k, v in metrics_t2v.items()}}

    video_rankings = build_rankings(similarity, video_paths, texts, topk, prefix="video")
    text_rankings = build_rankings(similarity.t(), texts, video_paths, topk, prefix="text")

    logger.info("Video->Text metrics: %s", {k: metrics[k] for k in metrics if k.startswith("video_to_text")})
    logger.info("Text->Video metrics: %s", {k: metrics[k] for k in metrics if k.startswith("text_to_video")})

    if args.output_json:
        payload = {
            "metrics": metrics,
            "video_to_text": video_rankings,
            "text_to_video": text_rankings,
        }
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Saved retrieval results to %s", args.output_json)


if __name__ == "__main__":
    main()
