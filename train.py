from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from transformers import (
    Siglip2Processor,
    Siglip2TextModel,
    Siglip2VisionModel,
    get_linear_schedule_with_warmup,
    set_seed,
)

from .data import (
    ContrastiveVideoTextCollator,
    VideoTextDataset,
)
from .sg_match_model import (
    SiglipVideoTextMatcher,
    compute_retrieval_metrics,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SigLIP-based video-text matcher with MLM.")
    parser.add_argument("--train-metadata", type=Path, required=True, help="Path to training metadata (json/jsonl/csv).")
    parser.add_argument("--eval-metadata", type=Path, help="Optional evaluation metadata path.")
    parser.add_argument("--video-root", type=Path, help="Optional root dir to prepend to relative video paths.")
    parser.add_argument("--negative-text-column", type=str, help="Column in metadata containing explicit negatives.")
    parser.add_argument("--pretrained", type=str, default="google/siglip2-large-patch16-512")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--num-negatives", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--max-text-length", type=int, default=77)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--random-start", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs when eval set provided.")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--resume-from", type=Path, help="Path to checkpoint to resume from.")
    parser.add_argument("--freeze-vision", action="store_true")
    parser.add_argument("--freeze-text", action="store_true")

    args = parser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    return args


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def load_models(args: argparse.Namespace) -> SiglipVideoTextMatcher:
    logger.info("Loading SigLIP2 models from %s", args.pretrained)
    vision = Siglip2VisionModel.from_pretrained(args.pretrained)
    text = Siglip2TextModel.from_pretrained(args.pretrained)

    model = SiglipVideoTextMatcher(
        vision_model=vision,
        text_model=text,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_position_embeddings=args.num_frames + args.max_text_length + 2,
    )

    if args.freeze_vision:
        for param in model.vision_model.parameters():
            param.requires_grad = False
    if args.freeze_text:
        for param in model.text_model.parameters():
            param.requires_grad = False

    return model


def create_dataloader(
    *,
    metadata_path: Path,
    processor: Siglip2Processor,
    args: argparse.Namespace,
    shuffle: bool,
    batch_size: int,
    mlm_probability: float,
) -> DataLoader:
    dataset = VideoTextDataset(
        metadata=metadata_path,
        processor=processor,
        video_root=args.video_root,
        negative_text_column=args.negative_text_column,
        num_frames=args.num_frames,
        frame_skip=args.frame_skip,
        random_start=args.random_start,
        max_text_length=args.max_text_length,
        num_negatives=args.num_negatives,
    )
    collator = ContrastiveVideoTextCollator(processor.tokenizer, mlm_probability=mlm_probability)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
    )
    return dataloader


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    if "pixel_attention_mask" not in out:
        mask = torch.ones_like(out["pixel_values"][..., 0], dtype=torch.long)
        out["pixel_attention_mask"] = mask
    if "spatial_shapes" not in out:
        batch_size, num_frames, patch_count, _ = out["pixel_values"].shape
        height = width = int(math.sqrt(patch_count))
        spatial = torch.full((batch_size, num_frames, 2), height, device=device, dtype=torch.long)
        out["spatial_shapes"] = spatial
    return out


def prepare_optimizer(
    model: SiglipVideoTextMatcher,
    args: argparse.Namespace,
    train_steps: int,
) -> tuple[torch.optim.Optimizer, int]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = max(1, int(train_steps * args.warmup_ratio))
    return optimizer, warmup_steps


def save_checkpoint_state(
    *,
    model: SiglipVideoTextMatcher,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
) -> dict[str, torch.Tensor]:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
    }
    return state


def load_checkpoint(
    *,
    path: Path,
    model: SiglipVideoTextMatcher,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return int(state.get("epoch", 0)), int(state.get("step", 0))


def evaluate(
    model: SiglipVideoTextMatcher,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> Dict[str, float]:
    model.eval()
    video_embeds = []
    text_embeds = []
    with torch.no_grad():
        progress = tqdm(dataloader, desc="Eval", leave=False, disable=not accelerator.is_main_process)
        for batch in progress:
            batch_on_device = move_batch_to_device(batch, accelerator.device)
            outputs = model(
                pixel_values=batch_on_device["pixel_values"],
                pixel_attention_mask=batch_on_device["pixel_attention_mask"],
                spatial_shapes=batch_on_device["spatial_shapes"],
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                return_loss=False,
            )
            video_embeds.append(outputs.video_embeddings)
            text_embeds.append(outputs.text_embeddings)

    if not video_embeds:
        return {}

    video_tensor = torch.cat(video_embeds, dim=0)
    text_tensor = torch.cat(text_embeds, dim=0)
    video_tensor = accelerator.gather_for_metrics(video_tensor)
    text_tensor = accelerator.gather_for_metrics(text_tensor)

    if not accelerator.is_main_process:
        return {}

    video_tensor = video_tensor.cpu()
    text_tensor = text_tensor.cpu()
    logits = torch.matmul(video_tensor, text_tensor.t())
    metrics = compute_retrieval_metrics(logits)
    return {k: v.item() for k, v in metrics.items()}


def train() -> None:
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if args.fp16 else "no",
        cpu=args.no_cuda,
    )

    if accelerator.is_main_process:
        setup_logging(args.output_dir)
    accelerator.wait_for_everyone()

    set_seed(args.seed)

    if accelerator.is_main_process:
        logger.info("Using device(s): %s", accelerator.device)

    processor = Siglip2Processor.from_pretrained(args.pretrained)
    train_loader = create_dataloader(
        metadata_path=args.train_metadata,
        processor=processor,
        args=args,
        shuffle=True,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_probability,
    )

    eval_loader = None
    if args.eval_metadata:
        eval_loader = create_dataloader(
            metadata_path=args.eval_metadata,
            processor=processor,
            args=args,
            shuffle=False,
            batch_size=args.eval_batch_size,
            mlm_probability=0.0,
        )

    model = load_models(args)

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = updates_per_epoch * args.num_epochs
    optimizer, warmup_steps = prepare_optimizer(model, args, total_steps)

    if eval_loader is not None:
        model, optimizer, train_loader, eval_loader = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            eval_loader,
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_epoch = 0
    global_step = 0
    if args.resume_from:
        accelerator.print(f"Loading checkpoint from {args.resume_from}")
        load_epoch, load_step = load_checkpoint(
            path=args.resume_from,
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = load_epoch
        global_step = load_step

    if accelerator.is_main_process:
        logger.info("Starting training: %d epochs, %d steps per epoch", args.num_epochs, len(train_loader))

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        running_loss = 0.0
        contrastive_meter = 0.0
        mlm_meter = 0.0
        negative_meter = 0.0
        step_in_epoch = 0

        progress = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            leave=False,
            disable=not accelerator.is_main_process,
        )

        optimizer.zero_grad()

        for batch in train_loader:
            with accelerator.accumulate(model):
                batch_on_device = move_batch_to_device(batch, accelerator.device)
                outputs = model(
                    pixel_values=batch_on_device["pixel_values"],
                    pixel_attention_mask=batch_on_device["pixel_attention_mask"],
                    spatial_shapes=batch_on_device["spatial_shapes"],
                    input_ids=batch_on_device["input_ids"],
                    attention_mask=batch_on_device["attention_mask"],
                    labels=batch_on_device["labels"],
                    negative_input_ids=batch_on_device.get("negative_input_ids"),
                    negative_attention_mask=batch_on_device.get("negative_attention_mask"),
                )

                accelerator.backward(outputs.loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                loss_value = accelerator.gather(outputs.loss.detach()).mean().item()
                running_loss += loss_value
                if outputs.contrastive_loss is not None:
                    contrastive_value = accelerator.gather(outputs.contrastive_loss.detach()).mean().item()
                    contrastive_meter += contrastive_value
                if outputs.mlm_loss is not None:
                    mlm_value = accelerator.gather(outputs.mlm_loss.detach()).mean().item()
                    mlm_meter += mlm_value
                if outputs.negative_loss is not None:
                    negative_value = accelerator.gather(outputs.negative_loss.detach()).mean().item()
                    negative_meter += negative_value

                global_step += 1
                step_in_epoch += 1
                progress.update(1)

                if accelerator.is_main_process and global_step % args.log_steps == 0:
                    avg_loss = running_loss / max(1, step_in_epoch)
                    msg = {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": avg_loss,
                        "contrastive": contrastive_meter / max(1, step_in_epoch),
                        "mlm": mlm_meter / max(1, step_in_epoch),
                        "neg": negative_meter / max(1, step_in_epoch),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    logger.info(json.dumps(msg))
                    
        progress.close()

        accelerator.wait_for_everyone()

        if (epoch + 1) % args.save_every == 0:
            state = save_checkpoint_state(
                model=accelerator.unwrap_model(model),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step=global_step,
            )
            if accelerator.is_main_process:
                ckpt_path = args.output_dir / f"checkpoint-epoch{epoch + 1:03d}-step{global_step:06d}.pt"
                accelerator.save(state, ckpt_path)
                logger.info("Saved checkpoint to %s", ckpt_path)

        if eval_loader and ((epoch + 1) % args.eval_every == 0):
            metrics = evaluate(model, eval_loader, accelerator)
            if accelerator.is_main_process and metrics:
                logger.info("Eval epoch %d: %s", epoch + 1, json.dumps(metrics))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = args.output_dir / "model_final.pt"
        accelerator.save(accelerator.unwrap_model(model).state_dict(), final_path)
        logger.info("Training complete. Final weights saved to %s", final_path)


if __name__ == "__main__":
    train()
