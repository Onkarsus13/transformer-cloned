from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:  # Optional video backends; pick the best available at runtime.
    import decord  # type: ignore

    _HAS_DECORD = True
    decord.bridge.set_bridge("numpy")
except Exception:  # pragma: no cover - optional dependency.
    decord = None
    _HAS_DECORD = False

try:
    import av  # type: ignore

    _HAS_PYAV = True
except Exception:  # pragma: no cover - optional dependency.
    av = None
    _HAS_PYAV = False

try:
    import imageio.v3 as imageio  # type: ignore

    _HAS_IMAGEIO = True
except Exception:  # pragma: no cover - optional dependency.
    imageio = None
    _HAS_IMAGEIO = False

from PIL import Image

logger = logging.getLogger(__name__)

MetadataLike = Union[str, Path, Sequence[Mapping[str, Any]]]


def _infer_available_backend() -> str:
    if _HAS_DECORD:
        return "decord"
    if _HAS_PYAV:
        return "pyav"
    if _HAS_IMAGEIO:
        return "imageio"
    raise ImportError(
        "No supported video decoding backend is available. Install one of: decord, pyav, or imageio[v3]."
    )


def _resolve_path(path: Union[str, Path], root: Optional[Union[str, Path]]) -> Path:
    path = Path(path)
    if path.is_absolute() or root is None:
        return path
    return Path(root) / path


def _compute_frame_indices(
    total_frames: int,
    num_frames: int,
    frame_skip: int,
    random_start: bool,
) -> List[int]:
    if total_frames <= 0:
        raise ValueError("Video has no frames to sample.")

    frame_skip = max(1, frame_skip)
    max_offset = max(0, total_frames - frame_skip * (num_frames - 1) - 1)
    start_index = random.randint(0, max_offset) if (random_start and max_offset > 0) else 0

    indices: List[int] = []
    last_index = total_frames - 1
    for i in range(num_frames):
        idx = start_index + i * frame_skip
        if idx > last_index:
            idx = last_index
        indices.append(idx)
    return indices


def load_video_frames(
    path: Union[str, Path],
    num_frames: int,
    frame_skip: int,
    random_start: bool = True,
    backend: Optional[str] = None,
) -> List[Image.Image]:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Unable to locate video file: {resolved_path}")

    backend = backend or _infer_available_backend()

    if backend == "decord":
        if not _HAS_DECORD:
            raise RuntimeError("Decord backend requested but not available.")
        reader = decord.VideoReader(str(resolved_path))
        indices = _compute_frame_indices(len(reader), num_frames, frame_skip, random_start)
        frames = reader.get_batch(indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]

    if backend == "pyav":
        if not _HAS_PYAV:
            raise RuntimeError("PyAV backend requested but not available.")
        with av.open(str(resolved_path)) as container:
            decoded_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        if not decoded_frames:
            raise ValueError(f"Video {resolved_path} yielded no frames via PyAV decoding.")
        indices = _compute_frame_indices(len(decoded_frames), num_frames, frame_skip, random_start)
        return [Image.fromarray(decoded_frames[i]) for i in indices]

    if backend == "imageio":
        if not _HAS_IMAGEIO:
            raise RuntimeError("imageio backend requested but not available.")
        decoded_frames = list(imageio.imiter(resolved_path))
        if not decoded_frames:
            raise ValueError(f"Video {resolved_path} yielded no frames via imageio decoding.")
        indices = _compute_frame_indices(len(decoded_frames), num_frames, frame_skip, random_start)
        return [Image.fromarray(decoded_frames[i]) for i in indices]

    raise ValueError(f"Unsupported backend '{backend}'.")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping):
        for key in ("data", "samples"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                return list(candidate)
        raise ValueError("JSON metadata must either be a list or contain a 'data'/'samples' list.")
    if not isinstance(data, list):
        raise ValueError("JSON metadata must be a list of records.")
    return list(data)


def _read_delimited(path: Path, delimiter: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [dict(row) for row in reader]


def _load_metadata(metadata: MetadataLike) -> List[Dict[str, Any]]:
    if isinstance(metadata, (str, Path)):
        path = Path(metadata)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file '{path}' does not exist.")
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".jl"}:
            return _read_jsonl(path)
        if suffix == ".json":
            return _read_json(path)
        if suffix == ".csv":
            return _read_delimited(path, ",")
        if suffix in {".tsv", ".txt"}:
            return _read_delimited(path, "\t")
        raise ValueError(f"Unsupported metadata extension '{suffix}'.")

    if isinstance(metadata, Sequence):
        return [dict(sample) for sample in metadata]

    raise TypeError("metadata must be a path or a sequence of dict-like annotations.")


@dataclass
class VideoTextSample:
    video: Path
    texts: Tuple[str, ...]
    negative_texts: Tuple[str, ...] = ()
    video_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def pick_positive(self) -> str:
        if len(self.texts) == 1:
            return self.texts[0]
        return random.choice(self.texts)

    def pick_negatives(
        self,
        num_negatives: int,
        *,
        global_pool: Sequence[str],
        avoid_text: str,
    ) -> List[str]:
        if num_negatives <= 0:
            return []

        negatives: List[str] = []

        if self.negative_texts:
            local_pool = list(self.negative_texts)
            random.shuffle(local_pool)
            for candidate in local_pool:
                if candidate and candidate != avoid_text:
                    negatives.append(candidate)
                if len(negatives) >= num_negatives:
                    break

        if len(negatives) >= num_negatives:
            return negatives[:num_negatives]

        fallback_pool = [text for text in global_pool if text and text != avoid_text]
        if not fallback_pool:
            raise ValueError("No negative texts available to sample from.")

        while len(negatives) < num_negatives:
            negatives.append(random.choice(fallback_pool))

        return negatives[:num_negatives]


class VideoTextDataset(Dataset):
    """Dataset that serves video/text pairs for contrastive + MLM objectives.

    Videos are decoded into ``num_frames`` frames, sampled with ``frame_skip``
    between them ("skip 5" corresponds to ``frame_skip=5``). Frames are
    processed through the provided SigLIP image processor and texts are
    tokenized with the matching tokenizer. Optional negative captions can be
    supplied via metadata or sampled from the global caption pool so the
    collator can form explicit positive/negative pairs. Masking is deferred to
    the collator so it can change every epoch.
    """

    def __init__(
        self,
        metadata: MetadataLike,
        processor: Optional[Any] = None,
        *,
        image_processor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        video_column: str = "video",
        text_column: str = "text",
        negative_text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        additional_columns: Optional[Sequence[str]] = None,
        video_root: Optional[Union[str, Path]] = None,
        num_frames: int = 64,
        frame_skip: int = 5,
        random_start: bool = True,
        backend: Optional[str] = None,
        max_text_length: int = 77,
        pad_to_max_length: bool = True,
        num_negatives: int = 1,
    ) -> None:
        self.samples_metadata = _load_metadata(metadata)
        if not self.samples_metadata:
            raise ValueError("No samples found in provided metadata.")

        if processor is not None:
            image_processor = getattr(processor, "image_processor", None) or image_processor
            tokenizer = getattr(processor, "tokenizer", None) or tokenizer

        if image_processor is None or tokenizer is None:
            raise ValueError("Both an image_processor and tokenizer (or a combined processor) must be provided.")

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.video_column = video_column
        self.text_column = text_column
        self.id_column = id_column
        self.additional_columns = list(additional_columns) if additional_columns else []
        self.video_root = Path(video_root) if video_root is not None else None
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.random_start = random_start
        self.backend = backend
        self.max_text_length = max_text_length
        self.pad_to_max_length = pad_to_max_length
        self.num_negatives = max(0, num_negatives)

        positive_pool: List[str] = []
        extra_negative_pool: List[str] = []

        self._resolved_samples: List[VideoTextSample] = []
        for sample in self.samples_metadata:
            if video_column not in sample or text_column not in sample:
                raise KeyError(f"Sample is missing required columns '{video_column}'/'{text_column}': {sample}")

            video_path = _resolve_path(sample[video_column], self.video_root)
            text_value = sample[text_column]
            if isinstance(text_value, str):
                texts = (text_value,)
            elif isinstance(text_value, Sequence):
                texts = tuple(str(t) for t in text_value if t)
                if not texts:
                    raise ValueError(f"Sample text list is empty for video '{video_path}'.")
            else:
                raise TypeError(f"Text column must resolve to a string or sequence of strings, got {type(text_value)}")

            negative_texts: Tuple[str, ...] = ()
            if negative_text_column:
                neg_value = sample.get(negative_text_column)
                if isinstance(neg_value, str):
                    negative_texts = (neg_value,)
                elif isinstance(neg_value, Sequence):
                    negative_texts = tuple(str(t) for t in neg_value if t)
                elif neg_value is not None:
                    raise TypeError(
                        f"Negative text column must resolve to a string or sequence of strings, got {type(neg_value)}"
                    )

            video_id = sample[id_column] if id_column and id_column in sample else None
            extra = {key: sample[key] for key in self.additional_columns if key in sample}
            self._resolved_samples.append(
                VideoTextSample(
                    video=video_path,
                    texts=texts,
                    negative_texts=negative_texts,
                    video_id=video_id,
                    metadata=extra or None,
                )
            )
            positive_pool.extend(texts)
            extra_negative_pool.extend(negative_texts)

        self._global_negative_pool: Tuple[str, ...] = tuple(positive_pool + extra_negative_pool)
        if self.num_negatives > 0 and len(self._global_negative_pool) <= 1:
            raise ValueError("Need at least two distinct captions to sample negatives from.")

    def __len__(self) -> int:
        return len(self._resolved_samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._resolved_samples[index]

        frames = load_video_frames(
            sample.video,
            num_frames=self.num_frames,
            frame_skip=self.frame_skip,
            random_start=self.random_start,
            backend=self.backend,
        )

        frame_inputs = self.image_processor(images=frames, return_tensors="pt")
        pixel_values = frame_inputs["pixel_values"]
        pixel_attention_mask = frame_inputs.get("pixel_attention_mask")
        spatial_shapes = frame_inputs.get("spatial_shapes")
        caption = sample.pick_positive()
        encoded_text = self.tokenizer(
            caption,
            max_length=self.max_text_length,
            padding="max_length" if self.pad_to_max_length else True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        example: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "text": caption,
        }
        example["video_path"] = str(sample.video)

        if pixel_attention_mask is not None:
            example["pixel_attention_mask"] = pixel_attention_mask
        if spatial_shapes is not None:
            example["spatial_shapes"] = spatial_shapes

        if self.num_negatives > 0:
            negative_texts = sample.pick_negatives(
                self.num_negatives,
                global_pool=self._global_negative_pool,
                avoid_text=caption,
            )
            padding_strategy: Union[bool, str] = "max_length" if self.pad_to_max_length else True
            encoded_negatives = self.tokenizer(
                negative_texts,
                max_length=self.max_text_length,
                padding=padding_strategy,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            example["negative_input_ids"] = encoded_negatives["input_ids"]
            example["negative_attention_mask"] = encoded_negatives["attention_mask"]
            example["negative_texts"] = negative_texts

        if sample.video_id is not None:
            example["video_id"] = sample.video_id
        if sample.metadata is not None:
            example.update(sample.metadata)

        return example


class ContrastiveVideoTextCollator:
    """Collate function producing SigLIP-ready batches with MLM targets.

    When negative captions are provided by the dataset, they are batched and
    returned as ``negative_input_ids``/``negative_attention_mask`` for explicit
    video-text matching with in-batch or per-sample negatives.
    """

    def __init__(self, tokenizer: Any, *, mlm_probability: float = 0.15) -> None:
        if tokenizer.mask_token is None:
            raise ValueError("Tokenizer must define a mask token for MLM.")
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad token id for MLM masking.")

        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples: Sequence[MutableMapping[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([example["pixel_values"] for example in examples], dim=0)
        input_ids = self._stack_with_padding(
            [example["input_ids"] for example in examples], pad_value=self.tokenizer.pad_token_id
        )
        attention_mask = self._stack_with_padding(
            [example["attention_mask"] for example in examples], pad_value=0
        )

        masked_input_ids, labels = self._mask_tokens(input_ids)

        batch: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "pixel_attention_mask" in examples[0]:
            batch["pixel_attention_mask"] = torch.stack(
                [example["pixel_attention_mask"] for example in examples], dim=0
            )
        if "spatial_shapes" in examples[0]:
            batch["spatial_shapes"] = torch.stack(
                [example["spatial_shapes"] for example in examples], dim=0
            )

        if "negative_input_ids" in examples[0]:
            negative_input_ids = self._stack_with_padding(
                [example["negative_input_ids"] for example in examples], pad_value=self.tokenizer.pad_token_id
            )
            negative_attention_mask = self._stack_with_padding(
                [example["negative_attention_mask"] for example in examples], pad_value=0
            )
            batch["negative_input_ids"] = negative_input_ids
            batch["negative_attention_mask"] = negative_attention_mask

        extras: Dict[str, List[Any]] = {}
        for example in examples:
            for key, value in example.items():
                if key in {
                    "pixel_values",
                    "input_ids",
                    "attention_mask",
                    "pixel_attention_mask",
                    "spatial_shapes",
                    "negative_input_ids",
                    "negative_attention_mask",
                }:
                    continue
                extras.setdefault(key, []).append(value)

        if "text" in extras:
            batch["texts"] = extras.pop("text")
        if "negative_texts" in extras:
            batch["negative_texts"] = extras.pop("negative_texts")
        for key, values in extras.items():
            if all(isinstance(v, torch.Tensor) for v in values):
                batch[key] = torch.stack(values)  # type: ignore[arg-type]
            else:
                batch[key] = values

        return batch

    def _mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs.clone()
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self._build_special_tokens_mask(labels)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(inputs == self.tokenizer.pad_token_id, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("Tokenizer.mask_token_id is required for MLM.")

        replace_with_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[replace_with_mask] = mask_token_id

        replace_with_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~replace_with_mask
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[replace_with_random] = random_words[replace_with_random]

        # Remaining 10% stay as original tokens.
        return inputs, labels

    def _build_special_tokens_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        special_tokens_mask = []
        for sequence in inputs.tolist():
            mask = self.tokenizer.get_special_tokens_mask(sequence, already_has_special_tokens=True)
            special_tokens_mask.append(mask)
        return torch.tensor(special_tokens_mask, dtype=torch.bool)

    def _stack_with_padding(self, tensors: Sequence[torch.Tensor], *, pad_value: int) -> torch.Tensor:
        if not tensors:
            raise ValueError("Cannot stack empty tensor sequence.")
        max_len = max(tensor.size(-1) for tensor in tensors)
        if all(tensor.size(-1) == max_len for tensor in tensors):
            return torch.stack(tensors, dim=0)

        padded = [self._pad_last_dim(tensor, max_len, pad_value) for tensor in tensors]
        return torch.stack(padded, dim=0)

    @staticmethod
    def _pad_last_dim(tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        pad_amount = target_len - tensor.size(-1)
        if pad_amount <= 0:
            return tensor
        return F.pad(tensor, (0, pad_amount), value=pad_value)


def build_video_text_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collator: Optional[ContrastiveVideoTextCollator] = None,
) -> DataLoader:
    if collator is None:
        raise ValueError("A ContrastiveVideoTextCollator instance must be provided.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )


__all__ = [
    "VideoTextDataset",
    "ContrastiveVideoTextCollator",
    "build_video_text_dataloader",
]
