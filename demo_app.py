

from typing import Optional
import os

import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


# -----------------------------
# Model and processor loading
# -----------------------------

def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda:1", torch.bfloat16
    return "cpu", torch.float32


DEVICE, DTYPE = get_device_and_dtype()

# Allow overriding via environment variables
MODEL_ID = "google/gemma-3-12b-it"
PEFT_PATH = "/data2/onkar/modi_script/model_epoch_3"
CACHE_DIR = "/data2/qyo9735/challenge/hf_models"


def load_model_and_processor():
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        cache_dir=CACHE_DIR,
    )

    model = PeftModel.from_pretrained(
        base,
        PEFT_PATH,
        device_map=DEVICE,
        torch_dtype=DTYPE,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


MODEL, PROCESSOR = load_model_and_processor()
MODEL.eval()


# -----------------------------
# Sampling helpers
# -----------------------------

def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative_probs > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probs = torch.where(cutoff, torch.zeros_like(sorted_probs), sorted_probs)
    # Map back to original index order
    filtered = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
    # avoid log(0) later
    filtered = torch.clamp(filtered, min=1e-9)
    return torch.log(filtered)


def generate_text(
    image: Image.Image,
    prompt: str = "Transliterate the following Modi script to Devanagari script.",
    max_new_tokens: int = 350,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: Optional[int] = None,
) -> str:
    if image is None:
        return "Please upload an input image."

    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text"}]},
    ]

    inputs = PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(MODEL.device, dtype=DTYPE)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"].to(dtype=MODEL.dtype, device=MODEL.device)
    input_len = input_ids.shape[-1]

    with torch.no_grad():
        while True:
            outputs = MODEL(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )

            logits = outputs.logits[:, -1, :]

            if temperature is not None and temperature > 0:
                logits = logits / max(temperature, 1e-6)

            logits = top_p_filtering(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

            if (
                next_token.item() == PROCESSOR.tokenizer.eos_token_id
                or (input_ids.shape[1] - input_len) >= max_new_tokens
            ):
                break

    generation = input_ids[:, input_len:][0]
    generated_text = PROCESSOR.decode(generation, skip_special_tokens=True)
    return generated_text.strip()


# -----------------------------
# Gradio UI
# -----------------------------

def build_ui():
    with gr.Blocks(title="Modi → Devanagari Transliteration") as demo:
        gr.Markdown("""
        # Modi → Devanagari Transliteration
        Upload an image containing Modi script and generate its transliteration.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Input Image")
                prompt_in = gr.Textbox(
                    label="Prompt",
                    value="Transliterate the following Modi script to Devanagari script.",
                    lines=2,
                )
                with gr.Accordion("Advanced settings", open=False):
                    max_tokens_in = gr.Slider(16, 1024, value=350, step=1, label="Max new tokens")
                    temperature_in = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                    top_p_in = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
                    seed_in = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)

                run_btn = gr.Button("Transliterate", variant="primary")

            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Output", lines=10)

        def _run(img, prmpt, max_tok, temp, topp, seed):
            s = None if seed is None or int(seed) < 0 else int(seed)
            return generate_text(
                image=img,
                prompt=prmpt,
                max_new_tokens=int(max_tok),
                temperature=float(temp),
                top_p=float(topp),
                seed=s,
            )

        run_btn.click(
            fn=_run,
            inputs=[image_in, prompt_in, max_tokens_in, temperature_in, top_p_in, seed_in],
            outputs=[output_text],
        )

        gr.Markdown(
            f"Running on `{DEVICE}` with dtype `{DTYPE}` | Model: `{MODEL_ID}` | Adapter: `{PEFT_PATH}`"
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=8001, show_api=False, share=True)
