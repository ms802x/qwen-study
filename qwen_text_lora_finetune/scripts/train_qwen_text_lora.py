#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import torch.nn.functional as F
from peft import LoraConfig
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from qwen_edit_runtime import load_pipeline, make_text_to_image_pipeline


MODEL_DIR = REPO / "official-models/Qwen-Image-Edit-2511"
META = ROOT / "dataset/metadata/train.jsonl"
IMAGE_ROOT = ROOT
OUT = ROOT / "outputs"
CHECKPOINTS = ROOT / "checkpoints"
ADAPTER_DIR = CHECKPOINTS / "qwen_text_lora"
CACHE = OUT / "training_cache.pt"
LOSS_JSONL = OUT / "training_loss.jsonl"

WIDTH = 512
HEIGHT = 512
MAX_SEQUENCE_LENGTH = 512
TRUE_CFG_SCALE = 4.0
NEGATIVE_PROMPT = "random text, fake text, misspelled text, watermark, logo, illustration, decoration"


def load_records() -> list[dict]:
    return [json.loads(line) for line in META.read_text(encoding="utf-8").splitlines() if line.strip()]


def pil_to_tensor(path: Path, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(HEIGHT, WIDTH, 3).permute(2, 0, 1).contiguous()
    tensor = data.to(device=device, dtype=dtype) / 127.5 - 1.0
    return tensor.unsqueeze(0)


def normalize_qwen_latents(pipe, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(pipe.vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    mean = mean.view(1, pipe.vae.config.z_dim, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(pipe.vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    inv_std = inv_std.view(1, pipe.vae.config.z_dim, 1, 1, 1)
    return (latents - mean) * inv_std


def cache_training_tensors(pipe, records: list[dict], force: bool = False) -> list[dict]:
    if CACHE.exists() and not force:
        print(f"loading cache {CACHE}", flush=True)
        return torch.load(CACHE, map_location="cpu", weights_only=False)

    print("caching VAE latents and prompt embeddings", flush=True)
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    cached: list[dict] = []
    pipe.vae.eval()
    pipe.text_encoder.eval()
    with torch.no_grad():
        for idx, row in enumerate(records, start=1):
            prompt = row["prompt"]
            image_path = IMAGE_ROOT / row["image_path"]
            image = pil_to_tensor(image_path, device=device, dtype=pipe.vae.dtype)
            image = image.unsqueeze(2)
            encoded = pipe.vae.encode(image).latent_dist.sample()
            latents = normalize_qwen_latents(pipe, encoded)
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
            packed = pipe._pack_latents(latents, 1, pipe.transformer.config.in_channels // 4, HEIGHT // pipe.vae_scale_factor, WIDTH // pipe.vae_scale_factor)
            prompt_embeds, prompt_mask = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )
            cached.append(
                {
                    "index": row["index"],
                    "kind": row["kind"],
                    "prompt": prompt,
                    "latents": packed.to("cpu", dtype=torch.bfloat16),
                    "prompt_embeds": prompt_embeds.to("cpu", dtype=torch.bfloat16),
                    "prompt_mask": None if prompt_mask is None else prompt_mask.to("cpu"),
                }
            )
            if idx % 10 == 0:
                print(f"  cached {idx}/{len(records)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(cached, CACHE)
    print(f"wrote cache {CACHE}", flush=True)
    return cached


def add_lora(transformer, *, rank: int, alpha: int, dropout: float, layer_mode: str) -> None:
    if layer_mode == "attention":
        target_modules = ["to_q", "to_k", "to_v", "to_out.0", "to_add_out"]
    elif layer_mode == "attention_mlp":
        target_modules = ["to_q", "to_k", "to_v", "to_out.0", "to_add_out", "net.0.proj", "net.2"]
    else:
        raise ValueError(f"unsupported layer mode {layer_mode}")
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
        lora_dropout=dropout,
    )
    transformer.add_adapter(config, adapter_name="text_lora")
    transformer.set_adapter("text_lora")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    records = load_records()
    print(f"records={len(records)} model={args.model_dir}", flush=True)
    edit_pipe = load_pipeline(
        args.model_dir,
        device=args.device,
        dtype_name=args.dtype,
        progress_bar=False,
        scheduler_name="beta",
        stochastic_sampling=False,
    )
    pipe = make_text_to_image_pipeline(edit_pipe, progress_bar=False)
    cached = cache_training_tensors(pipe, records, force=args.recache)

    transformer = pipe.transformer
    transformer.requires_grad_(False)
    add_lora(transformer, rank=args.rank, alpha=args.alpha, dropout=args.dropout, layer_mode=args.layer_mode)
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    transformer.train()

    trainable = [param for param in transformer.parameters() if param.requires_grad]
    trainable_count = sum(param.numel() for param in trainable)
    print(f"trainable_parameters={trainable_count:,}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    device = torch.device(args.device)
    dtype = transformer.dtype
    img_shapes = [[(1, HEIGHT // pipe.vae_scale_factor // 2, WIDTH // pipe.vae_scale_factor // 2)]]
    loss_rows: list[dict] = []
    start = time.perf_counter()
    LOSS_JSONL.write_text("", encoding="utf-8")

    for step in range(1, args.steps + 1):
        row = cached[(step - 1) % len(cached)]
        latents = row["latents"].to(device=device, dtype=dtype)
        prompt_embeds = row["prompt_embeds"].to(device=device, dtype=dtype)
        prompt_mask = None if row["prompt_mask"] is None else row["prompt_mask"].to(device=device)

        noise = torch.randn_like(latents)
        sigma = torch.rand((latents.shape[0],), device=device, dtype=dtype)
        sigma_view = sigma.view(-1, 1, 1)
        noisy_latents = (1.0 - sigma_view) * latents + sigma_view * noise
        target = noise - latents

        model_pred = transformer(
            hidden_states=noisy_latents,
            timestep=sigma,
            guidance=None,
            encoder_hidden_states_mask=prompt_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            attention_kwargs={},
            return_dict=False,
        )[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.detach().cpu())
        entry = {
            "step": step,
            "loss": loss_value,
            "kind": row["kind"],
            "dataset_index": row["index"],
            "elapsed_seconds": time.perf_counter() - start,
        }
        loss_rows.append(entry)
        with LOSS_JSONL.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if step == 1 or step % args.log_every == 0:
            recent = loss_rows[-min(len(loss_rows), args.log_every):]
            mean_loss = sum(item["loss"] for item in recent) / len(recent)
            print(f"step={step:04d}/{args.steps} loss={loss_value:.6f} recent_mean={mean_loss:.6f}", flush=True)

    transformer.save_lora_adapter(ADAPTER_DIR, adapter_name="text_lora", safe_serialization=True)
    summary = {
        "steps": args.steps,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "lr": args.lr,
        "layer_mode": args.layer_mode,
        "trainable_parameters": trainable_count,
        "final_loss": loss_rows[-1]["loss"],
        "mean_last_20_loss": sum(row["loss"] for row in loss_rows[-20:]) / min(20, len(loss_rows)),
        "elapsed_seconds": time.perf_counter() - start,
        "adapter_dir": str(ADAPTER_DIR),
        "cache": str(CACHE),
        "loss_jsonl": str(LOSS_JSONL),
    }
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    del pipe
    del edit_pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Qwen text-rendering LoRA on 100 synthetic images.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--layer-mode", choices=["attention", "attention_mlp"], default="attention")
    parser.add_argument("--recache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
