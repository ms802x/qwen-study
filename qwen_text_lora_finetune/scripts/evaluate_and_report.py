#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from qwen_edit_runtime import load_pipeline, make_text_to_image_pipeline, run_text_to_image


MODEL_DIR = REPO / "official-models/Qwen-Image-Edit-2511"
ADAPTER_DIR = ROOT / "checkpoints/qwen_text_lora"
OUT = ROOT / "outputs/eval"
REPORT = ROOT / "report"
REPORT_ASSETS = REPORT / "assets"
DATASET_CONTACT = ROOT / "dataset/contact_sheet.jpg"
TRAINING_SUMMARY = ROOT / "outputs/training_summary.json"
LOSS_JSONL = ROOT / "outputs/training_loss.jsonl"

FONT_LATIN = Path("/opt/conda/fonts/DejaVuSans.ttf")
WIDTH = 512
HEIGHT = 512
STEPS = 40
TRUE_CFG_SCALE = 4.0
MAX_SEQUENCE_LENGTH = 512
NEGATIVE_PROMPT = "random text, fake text, misspelled text, watermark, logo, illustration, decoration"


CASES = [
    {
        "id": "seen_english",
        "kind": "seen",
        "seed": 41001,
        "prompt": 'Create a clean white poster with the exact English text "clarity design signal".',
    },
    {
        "id": "seen_arabic",
        "kind": "seen",
        "seed": 41002,
        "prompt": 'Create a clean white poster with the exact Arabic text "ثابت دقيق حرف شكل".',
    },
    {
        "id": "seen_mixed",
        "kind": "seen",
        "seed": 41003,
        "prompt": 'Create a clean bilingual poster with the exact English text "stable exact glyph shape" and the exact Arabic text "ثابت دقيق حرف شكل".',
    },
    {
        "id": "heldout_english",
        "kind": "heldout",
        "seed": 41004,
        "prompt": 'Create a clean white poster with the exact English text "fresh marker system layout".',
    },
    {
        "id": "heldout_arabic",
        "kind": "heldout",
        "seed": 41005,
        "prompt": 'Create a clean white poster with the exact Arabic text "عنوان واضح مباشر حاد".',
    },
    {
        "id": "heldout_mixed",
        "kind": "heldout",
        "seed": 41006,
        "prompt": 'Create a clean bilingual poster with the exact English text "sharp layout focus" and the exact Arabic text "وضوح حاد مباشر".',
    },
]


def clean_html(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def generate(pipe, case: dict, variant: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{case['id']}_{variant}.jpg"
    if path.exists():
        return {"path": path, "elapsed_seconds": None}
    started = time.perf_counter()
    image, elapsed = run_text_to_image(
        pipe,
        prompt=case["prompt"],
        negative_prompt=NEGATIVE_PROMPT,
        true_cfg_scale=TRUE_CFG_SCALE,
        num_inference_steps=STEPS,
        seed=case["seed"],
        height=HEIGHT,
        width=WIDTH,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    wall = time.perf_counter() - started
    image.save(path, quality=95)
    return {"path": path, "elapsed_seconds": elapsed, "wall_seconds": wall}


def make_eval_sheet(rows: list[dict], out_path: Path) -> None:
    cols = 2
    cell_w = 340
    image_h = 310
    label_h = 58
    sheet = Image.new("RGB", (cols * cell_w, len(CASES) * (image_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT_LATIN), 15)
    small = ImageFont.truetype(str(FONT_LATIN), 12)
    for row_idx, case in enumerate(CASES):
        for col, variant in enumerate(["base", "lora"]):
            result = next(row for row in rows if row["id"] == case["id"] and row["variant"] == variant)
            x = col * cell_w
            y = row_idx * (image_h + label_h)
            draw.rectangle((x, y, x + cell_w - 1, y + image_h + label_h - 1), outline=(210, 210, 210))
            title = f"{case['id']} - {'LoRA' if variant == 'lora' else 'Base'}"
            draw.text((x + 8, y + 8), title[:38], fill=(20, 20, 20), font=font)
            draw.text((x + 8, y + 31), case["kind"], fill=(90, 90, 90), font=small)
            image = ImageOps.exif_transpose(Image.open(result["path"])).convert("RGB")
            image.thumbnail((cell_w - 20, image_h - 14), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (cell_w - image.width) // 2, y + label_h + (image_h - image.height) // 2))
    sheet.save(out_path, quality=94)


def make_loss_plot(out_path: Path) -> None:
    rows = [json.loads(line) for line in LOSS_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    width, height = 900, 320
    pad = 44
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT_LATIN), 14)
    losses = [min(row["loss"], 0.25) for row in rows]
    max_loss = max(losses) if losses else 1.0
    draw.rectangle((pad, pad, width - pad, height - pad), outline=(210, 210, 210))
    if len(losses) > 1:
        points = []
        for idx, value in enumerate(losses):
            x = pad + idx * (width - 2 * pad) / (len(losses) - 1)
            y = height - pad - (value / max_loss) * (height - 2 * pad)
            points.append((x, y))
        draw.line(points, fill="#225f83", width=2)
    draw.text((pad, 14), "Training loss clipped at 0.25 for readability", fill="black", font=font)
    draw.text((pad, height - 32), f"steps: {len(rows)}", fill=(80, 80, 80), font=font)
    image.save(out_path)


def copy_asset(src: Path) -> str:
    REPORT_ASSETS.mkdir(parents=True, exist_ok=True)
    dst = REPORT_ASSETS / src.name
    shutil.copy2(src, dst)
    return f"assets/{dst.name}"


def write_report(eval_rows: list[dict]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    REPORT_ASSETS.mkdir(parents=True, exist_ok=True)
    summary = json.loads(TRAINING_SUMMARY.read_text(encoding="utf-8"))
    eval_sheet = OUT / "before_after_contact_sheet.jpg"
    make_eval_sheet(eval_rows, eval_sheet)
    loss_plot = OUT / "training_loss_plot.jpg"
    make_loss_plot(loss_plot)

    dataset_rel = copy_asset(DATASET_CONTACT)
    eval_rel = copy_asset(eval_sheet)
    loss_rel = copy_asset(loss_plot)
    adapter_rel = "../checkpoints/qwen_text_lora/pytorch_lora_weights.safetensors"

    rows_html = "\n".join(
        f"<tr><td>{case['id']}</td><td>{case['kind']}</td><td>{case['prompt']}</td></tr>"
        for case in CASES
    )
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen Text LoRA Finetune Report</title>
  <style>
    body{{margin:0;background:#f6f1e7;color:#23211c;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.58}}
    main{{width:min(1180px,calc(100vw - 28px));margin:0 auto;padding:28px 0 64px}}
    section{{background:#fffdf8;border:1px solid #d8c8a4;border-radius:8px;padding:22px;margin:18px 0}}
    h1{{font-size:clamp(32px,5vw,54px);line-height:1.05;margin:0 0 12px}}
    h2{{font-size:27px;margin:0 0 12px}}
    img{{max-width:100%;height:auto;display:block;border-radius:4px}}
    .note{{border-left:4px solid #b53a31;background:#fff3ee;padding:13px 15px;color:#23211c}}
    table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}
    td,th{{border:1px solid #e2d6bd;padding:8px 10px;text-align:left;vertical-align:top}}
    th{{background:#f1e8d7}}
    code{{background:#f0e8d7;border:1px solid #dfd0ae;border-radius:5px;padding:1px 5px}}
  </style>
</head>
<body>
<main>
  <h1>Qwen Text LoRA Finetune Report</h1>
  <p>Small controlled LoRA experiment on 100 synthetic text-poster images. The dataset includes English-only, Arabic-only, stacked bilingual, and structured bilingual layouts.</p>
  <section>
    <h2>Training Setup</h2>
    <p>Model: <code>Qwen-Image-Edit-2511</code> used as text-to-image through the shared Qwen Image transformer. Training resolution: 512x512. Adapter: rank {summary['rank']} attention LoRA on Q/K/V/output projections. Trainable parameters: {summary['trainable_parameters']:,}. Steps: {summary['steps']}. Learning rate: {summary['lr']}.</p>
    <p>Saved adapter: <a href="{adapter_rel}">pytorch_lora_weights.safetensors</a></p>
  </section>
  <section>
    <h2>Result Summary</h2>
    <p class="note">This run does not solve Arabic text rendering. The LoRA learned the synthetic poster prior strongly and improves simple English poster text, but the before/after generations do not show reliable exact Arabic or bilingual transcription. Treat it as a useful pipeline proof, not a quality fix.</p>
    <p>The main evidence is visual: compare the base and LoRA columns below. The LoRA changes layout/style toward the synthetic dataset, makes English prompts cleaner in this small test, and still leaves Arabic glyph identity/word correctness weak.</p>
  </section>
  <section>
    <h2>Dataset Contact Sheet</h2>
    <img src="{dataset_rel}" alt="Synthetic training dataset contact sheet">
  </section>
  <section>
    <h2>Training Loss</h2>
    <p>Final loss: {summary['final_loss']:.6f}; mean of last 20 logged steps: {summary['mean_last_20_loss']:.6f}. The curve is noisy because every step samples a different image and noise timestep.</p>
    <img src="{loss_rel}" alt="Training loss plot">
  </section>
  <section>
    <h2>Before / After Generations</h2>
    <img src="{eval_rel}" alt="Before and after LoRA generation contact sheet">
  </section>
  <section>
    <h2>Evaluation Prompts</h2>
    <table><thead><tr><th>ID</th><th>Type</th><th>Prompt</th></tr></thead><tbody>{rows_html}</tbody></table>
  </section>
  <section>
    <h2>Recommendation</h2>
    <p>For a real Arabic poster fix, this result says the pipeline is working but the dataset is too small and too synthetic. Next run should use thousands of high-quality rendered poster crops, include exact OCR/CLIP-style text scoring, and likely target the non-catastrophic text-sensitive layer bands from the ablation probe rather than only generic attention LoRA.</p>
  </section>
</main>
</body>
</html>
"""
    (REPORT / "index.html").write_text(clean_html(html), encoding="utf-8")
    (OUT / "eval_results.json").write_text(json.dumps(eval_rows, default=str, indent=2), encoding="utf-8")
    print(REPORT / "index.html")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading base pipeline", flush=True)
    edit_pipe = load_pipeline(MODEL_DIR, device="cuda", dtype_name="bf16", progress_bar=False, scheduler_name="beta")
    pipe = make_text_to_image_pipeline(edit_pipe, progress_bar=False)
    rows: list[dict] = []
    for case in CASES:
        print(f"base {case['id']}", flush=True)
        result = generate(pipe, case, "base")
        rows.append({"id": case["id"], "variant": "base", **result})

    print(f"loading LoRA {ADAPTER_DIR}", flush=True)
    pipe.transformer.load_lora_adapter(
        ADAPTER_DIR,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="text_lora",
        prefix=None,
    )
    pipe.transformer.set_adapter("text_lora")
    for case in CASES:
        print(f"lora {case['id']}", flush=True)
        result = generate(pipe, case, "lora")
        rows.append({"id": case["id"], "variant": "lora", **result})
    write_report(rows)


if __name__ == "__main__":
    run()
