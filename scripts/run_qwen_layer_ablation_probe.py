#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from qwen_edit_runtime import load_pipeline, make_text_to_image_pipeline, run_text_to_image

MODEL_DIR = ROOT / "official-models/Qwen-Image-Edit-2511"
OUT = ROOT / "outputs/qwen_text_failure_next_experiments/exp_07_layer_ablation"
WEB = ROOT / "research_pages/qwen_text_failure"
WEB_EXP = WEB / "experiments"
WEB_ASSETS = WEB / "assets/next_experiments/exp_07_layer_ablation"

PROMPT = (
    'Create a plain white square poster with only the exact text "AI PIZZA" centered '
    "in large black clean sans-serif letters. No other text, no logo, no watermark, no decoration."
)
NEGATIVE_PROMPT = "extra text, misspelled text, pseudo text, watermark, logo, decorative letters"
WIDTH = 512
HEIGHT = 512
STEPS = 20
TRUE_CFG_SCALE = 4.0
MAX_SEQUENCE_LENGTH = 512
SEED = 97123
FONT_LATIN = Path("/opt/conda/fonts/DejaVuSans.ttf")


def clean_html(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def skip_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    encoder_hidden_states_mask=None,
    temb=None,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    modulate_index=None,
):
    return encoder_hidden_states, hidden_states


def set_skipped_layers(transformer, originals: dict[int, Any], skipped: set[int]) -> None:
    blocks = transformer.transformer_blocks
    for idx, block in enumerate(blocks):
        if idx in skipped:
            if idx not in originals:
                originals[idx] = block.forward
            block.forward = MethodType(skip_forward, block)
        elif idx in originals:
            block.forward = originals[idx]


def restore_layers(transformer, originals: dict[int, Any]) -> None:
    for idx, original in originals.items():
        transformer.transformer_blocks[idx].forward = original
    originals.clear()


def image_metrics(path: Path, baseline_path: Path) -> dict[str, float]:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    baseline = ImageOps.exif_transpose(Image.open(baseline_path)).convert("RGB")
    arr = np.asarray(image).astype(np.float32)
    base = np.asarray(baseline).astype(np.float32)
    diff = np.abs(arr - base)
    return {
        "pixel_mae": float(diff.mean()),
        "pixel_rmse": float(np.sqrt(np.mean((arr - base) ** 2))),
        "pixel_max_delta": float(diff.max()),
    }


def save_diff_image(path: Path, baseline_path: Path, out_path: Path) -> None:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    baseline = ImageOps.exif_transpose(Image.open(baseline_path)).convert("RGB")
    diff = ImageChops.difference(image, baseline)
    # Amplify subtle differences enough to read in the report.
    diff = diff.point(lambda value: min(255, value * 3))
    diff.save(out_path, quality=94)


def run_case(pipe, transformer, originals: dict[int, Any], *, layer: int | None, force: bool) -> dict[str, Any]:
    case_id = "baseline" if layer is None else f"skip_layer_{layer:02d}"
    image_path = OUT / f"{case_id}.jpg"
    meta_path = OUT / f"{case_id}.json"
    if image_path.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    set_skipped_layers(transformer, originals, set() if layer is None else {layer})
    started = time.perf_counter()
    image, elapsed = run_text_to_image(
        pipe,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        true_cfg_scale=TRUE_CFG_SCALE,
        num_inference_steps=STEPS,
        seed=SEED,
        height=HEIGHT,
        width=WIDTH,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    wall_seconds = time.perf_counter() - started
    image.save(image_path, quality=95)
    meta = {
        "experiment": "exp_07_layer_ablation",
        "case_id": case_id,
        "skipped_layer": layer,
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "steps": STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "elapsed_seconds": elapsed,
        "wall_seconds": wall_seconds,
        "output_path": str(image_path),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def copy_asset(src: Path) -> str:
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    dst = WEB_ASSETS / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return f"../assets/next_experiments/exp_07_layer_ablation/{dst.name}"


def make_contact_sheet(rows: list[dict[str, Any]], out_path: Path) -> None:
    cols = 4
    cell_w = 280
    image_h = 250
    label_h = 56
    sheet = Image.new("RGB", (cols * cell_w, ((len(rows) + cols - 1) // cols) * (image_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.truetype(str(FONT_LATIN), 16)
    small_font = ImageFont.truetype(str(FONT_LATIN), 12)
    for idx, row in enumerate(rows):
        x = (idx % cols) * cell_w
        y = (idx // cols) * (image_h + label_h)
        draw.rectangle((x, y, x + cell_w - 1, y + image_h + label_h - 1), outline=(210, 210, 210))
        title = "Baseline" if row["skipped_layer"] is None else f"Skip layer {row['skipped_layer']:02d}"
        draw.text((x + 8, y + 8), title, fill=(20, 20, 20), font=label_font)
        draw.text((x + 8, y + 31), f"MAE {row.get('pixel_mae', 0):.2f}", fill=(90, 90, 90), font=small_font)
        image = ImageOps.exif_transpose(Image.open(row["output_path"])).convert("RGB")
        image.thumbnail((cell_w - 18, image_h - 16), Image.Resampling.LANCZOS)
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + label_h + (image_h - image.height) // 2))
    sheet.save(out_path, quality=94)


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fields = [
        "case_id",
        "skipped_layer",
        "pixel_mae",
        "pixel_rmse",
        "pixel_max_delta",
        "elapsed_seconds",
        "wall_seconds",
        "output_path",
        "prompt",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_experiment_index() -> None:
    experiments = [
        ("Experiment 01: Text Size Sweep", "Tests whether requested text size changes success/failure across English, Arabic, and Chinese.", "exp_01_size_sweep.html", "exp_01_size_sweep/exp_01_size_sweep_contact_sheet.jpg"),
        ("Experiment 02: Seed Sweep", "Tests whether success/failure is stable or seed-dependent under the same prompt.", "exp_02_seed_sweep.html", "exp_02_seed_sweep/exp_02_seed_sweep_contact_sheet.jpg"),
        ("Experiment 03: Reference Glyph Copy", "Tests whether already-rendered reference glyphs help Qwen copy or preserve exact text.", "exp_03_reference_glyph_copy.html", "exp_03_reference_glyph_copy/exp_03_reference_glyph_copy_contact_sheet.jpg"),
        ("Experiment 04: Background Complexity", "Tests whether visual context makes exact text rendering worse.", "exp_04_background_complexity.html", "exp_04_background_complexity/exp_04_background_complexity_contact_sheet.jpg"),
        ("Experiment 05: Arabic Prompt Variants", "Tests whether prompt wording alone can repair Arabic text fidelity.", "exp_05_arabic_prompt_variants.html", "exp_05_arabic_prompt_variants/exp_05_arabic_prompt_variants_contact_sheet.jpg"),
        ("Experiment 06: Long Text Stress Test", "Tests English, Arabic, and Chinese at 10, 20, and 50 target words.", "exp_06_long_text_stress.html", "exp_06_long_text_stress/exp_06_long_text_stress_contact_sheet.jpg"),
        ("Experiment 07: Layer Ablation Probe", "Skips one Qwen transformer block at a time and ranks visual impact versus baseline.", "exp_07_layer_ablation.html", "exp_07_layer_ablation/exp_07_layer_ablation_top_contact_sheet.jpg"),
    ]
    links = "\n".join(
        f"""<div class="card"><img src="../assets/next_experiments/{image}" alt="{html.escape(title)}"><div class="caption"><a href="{page}"><strong>{html.escape(title)}</strong></a><br>{html.escape(summary)}</div></div>"""
        for title, summary, page, image in experiments
    )
    index = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen Text Rendering Next Experiments</title>
  <link rel="stylesheet" href="../shared.css">
</head>
<body>
<main>
  <p class="breadcrumb"><a href="../">Text rendering study</a> / Next experiments</p>
  <h1>Qwen Text Rendering: Next Experiments</h1>
  <p class="lead">Each page is a separate investigation with its own contact sheet, individual images, prompts, and CSV.</p>
  <section><h2>Experiment Pages</h2><div class="grid">{links}</div></section>
</main>
</body>
</html>
"""
    (WEB_EXP / "index.html").write_text(clean_html(index), encoding="utf-8")


def write_page(rows: list[dict[str, Any]]) -> None:
    WEB_EXP.mkdir(parents=True, exist_ok=True)
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)

    baseline = next(row for row in rows if row["skipped_layer"] is None)
    ablated = [row for row in rows if row["skipped_layer"] is not None]
    ablated_sorted = sorted(ablated, key=lambda row: row["pixel_mae"], reverse=True)
    top_rows = [baseline] + ablated_sorted[:15]
    all_rows = [baseline] + sorted(ablated, key=lambda row: row["skipped_layer"])

    top_sheet = OUT / "exp_07_layer_ablation_top_contact_sheet.jpg"
    all_sheet = OUT / "exp_07_layer_ablation_all_contact_sheet.jpg"
    make_contact_sheet(top_rows, top_sheet)
    make_contact_sheet(all_rows, all_sheet)
    csv_path = OUT / "exp_07_layer_ablation_results.csv"
    write_csv([baseline] + ablated_sorted, csv_path)

    baseline_rel = copy_asset(Path(baseline["output_path"]))
    top_sheet_rel = copy_asset(top_sheet)
    all_sheet_rel = copy_asset(all_sheet)
    csv_rel = copy_asset(csv_path)
    for row in top_rows:
        copy_asset(Path(row["output_path"]))

    top_layer = ablated_sorted[0]
    top_five = ", ".join(f"{row['skipped_layer']:02d} (MAE {row['pixel_mae']:.2f})" for row in ablated_sorted[:5])
    table_rows = "\n".join(
        f"<tr><td>{row['skipped_layer']:02d}</td><td>{row['pixel_mae']:.3f}</td><td>{row['pixel_rmse']:.3f}</td><td>{row['elapsed_seconds']:.2f}s</td></tr>"
        for row in ablated_sorted[:20]
    )
    cards = "\n".join(
        f"""<div class="card"><img src="../assets/next_experiments/exp_07_layer_ablation/{Path(row['output_path']).name}" alt="Skip layer {row['skipped_layer']}"><div class="caption"><strong>Skip layer {row['skipped_layer']:02d}</strong><br>MAE {row['pixel_mae']:.2f}, RMSE {row['pixel_rmse']:.2f}</div></div>"""
        for row in ablated_sorted[:15]
    )
    page = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment 07: Layer Ablation Probe</title>
  <link rel="stylesheet" href="../shared.css">
</head>
<body>
<main>
  <p class="breadcrumb"><a href="../">Text rendering study</a> / Next experiments</p>
  <h1>Experiment 07: Layer Ablation Probe</h1>
  <p class="lead">A direct Qwen transformer probe on GPU 1. The script skips one of the 60 transformer blocks at a time, generates the same fixed <code>AI PIZZA</code> prompt, and ranks outputs by pixel difference from the unmodified baseline.</p>
  <section>
    <h2>Observed Result</h2>
    <p>The strongest single skipped layer by pixel MAE is layer {top_layer['skipped_layer']:02d}. The top five most disruptive layers are {html.escape(top_five)}.</p>
    <p>Interpretation: layers 00 and 59 are catastrophic trajectory layers in this probe, so they dominate the visual metric. For text-specific follow-up, the more useful candidates are the high-impact non-catastrophic skips where the poster remains but glyph shape, spacing, or layout changes: layers 23, 32, 34, 37, 49, and 51.</p>
    <p class="note">This is an ablation ranking, not proof that one layer alone stores text knowledge. Skipping a layer changes the whole denoising trajectory, and the metric is visual pixel delta rather than OCR accuracy. It is still useful for choosing layer bands to inspect or fine-tune next.</p>
  </section>
  <section>
    <h2>Baseline</h2>
    <div class="sheet"><img src="{baseline_rel}" alt="Layer ablation baseline image"></div>
    <p class="small">Prompt: {html.escape(PROMPT)}</p>
  </section>
  <section>
    <h2>Most Disruptive Layers</h2>
    <div class="sheet"><img src="{top_sheet_rel}" alt="Top layer ablation contact sheet"></div>
  </section>
  <section>
    <h2>Top Layer Outputs</h2>
    <div class="grid">{cards}</div>
  </section>
  <section>
    <h2>All Layers Contact Sheet</h2>
    <div class="sheet"><img src="{all_sheet_rel}" alt="All layer ablation contact sheet"></div>
  </section>
  <section>
    <h2>Metrics</h2>
    <p><a href="{csv_rel}">Download CSV for all 60 layer skips</a></p>
    <table><thead><tr><th>Skipped Layer</th><th>Pixel MAE</th><th>Pixel RMSE</th><th>Elapsed</th></tr></thead><tbody>{table_rows}</tbody></table>
  </section>
</main>
</body>
</html>
"""
    (WEB_EXP / "exp_07_layer_ablation.html").write_text(clean_html(page), encoding="utf-8")
    write_experiment_index()


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"loading pipeline from {args.model_dir}", flush=True)
    edit_pipe = load_pipeline(
        args.model_dir,
        device=args.device,
        dtype_name=args.dtype,
        progress_bar=False,
        scheduler_name="beta",
        stochastic_sampling=False,
    )
    pipe = make_text_to_image_pipeline(edit_pipe, progress_bar=False)
    transformer = pipe.transformer
    total_layers = len(transformer.transformer_blocks)
    originals: dict[int, Any] = {}
    print(f"loaded {total_layers} layers in {time.perf_counter() - started:.2f}s", flush=True)

    rows: list[dict[str, Any]] = []
    try:
        baseline = run_case(pipe, transformer, originals, layer=None, force=args.force)
        rows.append(baseline)
        baseline_path = Path(baseline["output_path"])
        print(f"baseline ok elapsed={baseline['elapsed_seconds']:.2f}s", flush=True)
        layer_indices = list(range(total_layers))
        if args.layers:
            layer_indices = [int(value) for value in args.layers.split(",")]
        for idx, layer in enumerate(layer_indices, start=1):
            row = run_case(pipe, transformer, originals, layer=layer, force=args.force)
            metrics = image_metrics(Path(row["output_path"]), baseline_path)
            row.update(metrics)
            meta_path = OUT / f"skip_layer_{layer:02d}.json"
            meta_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            print(f"[{idx}/{len(layer_indices)}] layer {layer:02d} mae={row['pixel_mae']:.3f} elapsed={row['elapsed_seconds']:.2f}s", flush=True)
    finally:
        restore_layers(transformer, originals)
        del pipe
        del edit_pipe
        torch.cuda.empty_cache()

    baseline.update({"pixel_mae": 0.0, "pixel_rmse": 0.0, "pixel_max_delta": 0.0})
    write_page(rows)
    print(f"page={WEB_EXP / 'exp_07_layer_ablation.html'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-layer skip ablations for Qwen Image transformer blocks.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--layers", help="Optional comma-separated subset of layer indices.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
