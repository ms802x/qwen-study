#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/qwen_text_failure_next_experiments/exp_06_long_text_stress"
WEB = ROOT / "research_pages/qwen_text_failure"
WEB_EXP = WEB / "experiments"
WEB_ASSETS = WEB / "assets/next_experiments/exp_06_long_text_stress"
API_DEFAULT = "http://127.0.0.1:8001/v1/edit"

WIDTH = 1024
HEIGHT = 1536
STEPS = 40
TRUE_CFG_SCALE = 4.0
MAX_SEQUENCE_LENGTH = 1024
WORD_COUNTS = [10, 50, 100, 500]

FONT_LATIN = Path("/opt/conda/fonts/DejaVuSans.ttf")

ENGLISH_POOL = [
    "clarity", "design", "research", "quality", "layout", "typography", "poster", "signal", "detail", "balance",
    "contrast", "margin", "section", "headline", "caption", "reader", "focus", "system", "image", "language",
    "model", "visual", "spacing", "grid", "rhythm", "shape", "meaning", "context", "structure", "workflow",
    "review", "sample", "output", "prompt", "style", "clean", "modern", "strong", "accurate", "stable",
]

ARABIC_POOL = [
    "تصميم", "واضح", "بحث", "جودة", "تخطيط", "طباعة", "ملصق", "إشارة", "تفاصيل", "توازن",
    "تباين", "هامش", "قسم", "عنوان", "شرح", "قارئ", "تركيز", "نظام", "صورة", "لغة",
    "نموذج", "بصري", "مسافة", "شبكة", "إيقاع", "شكل", "معنى", "سياق", "بنية", "مسار",
    "مراجعة", "عينة", "نتيجة", "موجه", "أسلوب", "نظيف", "حديث", "قوي", "دقيق", "ثابت",
]


def post_json(url: str, payload: dict[str, Any], timeout: int = 900) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def decode_response_image(data: str, path: Path) -> None:
    if "," in data:
        data = data.split(",", 1)[1]
    path.write_bytes(base64.b64decode(data))


def target_text(lang: str, words: int) -> str:
    pool = ARABIC_POOL if lang == "arabic" else ENGLISH_POOL
    return " ".join(pool[idx % len(pool)] for idx in range(words))


def prompt_for(lang: str, words: int, text: str) -> str:
    if lang == "arabic":
        return (
            f"Create a clean portrait white document poster that typesets exactly this {words}-word Arabic text block. "
            "Use black Arabic sans-serif typography, correct right-to-left shaping, clean margins, and simple line wrapping. "
            "Do not translate, do not summarize, do not add decorative letters, and do not add any extra words. "
            f'Exact Arabic text block: "{text}"'
        )
    return (
        f"Create a clean portrait white document poster that typesets exactly this {words}-word English text block. "
        "Use black sans-serif typography, clean margins, and simple line wrapping. "
        "Do not summarize, do not add decorative letters, and do not add any extra words. "
        f'Exact English text block: "{text}"'
    )


def run_job(api: str, lang: str, words: int, force: bool) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    text = target_text(lang, words)
    job_id = f"{lang}_{words}_words"
    image_path = OUT / f"{job_id}.jpg"
    meta_path = OUT / f"{job_id}.json"
    if image_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["skipped_existing"] = True
        return meta

    prompt = prompt_for(lang, words, text)
    payload = {
        "prompt": prompt,
        "images": [],
        "image_roles": [],
        "enhance_prompt": False,
        "negative_prompt": "random text, fake text, misspelled text, watermark, logo, illustration, decoration",
        "seed": 97000 + words + (1000 if lang == "arabic" else 0),
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "output_format": "jpeg",
        "jpeg_quality": 95,
    }
    started = time.perf_counter()
    response = post_json(api, payload)
    wall_seconds = time.perf_counter() - started
    decode_response_image(response["image"], image_path)

    meta = {key: value for key, value in response.items() if key != "image"}
    meta.update({
        "experiment": "exp_06_long_text_stress",
        "job_id": job_id,
        "title": f"{lang.title()} {words} words",
        "lang": lang,
        "target_word_count": words,
        "target_text": text,
        "seed": payload["seed"],
        "width": WIDTH,
        "height": HEIGHT,
        "steps": STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "wall_seconds": wall_seconds,
        "output_path": str(image_path),
        "prompt": prompt,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def copy_asset(src: Path) -> str:
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    dst = WEB_ASSETS / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return f"../assets/next_experiments/exp_06_long_text_stress/{dst.name}"


def make_contact_sheet(rows: list[dict[str, Any]], out_path: Path) -> None:
    cols = 2
    cell_w = 520
    image_h = 650
    label_h = 66
    sheet = Image.new("RGB", (cols * cell_w, math.ceil(len(rows) / cols) * (image_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.truetype(str(FONT_LATIN), 18)
    small_font = ImageFont.truetype(str(FONT_LATIN), 13)
    for idx, row in enumerate(rows):
        x = (idx % cols) * cell_w
        y = (idx // cols) * (image_h + label_h)
        draw.rectangle((x, y, x + cell_w - 1, y + image_h + label_h - 1), outline=(210, 210, 210))
        draw.text((x + 10, y + 8), row["title"], fill=(20, 20, 20), font=label_font)
        draw.text((x + 10, y + 34), f"seed {row['seed']} | {row['width']}x{row['height']}", fill=(90, 90, 90), font=small_font)
        image = ImageOps.exif_transpose(Image.open(row["output_path"])).convert("RGB")
        image.thumbnail((cell_w - 24, image_h - 18), Image.Resampling.LANCZOS)
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + label_h + (image_h - image.height) // 2))
    sheet.save(out_path, quality=94)


def clean_html(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def card(row: dict[str, Any]) -> str:
    image_rel = f"../assets/next_experiments/exp_06_long_text_stress/{Path(row['output_path']).name}"
    klass = "rtl" if row["lang"] == "arabic" else ""
    text = html.escape(row["target_text"])
    return f"""
    <div class="card">
      <img src="{image_rel}" alt="{html.escape(row['title'])}">
      <div class="caption">
        <strong>{html.escape(row['title'])}</strong><br>
        target words: {row['target_word_count']}<br>
        wall: {row.get('wall_seconds', 0):.2f}s
      </div>
      <details>
        <summary>Target text</summary>
        <p class="{klass}">{text}</p>
      </details>
    </div>
    """


def write_experiment_page(rows: list[dict[str, Any]]) -> None:
    WEB_EXP.mkdir(parents=True, exist_ok=True)
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["target_word_count"], row["lang"]))

    csv_path = OUT / "exp_06_long_text_stress_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    sheet_path = OUT / "exp_06_long_text_stress_contact_sheet.jpg"
    make_contact_sheet(rows, sheet_path)

    sheet_rel = copy_asset(sheet_path)
    csv_rel = copy_asset(csv_path)
    for row in rows:
        copy_asset(Path(row["output_path"]))

    prompt_rows = "\n".join(
        f"<tr><td>{html.escape(row['title'])}</td><td>{row['seed']}</td><td>{row['target_word_count']}</td><td>{html.escape(row['prompt'])}</td></tr>"
        for row in rows
    )
    body = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment 06: Long Text Stress Test</title>
  <link rel="stylesheet" href="../shared.css">
</head>
<body>
<main>
  <p class="breadcrumb"><a href="../">Text rendering study</a> / Next experiments</p>
  <h1>Experiment 06: Long Text Stress Test</h1>
  <p class="lead">English and Arabic only. Each prompt asks Qwen to typeset exact 10, 50, 100, and 500-word blocks on a portrait document-style poster. The API is run with max_sequence_length={MAX_SEQUENCE_LENGTH}, which is the highest value accepted by the current server.</p>
  <section>
    <h2>Observed Result</h2>
    <p>The portrait format does not make exact text reliable. English 10 words partially preserves the requested vocabulary but already introduces malformed words; English 50 words keeps some recognizable words while corrupting many others; English 100 and 500 words collapse into page-like microtext rather than exact readable copy. Arabic remains malformed at 10, 50, 100, and 500 words, matching the earlier Arabic findings.</p>
    <p class="note">The 500-word case is intentionally unrealistic for a raster image generator. It is included as a failure-boundary test, not as a practical production target.</p>
  </section>
  <section>
    <h2>Contact Sheet</h2>
    <div class="sheet"><img src="{sheet_rel}" alt="Long text stress test contact sheet"></div>
  </section>
  <section>
    <h2>Individual Outputs</h2>
    <div class="grid">
      {"".join(card(row) for row in rows)}
    </div>
  </section>
  <section>
    <h2>Prompts And Targets</h2>
    <p><a href="{csv_rel}">Download CSV for this experiment</a></p>
    <table><thead><tr><th>Case</th><th>Seed</th><th>Words</th><th>Prompt</th></tr></thead><tbody>{prompt_rows}</tbody></table>
  </section>
</main>
</body>
</html>
"""
    (WEB_EXP / "exp_06_long_text_stress.html").write_text(clean_html(body), encoding="utf-8")
    write_experiment_index()


def write_experiment_index() -> None:
    experiments = [
        ("Experiment 01: Text Size Sweep", "Tests whether requested text size changes success/failure across English, Arabic, and Chinese.", "exp_01_size_sweep.html", "exp_01_size_sweep/exp_01_size_sweep_contact_sheet.jpg"),
        ("Experiment 02: Seed Sweep", "Tests whether success/failure is stable or seed-dependent under the same prompt.", "exp_02_seed_sweep.html", "exp_02_seed_sweep/exp_02_seed_sweep_contact_sheet.jpg"),
        ("Experiment 03: Reference Glyph Copy", "Tests whether already-rendered reference glyphs help Qwen copy or preserve exact text.", "exp_03_reference_glyph_copy.html", "exp_03_reference_glyph_copy/exp_03_reference_glyph_copy_contact_sheet.jpg"),
        ("Experiment 04: Background Complexity", "Tests whether visual context makes exact text rendering worse.", "exp_04_background_complexity.html", "exp_04_background_complexity/exp_04_background_complexity_contact_sheet.jpg"),
        ("Experiment 05: Arabic Prompt Variants", "Tests whether prompt wording alone can repair Arabic text fidelity.", "exp_05_arabic_prompt_variants.html", "exp_05_arabic_prompt_variants/exp_05_arabic_prompt_variants_contact_sheet.jpg"),
        ("Experiment 06: Long Text Stress Test", "Tests English and Arabic at 10, 50, 100, and 500 target words.", "exp_06_long_text_stress.html", "exp_06_long_text_stress/exp_06_long_text_stress_contact_sheet.jpg"),
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


def run(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for words in WORD_COUNTS:
        for lang in ["english", "arabic"]:
            print(f"{lang} {words} words", flush=True)
            row = run_job(args.api, lang, words, args.force)
            rows.append(row)
            print(f"  ok elapsed={row.get('elapsed_seconds')} wall={row.get('wall_seconds'):.2f}s", flush=True)
    write_experiment_page(rows)
    print(f"page={WEB_EXP / 'exp_06_long_text_stress.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen long-text stress tests and generate an HTML page.")
    parser.add_argument("--api", default=API_DEFAULT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
