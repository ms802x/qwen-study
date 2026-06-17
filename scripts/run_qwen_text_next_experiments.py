#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arabic_reshaper
import pandas as pd
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/qwen_text_failure_next_experiments"
WEB = ROOT / "research_pages/qwen_text_failure"
WEB_EXP = WEB / "experiments"
WEB_ASSETS = WEB / "assets/next_experiments"
API_DEFAULT = "http://127.0.0.1:8001/v1/edit"

FONT_LATIN = Path("/opt/conda/fonts/DejaVuSans.ttf")
FONT_AR_REG = ROOT / "assets/fonts/NotoSansArabic-Regular.ttf"
FONT_AR_BOLD = ROOT / "assets/fonts/NotoSansArabic-Bold.ttf"
FONT_CJK = ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"

WIDTH = 768
HEIGHT = 768
STEPS = 40
TRUE_CFG_SCALE = 4.0

LANGS = {
    "english": {
        "text": "AI PIZZA",
        "class": "",
        "label": "English",
    },
    "arabic": {
        "text": "بيتزا الذكاء الاصطناعي",
        "class": "rtl",
        "label": "Arabic",
    },
    "chinese": {
        "text": "人工智能披萨",
        "class": "cjk",
        "label": "Chinese",
    },
}


@dataclass(frozen=True)
class Job:
    experiment: str
    job_id: str
    title: str
    lang: str
    prompt: str
    seed: int
    images: list[Path]
    image_roles: list[str]
    notes: str = ""
    negative_prompt: str = "extra text, misspelled text, pseudo text, watermark, logo, decorative letters"


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def decode_response_image(data: str, path: Path) -> None:
    if "," in data:
        data = data.split(",", 1)[1]
    path.write_bytes(base64.b64decode(data))


def post_json(url: str, payload: dict[str, Any], timeout: int = 900) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def shape_arabic(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def font_for(lang: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if lang == "arabic":
        path = FONT_AR_BOLD if bold and FONT_AR_BOLD.exists() else FONT_AR_REG
    elif lang == "chinese":
        path = FONT_CJK
    else:
        path = FONT_LATIN
    return ImageFont.truetype(str(path), size)


def fit_text(text: str, lang: str, draw: ImageDraw.ImageDraw, requested_size: int, max_width: int, max_height: int):
    rendered = shape_arabic(text) if lang == "arabic" else text
    for size in range(requested_size, 10, -1):
        font = font_for(lang, size, bold=True)
        box = draw.textbbox((0, 0), rendered, font=font)
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width <= max_width and height <= max_height:
            return rendered, font, size, box
    font = font_for(lang, 11, bold=True)
    box = draw.textbbox((0, 0), rendered, font=font)
    return rendered, font, 11, box


def make_text_card(path: Path, text: str, lang: str, requested_size: int, *, kind: str = "cream") -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    bg = "#fbf7ed" if kind == "cream" else "white"
    image = Image.new("RGB", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 170, WIDTH - 70, HEIGHT - 170), radius=28, fill="white", outline="#b59148", width=4)
    rendered, font, actual_size, bbox = fit_text(
        text,
        lang,
        draw,
        requested_size,
        max_width=WIDTH - 190,
        max_height=HEIGHT - 360,
    )
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((WIDTH - tw) / 2, (HEIGHT - th) / 2 - bbox[1]), rendered, fill="#202020", font=font)
    image.save(path)
    return {"requested_font_size": requested_size, "actual_font_size": actual_size}


def make_blank_card(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#fbf7ed")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 170, WIDTH - 70, HEIGHT - 170), radius=28, fill="white", outline="#b59148", width=4)
    image.save(path)


def experiment_jobs() -> list[Job]:
    jobs: list[Job] = []
    controls = OUT / "controls"
    blank = controls / "blank_card.png"
    make_blank_card(blank)

    # 1. Text size sweep.
    sizes = [
        ("large", "very large", 92011),
        ("medium", "medium-sized", 92012),
        ("small", "small but still readable", 92013),
    ]
    for lang, info in LANGS.items():
        for size_name, size_words, seed in sizes:
            text = info["text"]
            jobs.append(Job(
                experiment="exp_01_size_sweep",
                job_id=f"t2i_{lang}_{size_name}",
                title=f"{info['label']} {size_name}",
                lang=lang,
                seed=seed,
                images=[],
                image_roles=[],
                prompt=(
                    f'Create a plain white square poster with only the exact text "{text}" centered '
                    f"in {size_words} black clean sans-serif letters. No other text, no logo, "
                    "no watermark, no decoration."
                ),
                notes="Tests how requested apparent text size changes success/failure.",
            ))

    # 2. Seed sweep.
    for lang, info in LANGS.items():
        for seed in [93101, 93102, 93103]:
            text = info["text"]
            jobs.append(Job(
                experiment="exp_02_seed_sweep",
                job_id=f"seed_{seed}_{lang}",
                title=f"{info['label']} seed {seed}",
                lang=lang,
                seed=seed,
                images=[],
                image_roles=[],
                prompt=(
                    f'Create a plain white square poster with only the exact text "{text}" centered '
                    "in very large black clean sans-serif letters. No other text, no logo, no watermark, no decoration."
                ),
                notes="Tests whether success/failure is stable or seed-dependent.",
            ))

    # 3. Reference glyph copy and preserve.
    for lang, info in LANGS.items():
        text = info["text"]
        ref = controls / f"{lang}_glyph_reference.png"
        make_text_card(ref, text, lang, 96)
        jobs.append(Job(
            experiment="exp_03_reference_glyph_copy",
            job_id=f"copy_ref_to_blank_{lang}",
            title=f"{info['label']} copy glyph reference to blank",
            lang=lang,
            seed=94101,
            images=[blank, ref],
            image_roles=[
                "Target blank cream card to edit.",
                "Glyph reference containing the exact text to copy.",
            ],
            prompt=(
                f'Edit Image 1 by writing exactly "{text}" centered on the card. '
                "Use Image 2 only as the visual glyph reference for the exact text shapes and order. "
                "Preserve the blank cream card and border. No other text, no logo, no watermark."
            ),
            notes="Tests whether already-rendered reference glyphs help Qwen copy exact text.",
        ))
        jobs.append(Job(
            experiment="exp_03_reference_glyph_copy",
            job_id=f"preserve_existing_text_{lang}",
            title=f"{info['label']} preserve existing text",
            lang=lang,
            seed=94102,
            images=[ref],
            image_roles=["Source card already containing the exact target text."],
            prompt=(
                f'Preserve the exact existing text "{text}" and make the card slightly cleaner and sharper. '
                "Do not change any letters, language, order, or spelling. No other text, no logo, no watermark."
            ),
            notes="Tests whether Qwen can preserve exact text that already exists in the source image.",
        ))

    # 4. Background complexity.
    backgrounds = [
        ("white_sign", "a perfectly plain white square poster"),
        ("paper_poster", "a subtle cream paper poster with a thin gold border"),
        ("photo_sign", "a realistic restaurant wall sign on a softly lit neutral wall"),
    ]
    for lang, info in LANGS.items():
        for idx, (bg_name, bg_text) in enumerate(backgrounds):
            text = info["text"]
            jobs.append(Job(
                experiment="exp_04_background_complexity",
                job_id=f"{bg_name}_{lang}",
                title=f"{info['label']} on {bg_name.replace('_', ' ')}",
                lang=lang,
                seed=95100 + idx,
                images=[],
                image_roles=[],
                prompt=(
                    f'Create {bg_text} with only the exact text "{text}" centered in large black clean sans-serif letters. '
                    "No other text, no logo, no watermark."
                ),
                notes="Tests whether added visual context makes text fidelity worse.",
            ))

    # 5. Arabic prompt variants.
    arabic = LANGS["arabic"]["text"]
    prompt_variants = [
        (
            "quoted_exact",
            f'Create a plain white square poster with only this exact Arabic text centered: "{arabic}". No other text.',
        ),
        (
            "unicode_literal",
            f'Render the exact Unicode Arabic string "{arabic}" as text. Keep the characters and order unchanged. Plain white square poster.',
        ),
        (
            "rtl_instruction",
            f'Create a plain white square poster. Write exactly "{arabic}" in Arabic, right-to-left, centered, large black letters. Do not translate or substitute words.',
        ),
        (
            "arabic_language_prompt",
            f'صمّم ملصقاً أبيض بسيطاً يحتوي فقط على النص التالي في الوسط: "{arabic}". لا تضف أي نص آخر.',
        ),
        (
            "token_spaced",
            'Create a plain white square poster with only the exact Arabic phrase "بيتزا الذكاء الاصطناعي". The words are: "بيتزا" then "الذكاء" then "الاصطناعي".',
        ),
    ]
    for idx, (variant, prompt) in enumerate(prompt_variants):
        jobs.append(Job(
            experiment="exp_05_arabic_prompt_variants",
            job_id=f"arabic_{variant}",
            title=f"Arabic prompt variant: {variant.replace('_', ' ')}",
            lang="arabic",
            seed=96100 + idx,
            images=[],
            image_roles=[],
            prompt=prompt + " No logo, no watermark, no decoration.",
            notes="Tests whether prompt wording alone can repair Arabic text fidelity.",
        ))

    return jobs


def run_job(api: str, job: Job, force: bool) -> dict[str, Any]:
    exp_dir = OUT / job.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    image_path = exp_dir / f"{job.job_id}.jpg"
    meta_path = exp_dir / f"{job.job_id}.json"
    if image_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        meta["skipped_existing"] = True
        return meta

    payload = {
        "prompt": job.prompt,
        "images": [b64_image(path) for path in job.images],
        "image_roles": job.image_roles,
        "enhance_prompt": False,
        "negative_prompt": job.negative_prompt,
        "seed": job.seed,
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "max_sequence_length": 512,
        "output_format": "jpeg",
        "jpeg_quality": 95,
    }
    started = time.perf_counter()
    response = post_json(api, payload)
    wall_seconds = time.perf_counter() - started
    decode_response_image(response["image"], image_path)
    meta = {k: v for k, v in response.items() if k != "image"}
    meta.update({
        "experiment": job.experiment,
        "job_id": job.job_id,
        "title": job.title,
        "lang": job.lang,
        "target_text": LANGS[job.lang]["text"],
        "seed": job.seed,
        "width": WIDTH,
        "height": HEIGHT,
        "steps": STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "wall_seconds": wall_seconds,
        "output_path": str(image_path),
        "prompt": job.prompt,
        "image_inputs": [str(path) for path in job.images],
        "image_roles": job.image_roles,
        "notes": job.notes,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def make_contact_sheet(rows: list[dict[str, Any]], out_path: Path, *, cols: int = 3) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cell_w = 360
    image_h = 330
    label_h = 68
    rows_count = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows_count * (image_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.truetype(str(FONT_LATIN), 17)
    small_font = ImageFont.truetype(str(FONT_LATIN), 13)
    for idx, row in enumerate(rows):
        x = (idx % cols) * cell_w
        y = (idx // cols) * (image_h + label_h)
        draw.rectangle((x, y, x + cell_w - 1, y + image_h + label_h - 1), outline=(210, 210, 210))
        draw.text((x + 8, y + 8), str(row["title"])[:38], fill=(20, 20, 20), font=label_font)
        draw.text((x + 8, y + 33), f"seed {row['seed']} | {row['lang']}", fill=(90, 90, 90), font=small_font)
        image = ImageOps.exif_transpose(Image.open(row["output_path"])).convert("RGB")
        image.thumbnail((cell_w - 22, image_h - 18), Image.Resampling.LANCZOS)
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + label_h + (image_h - image.height) // 2))
    sheet.save(out_path, quality=94)


def copy_asset(src: Path, asset_dir: Path) -> str:
    asset_dir.mkdir(parents=True, exist_ok=True)
    dst = asset_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return f"../assets/next_experiments/{asset_dir.name}/{dst.name}"


def clean_html(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def html_page(title: str, body: str) -> str:
    return clean_html(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="../shared.css">
</head>
<body>
<main>
  <p class="breadcrumb"><a href="../">Text rendering study</a> / Next experiments</p>
  {body}
</main>
</body>
</html>
""")


def write_shared_css() -> None:
    (WEB / "shared.css").write_text("""@font-face{font-family:NotoArabicLocal;src:url('../../assets/fonts/NotoSansArabic-Regular.ttf')}@font-face{font-family:NotoCJKLocal;src:url('../../assets/fonts/NotoSansCJKsc-Regular.otf')}*{box-sizing:border-box}body{margin:0;background:#f6f1e7;color:#23211c;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.58}main{width:min(1180px,calc(100vw - 28px));margin:0 auto;padding:28px 0 64px}section{background:#fffdf8;border:1px solid #d8c8a4;border-radius:8px;padding:22px;margin:18px 0}h1{font-size:clamp(32px,5vw,54px);line-height:1.05;margin:0 0 12px}h2{font-size:27px;margin:0 0 12px}.lead{font-size:18px;max-width:900px;color:#6f685b}.breadcrumb{font-size:14px;color:#6f685b}a{color:#225f83}.note{border-left:4px solid #b53a31;background:#fff3ee;padding:13px 15px;color:#23211c}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{background:white;border:1px solid #d8c8a4;border-radius:8px;padding:10px}.card img{width:100%;display:block;border-radius:4px}.caption{font-size:14px;color:#6f685b;margin-top:8px}.rtl{direction:rtl;font-family:NotoArabicLocal,system-ui,sans-serif}.cjk{font-family:NotoCJKLocal,system-ui,sans-serif}table{width:100%;border-collapse:collapse;margin-top:12px;background:white;font-size:14px}td,th{border:1px solid #e2d6bd;padding:8px 10px;text-align:left;vertical-align:top}th{background:#f1e8d7}.sheet{background:white;border:1px solid #d8c8a4;border-radius:8px;padding:10px}.sheet img{width:100%;display:block}@media(max-width:900px){.grid{grid-template-columns:1fr}section{padding:16px}}""", encoding="utf-8")


def row_card(row: dict[str, Any], asset_rel: str) -> str:
    target = LANGS[row["lang"]]["text"]
    klass = LANGS[row["lang"]]["class"]
    return f"""
    <div class="card">
      <img src="{asset_rel}" alt="{html.escape(row['title'])}">
      <div class="caption"><strong>{html.escape(row['title'])}</strong><br>
      target: <span class="{klass}">{html.escape(target)}</span><br>
      seed {row['seed']}</div>
    </div>
    """


def write_experiment_pages(all_rows: list[dict[str, Any]]) -> None:
    WEB_EXP.mkdir(parents=True, exist_ok=True)
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    write_shared_css()
    exp_titles = {
        "exp_01_size_sweep": "Experiment 01: Text Size Sweep",
        "exp_02_seed_sweep": "Experiment 02: Seed Sweep",
        "exp_03_reference_glyph_copy": "Experiment 03: Reference Glyph Copy",
        "exp_04_background_complexity": "Experiment 04: Background Complexity",
        "exp_05_arabic_prompt_variants": "Experiment 05: Arabic Prompt Variants",
    }
    summaries = {
        "exp_01_size_sweep": "Tests whether requested text size changes success/failure across English, Arabic, and Chinese.",
        "exp_02_seed_sweep": "Tests whether success/failure is stable or seed-dependent under the same prompt.",
        "exp_03_reference_glyph_copy": "Tests whether already-rendered reference glyphs help Qwen copy or preserve exact text.",
        "exp_04_background_complexity": "Tests whether visual context makes exact text rendering worse.",
        "exp_05_arabic_prompt_variants": "Tests whether prompt wording alone can repair Arabic text fidelity.",
    }
    observations = {
        "exp_01_size_sweep": (
            "English stays correct at large, medium, and small sizes. Arabic is malformed even at large size, "
            "so this run does not support a simple low-resolution-only explanation. Chinese remains much stronger "
            "than Arabic under the same prompt template."
        ),
        "exp_02_seed_sweep": (
            "English is stable across all three seeds. Arabic fails across all three seeds with different malformed "
            "glyph patterns, which points to a systematic prompt-to-glyph weakness rather than a single unlucky sample. "
            "Chinese remains comparatively stable."
        ),
        "exp_03_reference_glyph_copy": (
            "This is the key positive control. When the exact Arabic glyphs are provided visually, Qwen can preserve "
            "and copy the Arabic text cleanly. That argues against the VAE or image latent path being fundamentally "
            "unable to represent Arabic; the weak point is more likely text-conditioned glyph synthesis."
        ),
        "exp_04_background_complexity": (
            "English and Chinese stay readable across plain sign, paper poster, and photo sign contexts. Arabic remains "
            "malformed in all three. Visual complexity can hurt text in general, but it is not the main cause of the "
            "Arabic failure seen here."
        ),
        "exp_05_arabic_prompt_variants": (
            "Quoted text, Unicode wording, explicit RTL instruction, an Arabic-language prompt, and word-level spelling "
            "instructions all fail to recover the target phrase. Prompt phrasing alone is therefore not a reliable fix "
            "for this Arabic case."
        ),
    }
    index_links = []
    for exp, title in exp_titles.items():
        rows = [row for row in all_rows if row["experiment"] == exp]
        if not rows:
            continue
        exp_out = OUT / exp
        sheet = exp_out / f"{exp}_contact_sheet.jpg"
        make_contact_sheet(rows, sheet)
        asset_dir = WEB_ASSETS / exp
        sheet_rel = copy_asset(sheet, asset_dir)
        for row in rows:
            copy_asset(Path(row["output_path"]), asset_dir)
        csv_path = exp_out / f"{exp}_results.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        csv_rel = copy_asset(csv_path, asset_dir)

        cards = "\n".join(
            row_card(row, f"../assets/next_experiments/{exp}/{Path(row['output_path']).name}")
            for row in rows
        )
        prompt_table = "".join(
            f"<tr><td>{html.escape(row['title'])}</td><td>{row['seed']}</td><td>{html.escape(row['prompt'])}</td></tr>"
            for row in rows
        )
        body = f"""
        <h1>{html.escape(title)}</h1>
        <p class="lead">{html.escape(summaries[exp])}</p>
        <section>
          <h2>Observed Result</h2>
          <p>{html.escape(observations[exp])}</p>
        </section>
        <section>
          <h2>Contact Sheet</h2>
          <div class="sheet"><img src="{sheet_rel}" alt="{html.escape(title)} contact sheet"></div>
        </section>
        <section>
          <h2>Individual Outputs</h2>
          <div class="grid">{cards}</div>
        </section>
        <section>
          <h2>Prompts</h2>
          <p><a href="{csv_rel}">Download CSV for this experiment</a></p>
          <table><thead><tr><th>Case</th><th>Seed</th><th>Prompt</th></tr></thead><tbody>{prompt_table}</tbody></table>
        </section>
        <section>
          <h2>Review Note</h2>
          <p class="note">These observations come from visual review of the rendered outputs, not OCR. Exact automated scoring should be added later with a text-aware evaluator.</p>
        </section>
        """
        page_name = f"{exp}.html"
        (WEB_EXP / page_name).write_text(html_page(title, body), encoding="utf-8")
        index_links.append((title, summaries[exp], page_name, sheet_rel))

    links_html = "\n".join(
        f"""<div class="card"><img src="{sheet_rel}" alt="{html.escape(title)}"><div class="caption"><a href="{page}"><strong>{html.escape(title)}</strong></a><br>{html.escape(summary)}</div></div>"""
        for title, summary, page, sheet_rel in index_links
    )
    index = html_page(
        "Qwen Text Rendering Next Experiments",
        f"""
        <h1>Qwen Text Rendering: Next Experiments</h1>
        <p class="lead">Each page is a separate investigation with its own contact sheet, individual images, prompts, and CSV.</p>
        <section><h2>Experiment Pages</h2><div class="grid">{links_html}</div></section>
        <section><h2>Public Key For GitHub</h2><p>The public key is not embedded here. Use the key printed in the terminal output for GitHub access.</p></section>
        """,
    )
    (WEB_EXP / "index.html").write_text(index, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = experiment_jobs()
    if args.only:
        wanted = set(args.only.split(","))
        jobs = [job for job in jobs if job.experiment in wanted]
    rows: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {job.experiment}/{job.job_id}", flush=True)
        try:
            meta = run_job(args.api, job, args.force)
            rows.append(meta)
            print(f"  ok elapsed={meta.get('elapsed_seconds')} wall={meta.get('wall_seconds')}", flush=True)
        except Exception as exc:
            rows.append({
                "experiment": job.experiment,
                "job_id": job.job_id,
                "title": job.title,
                "lang": job.lang,
                "target_text": LANGS[job.lang]["text"],
                "seed": job.seed,
                "status": "failed",
                "error": repr(exc),
                "prompt": job.prompt,
                "output_path": "",
                "notes": job.notes,
            })
            print(f"  failed {exc!r}", flush=True)
    summary = OUT / "all_next_experiments_results.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    write_experiment_pages([row for row in rows if row.get("output_path")])
    print(f"summary={summary}")
    print(f"pages={WEB_EXP / 'index.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next Qwen text-rendering experiments and generate readable pages.")
    parser.add_argument("--api", default=API_DEFAULT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", help="Comma-separated experiment ids to run.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
