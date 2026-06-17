#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
IMAGES = ROOT / "dataset/images"
META_DIR = ROOT / "dataset/metadata"
CONTACT = ROOT / "dataset/contact_sheet.jpg"
WIDTH = 512
HEIGHT = 512
SEED = 3407

FONT_LATIN = Path("/opt/conda/fonts/DejaVuSans.ttf")
FONT_LATIN_BOLD = Path("/opt/conda/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
FONT_AR = REPO / "assets/fonts/NotoSansArabic-Regular.ttf"
FONT_AR_BOLD = REPO / "assets/fonts/NotoSansArabic-Bold.ttf"

EN_WORDS = [
    "clarity", "design", "signal", "layout", "research", "quality", "poster", "focus", "reader", "modern",
    "balance", "margin", "headline", "system", "visual", "stable", "exact", "glyph", "shape", "spacing",
    "marker", "caption", "studio", "clean", "strong", "bright", "simple", "sharp", "fresh", "direct",
]

AR_WORDS = [
    "وضوح", "تصميم", "إشارة", "تخطيط", "بحث", "جودة", "ملصق", "تركيز", "قارئ", "حديث",
    "توازن", "هامش", "عنوان", "نظام", "بصري", "ثابت", "دقيق", "حرف", "شكل", "مسافة",
    "علامة", "شرح", "استوديو", "نظيف", "قوي", "مشرق", "بسيط", "حاد", "جديد", "مباشر",
]

PALETTES = [
    ("#fbf7ef", "#1f1f1f", "#b59148", "#ffffff"),
    ("#f5f9fb", "#111827", "#287271", "#ffffff"),
    ("#fff7f0", "#221b18", "#b23a48", "#ffffff"),
    ("#f8f8f5", "#171717", "#5b6c2f", "#ffffff"),
]


def shape_arabic(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def wrap_text(text: str, lang: str, draw: ImageDraw.ImageDraw, font_obj: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        rendered = shape_arabic(candidate) if lang == "ar" else candidate
        box = draw.textbbox((0, 0), rendered, font=font_obj)
        if box[2] - box[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    lang: str,
    xy: tuple[int, int],
    max_width: int,
    font_obj: ImageFont.FreeTypeFont,
    fill: str,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in lines:
        rendered = shape_arabic(line) if lang == "ar" else line
        box = draw.textbbox((0, 0), rendered, font=font_obj)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        draw.text((x + (max_width - tw) / 2, y - box[1]), rendered, font=font_obj, fill=fill)
        y += th + line_gap
    return y


def make_phrase(words: list[str], idx: int, count: int) -> str:
    offset = (idx * 3) % len(words)
    selected = [words[(offset + j) % len(words)] for j in range(count)]
    return " ".join(selected)


def sample(idx: int) -> dict[str, str | int]:
    group = idx // 25
    local = idx % 25
    en = make_phrase(EN_WORDS, idx, 3 + (local % 4))
    ar = make_phrase(AR_WORDS, idx, 3 + ((local + 1) % 4))
    if group == 0:
        kind = "english_single"
        prompt = f'Create a clean white poster with the exact English text "{en}".'
    elif group == 1:
        kind = "arabic_single"
        prompt = f'Create a clean white poster with the exact Arabic text "{ar}".'
    elif group == 2:
        kind = "mixed_stacked"
        prompt = f'Create a clean bilingual poster with the exact English text "{en}" and the exact Arabic text "{ar}".'
    else:
        kind = "mixed_structured"
        prompt = f'Create a clean structured bilingual poster with the exact English label "{en}" and the exact Arabic label "{ar}".'
    return {"index": idx, "kind": kind, "english": en, "arabic": ar, "prompt": prompt}


def render(item: dict[str, str | int], path: Path) -> None:
    idx = int(item["index"])
    rng = random.Random(SEED + idx)
    bg, ink, accent, card = PALETTES[idx % len(PALETTES)]
    image = Image.new("RGB", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(image)
    kind = str(item["kind"])
    en = str(item["english"])
    ar = str(item["arabic"])

    margin = 42 + (idx % 5) * 3
    draw.rounded_rectangle((margin, margin, WIDTH - margin, HEIGHT - margin), radius=12, fill=card, outline=accent, width=2)
    if kind == "english_single":
        size = 46 if idx % 3 else 54
        lines = wrap_text(en, "en", draw, font(FONT_LATIN_BOLD, size), WIDTH - 150)
        draw_centered_lines(draw, lines, "en", (75, 200), WIDTH - 150, font(FONT_LATIN_BOLD, size), ink, 9)
    elif kind == "arabic_single":
        size = 44 if idx % 3 else 52
        lines = wrap_text(ar, "ar", draw, font(FONT_AR_BOLD, size), WIDTH - 150)
        draw_centered_lines(draw, lines, "ar", (75, 190), WIDTH - 150, font(FONT_AR_BOLD, size), ink, 9)
    elif kind == "mixed_stacked":
        draw.text((74, 78), "BILINGUAL SAMPLE", font=font(FONT_LATIN_BOLD, 18), fill=accent)
        en_lines = wrap_text(en, "en", draw, font(FONT_LATIN_BOLD, 35), WIDTH - 145)
        y = draw_centered_lines(draw, en_lines, "en", (72, 145), WIDTH - 144, font(FONT_LATIN_BOLD, 35), ink, 6)
        draw.line((86, y + 20, WIDTH - 86, y + 20), fill=accent, width=2)
        ar_lines = wrap_text(ar, "ar", draw, font(FONT_AR_BOLD, 35), WIDTH - 145)
        draw_centered_lines(draw, ar_lines, "ar", (72, y + 52), WIDTH - 144, font(FONT_AR_BOLD, 35), ink, 6)
    else:
        draw.rounded_rectangle((70, 96, WIDTH - 70, 218), radius=8, fill=bg, outline=accent, width=2)
        draw.rounded_rectangle((70, 292, WIDTH - 70, 414), radius=8, fill=bg, outline=accent, width=2)
        draw.text((88, 112), "EN", font=font(FONT_LATIN_BOLD, 18), fill=accent)
        draw.text((WIDTH - 116, 308), shape_arabic("عربي"), font=font(FONT_AR_BOLD, 18), fill=accent)
        en_lines = wrap_text(en, "en", draw, font(FONT_LATIN_BOLD, 28), WIDTH - 170)
        draw_centered_lines(draw, en_lines, "en", (92, 143), WIDTH - 184, font(FONT_LATIN_BOLD, 28), ink, 5)
        ar_lines = wrap_text(ar, "ar", draw, font(FONT_AR_BOLD, 28), WIDTH - 170)
        draw_centered_lines(draw, ar_lines, "ar", (92, 337), WIDTH - 184, font(FONT_AR_BOLD, 28), ink, 5)

    if idx % 7 == 0:
        for _ in range(12):
            x = rng.randint(margin + 8, WIDTH - margin - 8)
            y = rng.randint(margin + 8, HEIGHT - margin - 8)
            draw.ellipse((x, y, x + 1, y + 1), fill="#e9e0d2")
    image.save(path)


def make_contact_sheet(records: list[dict[str, str | int]], out_path: Path) -> None:
    cols = 10
    thumb = 130
    label_h = 24
    rows = math.ceil(len(records) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = font(FONT_LATIN, 11)
    for idx, record in enumerate(records):
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + label_h)
        image = ImageOps.exif_transpose(Image.open(ROOT / str(record["image_path"]))).convert("RGB")
        image.thumbnail((thumb - 8, thumb - 8), Image.Resampling.LANCZOS)
        sheet.paste(image, (x + (thumb - image.width) // 2, y + 4))
        draw.text((x + 4, y + thumb + 2), f"{idx:03d} {record['kind'][:8]}", fill="black", font=label_font)
    sheet.save(out_path, quality=92)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    for idx in range(100):
        item = sample(idx)
        image_path = IMAGES / f"text_train_{idx:03d}.png"
        render(item, image_path)
        record = {
            **item,
            "image_path": str(image_path.relative_to(ROOT)),
            "width": WIDTH,
            "height": HEIGHT,
        }
        records.append(record)
    (META_DIR / "train.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    (META_DIR / "train_summary.json").write_text(
        json.dumps(
            {
                "count": len(records),
                "seed": SEED,
                "groups": {
                    "english_single": 25,
                    "arabic_single": 25,
                    "mixed_stacked": 25,
                    "mixed_structured": 25,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    make_contact_sheet(records, CONTACT)
    print(f"wrote {len(records)} records")
    print(META_DIR / "train.jsonl")
    print(CONTACT)


if __name__ == "__main__":
    main()
