#!/usr/bin/env python3
"""Build a fixed-ten visual diagnostic grid for the triplet-fuser probe."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HEADERS = ("base clean", "fuser-500 clean", "base primary", "fuser-500 primary")


def open_image(folder: Path, image_id: str, size: int) -> Image.Image:
    return Image.open(folder / "fake" / f"{image_id}.png").convert("RGB").resize((size, size))


def caption(folder: Path, image_id: str) -> str:
    path = folder / "sample_metadata" / f"{image_id}.json"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return str(data.get("caption", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-clean", type=Path, required=True)
    parser.add_argument("--fuser-clean", type=Path, required=True)
    parser.add_argument("--base-primary", type=Path, required=True)
    parser.add_argument("--fuser-primary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    folders = (args.base_clean, args.fuser_clean, args.base_primary, args.fuser_primary)
    ids = sorted(path.stem for path in args.base_clean.joinpath("fake").glob("*.png"))
    cell, header, caption_height = 256, 26, 44
    canvas = Image.new("RGB", (cell * 4, header + len(ids) * (cell + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col, text in enumerate(HEADERS):
        draw.text((col * cell + 4, 6), text, fill="black", font=font)
    for row, image_id in enumerate(ids):
        y = header + row * (cell + caption_height)
        clean_caption = caption(args.base_clean, image_id)[:120]
        draw.text((2, y), f"{image_id}: {clean_caption}", fill="black", font=font)
        for col, folder in enumerate(folders):
            canvas.paste(open_image(folder, image_id, cell), (col * cell, y + caption_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
