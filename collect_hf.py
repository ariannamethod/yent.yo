#!/usr/bin/env python3
"""Collect training images from HuggingFace datasets + web.

Sources:
  - HuggingFace: art datasets with style labels
  - WikiArt: public domain art by style
  - Direct URLs: curated lists

Usage:
  python3 collect_hf.py <style> [--max N] [--size S]
"""

import os
import sys
import hashlib
from pathlib import Path
from PIL import Image
from io import BytesIO
from urllib.request import urlopen, Request
import time
import json

DATA_DIR = Path(__file__).parent / "data"

# Curated image URLs for each style (public domain / CC)
STYLE_URLS = {
    "caricature": {
        "description": "Political caricature + soc-art + satirical illustration",
        "search_terms": [
            # Daumier
            "honore daumier lithograph",
            "thomas nast cartoon",
            "james gillray caricature",
            # Soc-art
            "komar melamid soc art",
            "soviet satirical poster",
            "political caricature illustration",
        ],
    },
    "graffiti": {
        "description": "Street art, spray paint, murals, tags",
        "search_terms": [
            "banksy street art",
            "graffiti mural urban art",
            "spray paint street art",
            "urban graffiti wall",
        ],
    },
    "propaganda": {
        "description": "Soviet constructivism, propaganda posters, revolutionary art",
        "search_terms": [
            "soviet propaganda poster",
            "russian constructivism poster",
            "rodchenko poster design",
            "revolutionary propaganda art",
            "workers unite poster",
        ],
    },
    "pixel": {
        "description": "Pixel art, 8-bit, retro game art",
        "search_terms": [
            "pixel art character sprite",
            "8 bit game art",
            "retro pixel art scene",
            "pixel art landscape",
        ],
    },
}


def download_from_hf_dataset(style, max_images=50, size=512):
    """Try to download from HuggingFace art datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not installed, skipping HF")
        return 0

    out_dir = DATA_DIR / style
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*")))

    count = 0

    # Try wikiart dataset
    if style in ("caricature", "propaganda"):
        print(f"  Trying HuggingFace wikiart dataset...")
        try:
            # huggan/wikiart has style labels
            ds = load_dataset("huggan/wikiart", split="train", streaming=True)
            style_map = {
                "caricature": ["Expressionism", "Naive_Art_Primitivism", "Symbolism"],
                "propaganda": ["Socialist_Realism", "Constructivism", "Art_Nouveau_Modern"],
            }
            target_styles = style_map.get(style, [])

            for item in ds:
                if count >= max_images:
                    break
                if item.get("style") in target_styles or item.get("genre") in target_styles:
                    img = item["image"]
                    if isinstance(img, Image.Image):
                        img = img.convert("RGB").resize((size, size), Image.LANCZOS)
                        h = hashlib.md5(img.tobytes()[:1000]).hexdigest()[:12]
                        fname = f"hf_{h}.png"
                        if not (out_dir / fname).exists():
                            img.save(out_dir / fname)
                            count += 1
                            if count % 10 == 0:
                                print(f"    {count} images...")
        except Exception as e:
            print(f"    wikiart failed: {e}")

    # Try pixel art datasets
    if style == "pixel":
        print(f"  Trying HuggingFace pixel art datasets...")
        for ds_name in ["Norod78/PixelArt-SpriteSheet-and-Animation", "jainr3/diffusiondb-pixelart"]:
            try:
                ds = load_dataset(ds_name, split="train", streaming=True)
                for item in ds:
                    if count >= max_images:
                        break
                    img = item.get("image")
                    if img and isinstance(img, Image.Image):
                        img = img.convert("RGB").resize((size, size), Image.LANCZOS)
                        h = hashlib.md5(img.tobytes()[:1000]).hexdigest()[:12]
                        fname = f"hf_{h}.png"
                        if not (out_dir / fname).exists():
                            img.save(out_dir / fname)
                            count += 1
            except Exception as e:
                print(f"    {ds_name} failed: {e}")

    # Generic: try to get images from DiffusionDB filtered by prompt keywords
    if count < max_images:
        print(f"  Trying DiffusionDB with keyword filter...")
        try:
            ds = load_dataset("poloclub/diffusiondb", "2m_first_1k", split="train", streaming=True)
            keywords = STYLE_URLS[style]["search_terms"]
            for item in ds:
                if count >= max_images:
                    break
                prompt = item.get("prompt", "").lower()
                if any(kw.lower() in prompt for kw in keywords[:3]):
                    img = item.get("image")
                    if img and isinstance(img, Image.Image):
                        img = img.convert("RGB").resize((size, size), Image.LANCZOS)
                        h = hashlib.md5(img.tobytes()[:1000]).hexdigest()[:12]
                        fname = f"db_{h}.png"
                        if not (out_dir / fname).exists():
                            img.save(out_dir / fname)
                            count += 1
                            if count % 5 == 0:
                                print(f"    {count} images from DiffusionDB...")
        except Exception as e:
            print(f"    DiffusionDB failed: {e}")

    return count


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 collect_hf.py <style> [--max N] [--size S]")
        print(f"Styles: {', '.join(STYLE_URLS.keys())}")
        sys.exit(1)

    style = sys.argv[1]
    max_images = 50
    size = 512

    for i, arg in enumerate(sys.argv):
        if arg == "--max" and i + 1 < len(sys.argv):
            max_images = int(sys.argv[i + 1])
        if arg == "--size" and i + 1 < len(sys.argv):
            size = int(sys.argv[i + 1])

    print(f"=== Collecting {style} (max {max_images}, {size}px) ===")
    out_dir = DATA_DIR / style
    existing = len(list(out_dir.glob("*"))) if out_dir.exists() else 0
    print(f"  Existing: {existing} images")

    count = download_from_hf_dataset(style, max_images, size)

    total = len(list(out_dir.glob("*"))) if out_dir.exists() else 0
    print(f"\n  New: {count}, Total: {total}")
    print(f"  Dir: {out_dir}")


if __name__ == "__main__":
    main()
