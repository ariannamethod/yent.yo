#!/usr/bin/env python3
"""
Vitriol — Dataset Collector
Downloads caricature art from Wikimedia Commons.
Artists: Honoré Daumier, Thomas Nast, James Gillray
"""

import os
import json
import time
import hashlib
import argparse
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote
from PIL import Image
from io import BytesIO

DATA_DIR = Path(__file__).parent / "data"

CATEGORIES = {
    "daumier": [
        "Category:Prints by Honoré Daumier",
        "Category:Caricatures by Honoré Daumier",
        "Category:Lithographs by Honoré Daumier",
    ],
    "nast": [
        "Category:Thomas Nast",
        "Category:Cartoons by Thomas Nast",
    ],
    "gillray": [
        "Category:James Gillray",
        "Category:Prints by James Gillray",
    ],
}

HEADERS = {"User-Agent": "Vitriol/1.0 (ariannamethod; dataset collection)"}
IMG_SIZE = 128
THUMB_WIDTH = 512  # download at this width, then resize


def fetch_json(url):
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get_category_images(category, limit=200):
    """Fetch image URLs from a Wikimedia Commons category."""
    base = "https://commons.wikimedia.org/w/api.php"
    images = []
    cont = None

    while len(images) < limit:
        params = (
            f"?action=query&generator=categorymembers"
            f"&gcmtitle={quote(category)}"
            f"&gcmtype=file&gcmlimit=50"
            f"&prop=imageinfo&iiprop=url|mime"
            f"&iiurlwidth={THUMB_WIDTH}"
            f"&format=json"
        )
        if cont:
            params += f"&gcmcontinue={cont}"

        url = base + params
        data = fetch_json(url)

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            mime = info.get("mime", "")
            if mime.startswith("image/") and mime != "image/svg+xml":
                thumb = info.get("thumburl") or info.get("url")
                if thumb:
                    images.append(thumb)

        cont = data.get("continue", {}).get("gcmcontinue")
        if not cont:
            break
        time.sleep(0.5)  # be polite

    return images[:limit]


def download_and_resize(url, out_path, size=IMG_SIZE):
    """Download image, resize to square, save as PNG."""
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=30) as resp:
            img_data = resp.read()
        img = Image.open(BytesIO(img_data)).convert("RGB")
        # center crop to square
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        img = img.resize((size, size), Image.LANCZOS)
        img.save(out_path, "PNG")
        return True
    except Exception as e:
        print(f"  SKIP {url[:60]}... ({e})")
        return False


def collect(artists=None, per_category=200):
    DATA_DIR.mkdir(exist_ok=True)
    if artists is None:
        artists = list(CATEGORIES.keys())

    total = 0
    for artist in artists:
        cats = CATEGORIES.get(artist, [])
        print(f"\n=== {artist.upper()} ({len(cats)} categories) ===")
        artist_dir = DATA_DIR / artist
        artist_dir.mkdir(exist_ok=True)

        for cat in cats:
            print(f"\n  {cat}")
            urls = get_category_images(cat, limit=per_category)
            print(f"  Found {len(urls)} images")

            for i, url in enumerate(urls):
                # deterministic filename from URL
                name = hashlib.md5(url.encode()).hexdigest()[:12] + ".png"
                out = artist_dir / name
                if out.exists():
                    continue
                if download_and_resize(url, out):
                    total += 1
                    if (i + 1) % 10 == 0:
                        print(f"  Downloaded {i+1}/{len(urls)}")
                time.sleep(0.3)  # rate limit

    # merge all into flat data/ for training
    all_dir = DATA_DIR / "all"
    all_dir.mkdir(exist_ok=True)
    for artist in artists:
        artist_dir = DATA_DIR / artist
        if artist_dir.exists():
            for f in artist_dir.glob("*.png"):
                dest = all_dir / f"{artist}_{f.name}"
                if not dest.exists():
                    os.link(f, dest)

    final_count = len(list(all_dir.glob("*.png")))
    print(f"\n{'='*40}")
    print(f"Total images collected: {total}")
    print(f"Combined dataset: {final_count} images in {all_dir}")
    print(f"Ready for training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vitriol dataset collector")
    parser.add_argument("--artists", nargs="+", choices=list(CATEGORIES.keys()),
                        help="Which artists to collect (default: all)")
    parser.add_argument("--per-category", type=int, default=200,
                        help="Max images per category (default: 200)")
    parser.add_argument("--size", type=int, default=128,
                        help="Output image size (default: 128)")
    args = parser.parse_args()

    if args.size != IMG_SIZE:
        IMG_SIZE = args.size

    collect(artists=args.artists, per_category=args.per_category)
