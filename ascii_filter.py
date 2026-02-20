#!/usr/bin/env python3
"""Colored ASCII art filter for yent.yo images.

Converts PNG/JPG to colored ASCII art using ANSI truecolor escape codes.
Can be used as standalone CLI or imported as module.

Usage:
  python3 ascii_filter.py <image> [width] [--no-color] [--charset <set>]
  python3 ascii_filter.py ort_cat_25step_gpu.png 120
  python3 ascii_filter.py lora_cat_graffiti.png 80 --charset blocks

Charsets:
  techno    — " .·:;=+×*#%@█" (13 levels, yent.yo default)
  standard  — " .:-=+*#%@" (classic)
  detailed  — " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
  blocks    — " ░▒▓█" (unicode blocks)
  minimal   — " .oO@" (5 levels)
"""

import sys
import os
from PIL import Image


# ASCII character sets ordered by "darkness" (light → dark)
CHARSETS = {
    "techno": " .'·:;~=+×*#%@▓█",  # 16 levels — yent.yo default
    "punk": " ·•×#█",  # 6 levels — legacy
    "standard": " .:-=+*#%@",
    "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "blocks": " ░▒▓█",
    "minimal": " .oO@",
}


def image_to_ascii(image_path, width=100, charset="techno", color=True, bg_color=False):
    """Convert image to colored ASCII string.

    Args:
        image_path: Path to image file
        width: Output width in characters
        charset: Character set name or custom string
        color: Use ANSI truecolor (24-bit) escape codes
        bg_color: Also set background color (heavier but richer)

    Returns:
        String with ANSI escape codes (if color=True) or plain ASCII
    """
    img = Image.open(image_path).convert("RGB")

    # Calculate height preserving aspect ratio
    # Terminal chars are ~2x taller than wide, so halve the height
    aspect = img.height / img.width
    height = int(width * aspect * 0.5)

    # Resize
    img = img.resize((width, height), Image.LANCZOS)
    pixels = img.load()

    # Get charset
    chars = CHARSETS.get(charset, charset)
    num_chars = len(chars)

    lines = []
    for y in range(height):
        line = []
        for x in range(width):
            r, g, b = pixels[x, y]

            # Brightness (perceived luminance)
            brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

            # Map to character
            idx = int(brightness * (num_chars - 1))
            idx = max(0, min(idx, num_chars - 1))
            char = chars[idx]

            if color:
                if bg_color:
                    # Foreground + background (richer look)
                    # Darken bg slightly
                    br, bg_g, bb = max(0, r//3), max(0, g//3), max(0, b//3)
                    line.append(f"\033[38;2;{r};{g};{b}m\033[48;2;{br};{bg_g};{bb}m{char}\033[0m")
                else:
                    # Foreground only
                    line.append(f"\033[38;2;{r};{g};{b}m{char}\033[0m")
            else:
                line.append(char)

        lines.append("".join(line))

    return "\n".join(lines)


def image_to_ascii_png(image_path, out_path=None, width=100, charset="techno",
                       font_size=16, brightness_boost=2.8, bg_level=0.50):
    """Convert image to colored ASCII art rendered as PNG.

    This is the DEFAULT output mode for yent.yo — technopunk Warhol style.
    Each pixel becomes a colored glyph on a tinted background cell.

    Args:
        image_path: Path to source image (or PIL Image)
        out_path: Output PNG path (if None, returns PIL Image)
        width: Columns of characters
        font_size: Monospace font size in pixels
        brightness_boost: Foreground color multiplier (>1 = brighter glyphs)
        bg_level: Background fill intensity per cell (0=black, 1=full color)

    Returns:
        PIL Image if out_path is None, else saves and returns path
    """
    from PIL import ImageDraw, ImageFont

    # Load font (prefer bold monospace)
    font = None
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
               "/System/Library/Fonts/Menlo.ttc",
               "/System/Library/Fonts/Monaco.ttf"]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    char_w = font.getbbox("█")[2]
    char_h = font_size + 3

    # Load source
    if isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    chars = CHARSETS.get(charset, charset)
    num_chars = len(chars)

    aspect = img.height / img.width
    height = int(width * aspect * 0.45)
    img = img.resize((width, height), Image.LANCZOS)
    pixels = img.load()

    # Create canvas
    out_w = width * char_w
    out_h = height * char_h
    canvas = Image.new("RGB", (out_w, out_h), (8, 8, 12))
    draw = ImageDraw.Draw(canvas)

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            br = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            idx = max(0, min(int(br * (num_chars - 1)), num_chars - 1))
            ch = chars[idx]

            px, py = x * char_w, y * char_h

            # Background fill per cell — tinted with pixel color
            bg_r = min(255, int(r * bg_level))
            bg_g = min(255, int(g * bg_level))
            bg_b = min(255, int(b * bg_level))
            draw.rectangle([px, py, px + char_w - 1, py + char_h - 1],
                           fill=(bg_r, bg_g, bg_b))

            if ch == " ":
                continue

            # Bright foreground glyph
            cr = min(255, int(r * brightness_boost))
            cg = min(255, int(g * brightness_boost))
            cb = min(255, int(b * brightness_boost))
            draw.text((px, py), ch, fill=(cr, cg, cb), font=font)

    if out_path:
        canvas.save(out_path)
        return out_path
    return canvas


def apply_film_grain(image_path, out_path=None, intensity=25, seed=None):
    """Apply film grain to an image — hides SD artifacts, adds analog style.

    Args:
        image_path: Path to source image (or PIL Image)
        out_path: Output path (if None, returns PIL Image)
        intensity: Grain strength (0-100, default 25)
        seed: Random seed for reproducible grain

    Returns:
        PIL Image if out_path is None, else saves and returns path
    """
    import numpy as np

    if isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    arr = np.array(img, dtype=np.float32)
    rng = np.random.RandomState(seed)

    # Gaussian grain
    noise = rng.normal(0, intensity, arr.shape).astype(np.float32)

    # Slight luminance bias — grain is stronger in shadows (like real film)
    luminance = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    shadow_mask = 1.0 - (luminance / 255.0) * 0.4  # shadows get 1.0x, highlights 0.6x
    for c in range(3):
        noise[:,:,c] *= shadow_mask

    result = np.clip(arr + noise, 0, 255).astype(np.uint8)
    pil_out = Image.fromarray(result)

    if out_path:
        pil_out.save(out_path)
        return out_path
    return pil_out


def image_to_ascii_html(image_path, width=100, charset="techno"):
    """Convert image to colored ASCII as HTML (for saving/sharing)."""
    img = Image.open(image_path).convert("RGB")

    aspect = img.height / img.width
    height = int(width * aspect * 0.5)
    img = img.resize((width, height), Image.LANCZOS)
    pixels = img.load()

    chars = CHARSETS.get(charset, charset)
    num_chars = len(chars)

    html = ['<pre style="background:#000;color:#fff;font-family:monospace;font-size:8px;line-height:1;">']

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            idx = int(brightness * (num_chars - 1))
            idx = max(0, min(idx, num_chars - 1))
            char = chars[idx]
            if char == " ":
                char = "&nbsp;"
            elif char == "<":
                char = "&lt;"
            elif char == ">":
                char = "&gt;"
            elif char == "&":
                char = "&amp;"
            elif char == '"':
                char = "&quot;"
            html.append(f'<span style="color:rgb({r},{g},{b})">{char}</span>')
        html.append("\n")

    html.append("</pre>")
    return "".join(html)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ascii_filter.py <image> [width] [--no-color] [--charset <set>] [--bg] [--html out.html] [--png out.png] [--grain out.png]")
        print(f"Charsets: {', '.join(CHARSETS.keys())}")
        sys.exit(1)

    image_path = sys.argv[1]
    width = 100
    color = True
    charset = "techno"
    bg = False
    html_out = None
    png_out = None
    grain_out = None

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--no-color":
            color = False
        elif arg == "--bg":
            bg = True
        elif arg == "--charset" and i + 1 < len(sys.argv):
            i += 1
            charset = sys.argv[i]
        elif arg == "--html" and i + 1 < len(sys.argv):
            i += 1
            html_out = sys.argv[i]
        elif arg == "--png" and i + 1 < len(sys.argv):
            i += 1
            png_out = sys.argv[i]
        elif arg == "--grain" and i + 1 < len(sys.argv):
            i += 1
            grain_out = sys.argv[i]
        else:
            try:
                width = int(arg)
            except ValueError:
                pass
        i += 1

    if grain_out:
        apply_film_grain(image_path, grain_out)
        sz = os.path.getsize(grain_out) / 1024
        print(f"Saved: {grain_out} ({sz:.0f}KB, film grain)")
    elif png_out:
        image_to_ascii_png(image_path, png_out, width, charset)
        sz = os.path.getsize(png_out) / 1024
        print(f"Saved: {png_out} ({sz:.0f}KB)")
    elif html_out:
        html = image_to_ascii_html(image_path, width, charset)
        with open(html_out, "w") as f:
            f.write(html)
        print(f"Saved: {html_out}")
    else:
        result = image_to_ascii(image_path, width, charset, color, bg)
        print(result)
