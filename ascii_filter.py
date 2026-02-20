#!/usr/bin/env python3
"""Colored ASCII art filter for yent.yo images.

Converts PNG/JPG to colored ASCII art using ANSI truecolor escape codes.
Can be used as standalone CLI or imported as module.

Usage:
  python3 ascii_filter.py <image> [width] [--no-color] [--charset <set>]
  python3 ascii_filter.py ort_cat_25step_gpu.png 120
  python3 ascii_filter.py lora_cat_graffiti.png 80 --charset blocks

Charsets:
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
    "standard": " .:-=+*#%@",
    "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "blocks": " ░▒▓█",
    "minimal": " .oO@",
    "punk": " ·•×#█",  # yent.yo default
}


def image_to_ascii(image_path, width=100, charset="punk", color=True, bg_color=False):
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


def image_to_ascii_html(image_path, width=100, charset="punk"):
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
        print("Usage: python3 ascii_filter.py <image> [width] [--no-color] [--charset <set>] [--bg] [--html out.html]")
        print(f"Charsets: {', '.join(CHARSETS.keys())}")
        sys.exit(1)

    image_path = sys.argv[1]
    width = 100
    color = True
    charset = "punk"
    bg = False
    html_out = None

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
        else:
            try:
                width = int(arg)
            except ValueError:
                pass
        i += 1

    if html_out:
        html = image_to_ascii_html(image_path, width, charset)
        with open(html_out, "w") as f:
            f.write(html)
        print(f"Saved: {html_out}")
    else:
        result = image_to_ascii(image_path, width, charset, color, bg)
        print(result)
