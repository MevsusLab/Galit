"""Generate GALIT web icon rasters from the repository SVG geometry."""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "assets" / "icons" / "galit-mark.svg"
OUT = ROOT / "assets" / "icons"
SIZES = (24, 32, 40, 48)
SCALE = 16


def _sample_path(data: str) -> list[tuple[float, float]]:
    """Sample the SVG's single M/L/H/V/C path into a high-resolution polygon."""
    tokens = re.findall(r"[A-Za-z]|-?(?:\d+(?:\.\d*)?|\.\d+)", data)
    points: list[tuple[float, float]] = []
    cursor = (0.0, 0.0)
    start = cursor
    index = 0
    command = ""

    def number() -> float:
        nonlocal index
        value = float(tokens[index])
        index += 1
        return value

    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
        relative = command.islower()
        op = command.upper()
        if op == "Z":
            points.append(start)
            command = ""
            continue
        if op == "M" or op == "L":
            x, y = number(), number()
            if relative:
                x, y = cursor[0] + x, cursor[1] + y
            cursor = (x, y)
            points.append(cursor)
            if op == "M":
                start = cursor
                command = "l" if relative else "L"
        elif op == "H":
            x = number() + (cursor[0] if relative else 0)
            cursor = (x, cursor[1])
            points.append(cursor)
        elif op == "V":
            y = number() + (cursor[1] if relative else 0)
            cursor = (cursor[0], y)
            points.append(cursor)
        elif op == "C":
            values = [number() for _ in range(6)]
            x1, y1, x2, y2, x3, y3 = values
            if relative:
                x1, y1 = x1 + cursor[0], y1 + cursor[1]
                x2, y2 = x2 + cursor[0], y2 + cursor[1]
                x3, y3 = x3 + cursor[0], y3 + cursor[1]
            x0, y0 = cursor
            for step in range(1, 25):
                t = step / 24
                u = 1 - t
                points.append((
                    u**3*x0 + 3*u*u*t*x1 + 3*u*t*t*x2 + t**3*x3,
                    u**3*y0 + 3*u*u*t*y1 + 3*u*t*t*y2 + t**3*y3,
                ))
            cursor = (x3, y3)
        else:
            raise ValueError(f"Unsupported SVG command: {command}")
    return points


def main() -> None:
    markup = SVG.read_text(encoding="utf-8")
    viewbox = tuple(map(float, re.search(r'viewBox="([^"]+)"', markup).group(1).split()))
    data = re.search(r'<path[^>]+ d="([^"]+)"', markup).group(1)
    points = _sample_path(data)
    # Apply the SVG transform: translate(1 0) scale(.94 1).
    points = [(1 + .94*x, y) for x, y in points]
    width, height = int(viewbox[2]), int(viewbox[3])
    rendered: dict[int, Image.Image] = {}
    for size in SIZES:
        canvas = Image.new("RGBA", (size*SCALE, size*SCALE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        fit = min((size*SCALE-2*SCALE)/width, (size*SCALE-2*SCALE)/height)
        offset_x = (size*SCALE-width*fit)/2
        offset_y = (size*SCALE-height*fit)/2
        polygon = [(round(offset_x+x*fit), round(offset_y+y*fit)) for x, y in points]
        draw.polygon(polygon, fill=(8, 122, 61, 255))
        image = canvas.resize((size, size), Image.Resampling.LANCZOS)
        image.save(OUT / f"galit-mark-{size}.png", optimize=True)
        rendered[size] = image
    rendered[48].save(
        OUT / "galit-favicon.ico", format="ICO", sizes=[(24, 24), (32, 32), (48, 48)]
    )


if __name__ == "__main__":
    main()
