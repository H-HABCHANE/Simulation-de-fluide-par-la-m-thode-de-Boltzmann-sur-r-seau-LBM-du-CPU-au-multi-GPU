#!/usr/bin/env python3
"""Animation of the vorticity field of a cylinder run (von Karman vortex street).

Reads OUTPUT_DIR/frames/vort_*.npy and OUTPUT_DIR/solid.npy and writes:
- a light GIF for the README (default docs/karman.gif, checked to stay under --max-mb);
- an MP4 next to the frames if ffmpeg is available (not versioned: *.mp4 is git-ignored).

Colours: diverging scale, blue (clockwise) - neutral gray (no rotation) - red
(counter-clockwise), symmetric limits at the 99th percentile of |vorticity| over all frames
so that the colour of a given vorticity does not change from frame to frame; cylinder in dark
gray.

Usage: make_video.py OUTPUT_DIR [--gif docs/karman.gif] [--width 720] [--fps 20]
"""
import argparse
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# Reference diverging pair: blue <-> red around a neutral gray midpoint.
CMAP = LinearSegmentedColormap.from_list(
    "vorticity", ["#1d5aa8", "#2a78d6", "#f0efec", "#e34948", "#b22d2c"])
SOLID_RGB = np.array([82, 81, 78], dtype=np.uint8)  # #52514e


def render(vort, solid, vmax, width):
    rgb = (CMAP(np.clip(vort / vmax, -1, 1) * 0.5 + 0.5)[..., :3] * 255).astype(np.uint8)
    rgb[solid] = SOLID_RGB
    img = Image.fromarray(rgb[::-1])  # y up
    height = round(width * vort.shape[0] / vort.shape[1])
    return img.resize((width, height), Image.LANCZOS)


def write_gif(images, path, fps):
    palette = images[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=round(1000 / fps), loop=0, optimize=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return len(buf.getvalue()) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--gif", type=Path, default=Path("docs/karman.gif"))
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-mb", type=float, default=5.0)
    args = ap.parse_args()

    files = sorted((args.output_dir / "frames").glob("vort_*.npy"))
    if not files:
        sys.exit(f"no frames in {args.output_dir / 'frames'} (run with --frame-every)")
    solid = np.load(args.output_dir / "solid.npy").astype(bool)
    fields = [np.load(f) for f in files]
    # Frames written before the one-sided edge differences existed have a spurious vorticity
    # on the open edges (periodic central differences mixing inlet and outlet): blank them.
    for v in fields:
        v[:, 0] = v[:, -1] = 0.0
    vmax = float(np.percentile(np.abs(np.stack(fields)[:, ~solid]), 99))
    images = [render(v, solid, vmax, args.width) for v in fields]
    print(f"{len(images)} frames, |vorticity| scale = {vmax:.3e}")

    # Keep the GIF light: drop every other frame, then reduce the width, until it fits.
    width, step = args.width, 1
    size = write_gif(images, args.gif, args.fps)
    while size > args.max_mb:
        if step == 1:
            step = 2
        else:
            width = int(width * 0.8)
            images = [render(v, solid, vmax, width) for v in fields]
        size = write_gif(images[::step], args.gif, args.fps / step)
    print(f"-> {args.gif} ({size:.2f} MB, {len(images[::step])} frames, width {width} px)")

    if shutil.which("ffmpeg"):
        mp4 = args.output_dir / "karman.mp4"
        with tempfile.TemporaryDirectory() as tmp:
            for k, v in enumerate(fields):
                render(v, solid, vmax, 1280 - 1280 % 2).save(Path(tmp) / f"f{k:05d}.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                            "-i", str(Path(tmp) / "f%05d.png"), "-pix_fmt", "yuv420p",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(mp4)], check=True)
        print(f"-> {mp4} (not versioned)")
    else:
        print("ffmpeg not found: no MP4 written")


if __name__ == "__main__":
    main()
