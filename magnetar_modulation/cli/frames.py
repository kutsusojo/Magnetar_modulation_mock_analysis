#!/usr/bin/env python
"""frames_to_gif.py

Convert a sequence of PNG frames into **either** a GIF *or* an MP4.

Examples
--------
Create a looping GIF::

    $ python frames_to_gif.py frames_PA

Create an H.264 video (default 6 fps)::

    $ python frames_to_gif.py frames_PA mp4 --fps 12

The script discovers frames by globbing ``*.png`` inside *frames_dir*
and ordering them numerically (i.e. frame_0001.png, frame_0002.png …).

Dependencies
------------
* `imageio[v3] <https://imageio.readthedocs.io/>`_  (for reading PNGs)
* ``imageio-ffmpeg`` (only if *mp4* output is requested)

"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import imageio.v3 as iio
from imageio import get_writer, mimsave


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "frames_dir",
        type=Path,
        help="Directory that contains the *.png frame files",
    )
    parser.add_argument(
        "format",
        choices=("gif", "mp4"),
        nargs="?",
        default="gif",
        help="Output format (default: gif)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frame rate for MP4 output (ignored for GIF)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.15,
        help="Frame duration in seconds for GIF output",
    )
    return parser.parse_args()


def numeric_key(path: Path) -> int:
    """Return the last integer in *path.stem* for correct sort order."""
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else -1


def collect_frames(frames_dir: Path) -> list[Path]:
    pngs = sorted(frames_dir.glob("*.png"), key=numeric_key)
    if not pngs:
        sys.exit("[error] No *.png files found in the directory.")
    return pngs


def write_gif(pngs: list[Path], out_file: Path, duration: float) -> None:
    images = [iio.imread(p) for p in pngs]
    mimsave(out_file, images, duration=duration)
    print(f"GIF written → {out_file}")


def write_mp4(pngs: list[Path], out_file: Path, fps: int) -> None:
    with get_writer(out_file, fps=fps, codec="libx264") as writer:
        for p in pngs:
            writer.append_data(iio.imread(p))
    print(f"MP4 written → {out_file}")


def main() -> None:
    args = parse_args()
    frames = collect_frames(args.frames_dir)

    out_file = args.frames_dir.with_suffix("." + args.format)

    if args.format == "gif":
        write_gif(frames, out_file, args.duration)
    else:
        write_mp4(frames, out_file, fps=args.fps)


if __name__ == "__main__":  # pragma: no cover
    main()
