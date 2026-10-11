#!/usr/bin/env python3
"""Render a Traktor Pro-style frequency-coloured waveform for an audio file.

Colour follows the spectral balance (low = red, mid = orange, high = blue/violet)
and height follows amplitude, with an optional detected beat grid — the look of
Traktor's deck waveform.

Usage:
    python render_waveform.py <audio> [-o OUT.png] [--duration SEC] [--offset SEC]
                                      [--width W] [--height H] [--no-grid]
                                      [--playhead 0.5]

Examples:
    python render_waveform.py track.mp3
    python render_waveform.py track.mp3 -o results/wave.png --duration 30 --playhead 0.5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Load the renderer straight from its file so drawing a waveform only needs
# numpy/librosa/Pillow — not the full pipeline stack that `import src` pulls in.
_spec = importlib.util.spec_from_file_location(
    "djia_waveform", Path(__file__).parent / "src" / "viz" / "waveform.py"
)
_wf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _wf  # let dataclass resolve annotations on WaveformStyle
_spec.loader.exec_module(_wf)
WaveformStyle = _wf.WaveformStyle
render_waveform_file = _wf.render_waveform_file


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("audio", help="Path to the audio file")
    ap.add_argument("-o", "--out", help="Output PNG path (default: results/<name>_waveform.png)")
    ap.add_argument(
        "--duration", type=float, default=None, help="Seconds to render (default: whole track)"
    )
    ap.add_argument("--offset", type=float, default=0.0, help="Start offset in seconds")
    ap.add_argument("--width", type=int, default=2000, help="Image width in px")
    ap.add_argument("--height", type=int, default=340, help="Image height in px")
    ap.add_argument("--sr", type=int, default=44100, help="Analysis sample rate")
    ap.add_argument("--no-grid", action="store_true", help="Disable the beat grid")
    ap.add_argument(
        "--playhead", type=float, default=None, help="Draw playhead at fraction 0-1 of width"
    )
    args = ap.parse_args()

    audio = Path(args.audio)
    out = Path(args.out) if args.out else Path("results") / f"{audio.stem}_waveform.png"

    style = WaveformStyle(
        width=args.width,
        height=args.height,
        show_beatgrid=not args.no_grid,
        playhead=args.playhead,
    )
    written = render_waveform_file(
        audio, out, style=style, sr=args.sr, duration=args.duration, offset=args.offset
    )
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
