#!/usr/bin/env python
"""Create a notebook showing low-level DSP extraction for Marrakech"""
import json

notebook = {
    "cells": [
        # Cell 0: Title
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# DJIA Low-Level DSP Feature Extraction\n",
                "\n",
                "## Single Track Analysis: Marrakech (Hermanez)\n",
                "\n",
                "This notebook demonstrates feature extraction using **low-level DSP functions** directly,\n",
                "without the high-level `extract_track_features()` wrapper.\n",
                "\n",
                "**Goal:** Understand how each DSP engine works step-by-step.\n",
                "\n",
                "---"
            ]
        },

        # Cell 1: Load audio
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 1: Load audio with librosa\n",
                "import librosa\n",
                "import numpy as np\n",
                "from pathlib import Path\n",
                "\n",
                "# Find first audio file\n",
                "data_dir = Path('data')\n",
                "audio_files = sorted(data_dir.glob('*.mp3'))[:1]\n",
                "\n",
                "if not audio_files:\n",
                "    print('ERROR: No MP3 files found in data/')\n",
                "else:\n",
                "    audio_path = audio_files[0]\n",
                "    print(f'Selected: {audio_path.name}')\n",
                "    \n",
                "    # Load audio with librosa\n",
                "    # sr=22050: standard sample rate for analysis\n",
                "    # mono=True: convert to mono (1 channel)\n",
                "    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)\n",
                "    \n",
                "    duration = librosa.get_duration(y=y, sr=sr)\n",
                "    \n",
                "    print(f'\\nAudio Properties:')\n",
                "    print(f'  Sample Rate: {sr} Hz')\n",
                "    print(f'  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)')\n",
                "    print(f'  Samples: {len(y):,}')\n",
                "    print(f'  Channels: 1 (mono)')"
            ]
        },

        # Cell 2: Groove Engine
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Phase 2A: Groove Engine (Step 2)\n",
                "\n",
                "**Goal:** Extract BPM, beat grid, and swing characteristics\n",
                "\n",
                "**Functions used:**\n",
                "- `librosa.beat.onset_strength()` - detect beat onsets\n",
                "- `librosa.beat.tempo()` - estimate BPM\n",
                "- `analyze_groove()` - complete groove analysis"
            ]
        },

        # Cell 3: Run Groove Engine
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 2A: Extract Groove (BPM, beats, swing)\n",
                "from src.dsp.groove_engine import analyze_groove\n",
                "\n",
                "print('Analyzing Groove Engine...')\n",
                "print('  - Detecting beat onsets')\n",
                "print('  - Estimating BPM')\n",
                "print('  - Computing beat grid')\n",
                "print('  - Calculating swing score')\n",
                "print()\n",
                "\n",
                "# Call low-level groove analysis function\n",
                "groove_result = analyze_groove(y, sr, hop_length=512)\n",
                "\n",
                "print('Groove Results:')\n",
                "print(f'  BPM (decimal): {groove_result.bpm:.2f}')\n",
                "print(f'  Beats detected: {len(groove_result.beat_times)}')\n",
                "print(f'  Swing score: {groove_result.swing_score:.3f} (0=stiff, 1=groovy)')\n",
                "print(f'  Tempo stable: {groove_result.tempo_stability}')\n",
                "print(f'  Tempo variance: {groove_result.stability_variance:.4f}')\n",
                "print(f'\\nFirst 10 beat times:')\n",
                "for i, beat_time in enumerate(groove_result.beat_times[:10]):\n",
                "    print(f'    Beat {i+1}: {beat_time:.2f}s')"
            ]
        },

        # Cell 4: Mood Engine
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Phase 2B: Mood Engine (Step 3)\n",
                "\n",
                "**Goal:** Extract harmonic content (key, brightness)\n",
                "\n",
                "**Functions used:**\n",
                "- `librosa.feature.chroma_cqt()` - extract chroma features\n",
                "- `librosa.feature.spectral_centroid()` - compute brightness\n",
                "- `analyze_mood()` - complete mood analysis"
            ]
        },

        # Cell 5: Run Mood Engine
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 2B: Extract Mood (key, brightness)\n",
                "from src.dsp.mood_engine import analyze_mood\n",
                "\n",
                "print('Analyzing Mood Engine...')\n",
                "print('  - Extracting chroma features')\n",
                "print('  - Detecting musical key')\n",
                "print('  - Computing spectral brightness')\n",
                "print()\n",
                "\n",
                "# Call low-level mood analysis function\n",
                "mood_result = analyze_mood(y, sr)\n",
                "\n",
                "print('Mood Results:')\n",
                "print(f'  Musical Key: {mood_result.key}')\n",
                "print(f'  Camelot Key: {mood_result.camelot_key}')\n",
                "print(f'  Brightness: {mood_result.brightness:.3f} (0=dark, 1=bright)')\n",
                "print(f'  Key Confidence: {mood_result.key_confidence:.3f}')"
            ]
        },

        # Cell 6: Phrasing Engine
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Phase 2C: Phrasing Engine (Step 1)\n",
                "\n",
                "**Goal:** Extract structural segments and hot cue positions\n",
                "\n",
                "**Functions used:**\n",
                "- `librosa.stft()` - compute spectrogram\n",
                "- `compute_novelty_curve()` - measure spectral changes\n",
                "- `detect_segment_boundaries()` - find peaks\n",
                "- `create_segments()` - create labeled segments\n",
                "- `analyze_structure()` - complete phrasing analysis"
            ]
        },

        # Cell 7: Run Phrasing Engine
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 2C: Extract Phrasing (segments, structure, cues)\n",
                "from src.dsp.phrasing_engine import analyze_structure\n",
                "\n",
                "print('Analyzing Phrasing Engine...')\n",
                "print('  - Computing STFT')\n",
                "print('  - Calculating spectral novelty curve')\n",
                "print('  - Detecting segment boundaries')\n",
                "print('  - Creating segments and auto-labeling')\n",
                "print('  - Mapping hot cues')\n",
                "print()\n",
                "\n",
                "# Call low-level phrasing analysis function\n",
                "# Pass BPM from groove engine for proper segment labeling\n",
                "phrasing_result = analyze_structure(y, sr, bpm=groove_result.bpm, hop_length=512)\n",
                "\n",
                "print('Phrasing Results:')\n",
                "print(f'  Segment boundaries detected: {len(phrasing_result.segment_boundaries)}')\n",
                "print(f'  Total segments created: {len(phrasing_result.segments)}')\n",
                "print(f'  Hot cues mapped: {len(phrasing_result.cue_points)}')\n",
                "print(f'  Structure confidence: {phrasing_result.structure_confidence:.2f}')\n",
                "print(f'\\nFirst 10 segments:')\n",
                "for i, seg in enumerate(phrasing_result.segments[:10], 1):\n",
                "    duration = seg.end_time - seg.start_time\n",
                "    bar_start = (seg.start_time * groove_result.bpm) / 60 / 4\n",
                "    bar_end = (seg.end_time * groove_result.bpm) / 60 / 4\n",
                "    print(f'  {i:2d}. {seg.label:10s} {seg.start_time:7.2f}s → {seg.end_time:7.2f}s '\n",
                "          f'({duration:6.2f}s, bars {bar_start:6.2f}-{bar_end:6.2f}, conf {seg.confidence:.0%})')"
            ]
        },

        # Cell 8: Curation Engine
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Phase 2D: Curation Engine (Step 4)\n",
                "\n",
                "**Goal:** Extract danceability, energy, and semantic features\n",
                "\n",
                "**Functions used:**\n",
                "- `librosa.feature.zero_crossing_rate()` - percussiveness\n",
                "- `librosa.feature.tempogram()` - rhythm strength\n",
                "- `librosa.feature.spectral_centroid()` - brightness\n",
                "- `analyze_curation()` - complete curation analysis"
            ]
        },

        # Cell 9: Run Curation Engine
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 2D: Extract Curation (danceability, energy, tags)\n",
                "from src.dsp.curation_engine import analyze_curation\n",
                "\n",
                "print('Analyzing Curation Engine...')\n",
                "print('  - Computing zero-crossing rate (percussiveness)')\n",
                "print('  - Extracting tempogram (rhythm strength)')\n",
                "print('  - Analyzing spectral content')\n",
                "print('  - Generating semantic tags')\n",
                "print()\n",
                "\n",
                "# Call low-level curation analysis function\n",
                "# Note: Curation engine requires data from other engines\n",
                "curation_result = analyze_curation(\n",
                "    y, sr,\n",
                "    bpm=groove_result.bpm,\n",
                "    swing_score=groove_result.swing_score,\n",
                "    brightness=mood_result.brightness,\n",
                "    hop_length=512\n",
                ")\n",
                "\n",
                "print('Curation Results:')\n",
                "print(f'  Danceability: {curation_result.danceability:.3f} (0-1 scale)')\n",
                "print(f'  Energy type: {curation_result.energy_type}')\n",
                "print(f'  Complexity score: {curation_result.complexity_score:.3f}')\n",
                "print(f'  Semantic tags: {', '.join(curation_result.semantic_tags)}')"
            ]
        },

        # Cell 10: Summary
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Summary: Complete DSP Feature Extraction\n",
                "\n",
                "All 4 DSP engines have been applied to the Marrakech track using **low-level functions**."
            ]
        },

        # Cell 11: Display all results
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Summary of all extracted features\n",
                "print('\\n' + '='*80)\n",
                "print('COMPLETE FEATURE EXTRACTION FOR: Marrakech (Hermanez)')\n",
                "print('='*80)\n",
                "\n",
                "print('\\n[GROOVE ENGINE - Rhythm Analysis]')\n",
                "print(f'  BPM: {groove_result.bpm:.2f} (decimal precision)')\n",
                "print(f'  Beats: {len(groove_result.beat_times)} detected')\n",
                "print(f'  Swing: {groove_result.swing_score:.2f} (0=stiff, 1=groovy)')\n",
                "print(f'  Tempo Stable: {groove_result.tempo_stability}')\n",
                "\n",
                "print('\\n[MOOD ENGINE - Harmonic Analysis]')\n",
                "print(f'  Musical Key: {mood_result.key}')\n",
                "print(f'  Camelot Key: {mood_result.camelot_key}')\n",
                "print(f'  Brightness: {mood_result.brightness:.2f} (0=dark, 1=bright)')\n",
                "print(f'  Key Confidence: {mood_result.key_confidence:.2f}')\n",
                "\n",
                "print('\\n[PHRASING ENGINE - Structure Analysis]')\n",
                "print(f'  Total Segments: {len(phrasing_result.segments)}')\n",
                "print(f'  Segment Types: ', end='')\n",
                "types = {}\n",
                "for seg in phrasing_result.segments:\n",
                "    types[seg.label] = types.get(seg.label, 0) + 1\n",
                "print(', '.join(f'{t}={n}' for t, n in sorted(types.items())))\n",
                "print(f'  Hot Cues: {len(phrasing_result.cue_points)} mapped')\n",
                "\n",
                "print('\\n[CURATION ENGINE - Semantic Analysis]')\n",
                "print(f'  Danceability: {curation_result.danceability:.2f}')\n",
                "print(f'  Energy Type: {curation_result.energy_type}')\n",
                "print(f'  Tags: {', '.join(curation_result.semantic_tags)}')\n",
                "\n",
                "print('\\n' + '='*80)\n",
                "print('Feature extraction complete! All data ready for database storage.')\n",
                "print('='*80)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write notebook
import json
from pathlib import Path

notebook_path = Path("DJIA_LowLevel_FeatureExtraction.ipynb")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Created {notebook_path}")
print(f"[OK] Contains 12 cells showing low-level DSP extraction")
print(f"[OK] Single track (Marrakech) analysis only")
