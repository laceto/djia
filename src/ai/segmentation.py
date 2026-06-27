"""Structural segmentation to detect musical sections and landmarks."""

from typing import List, Dict, Optional, Tuple
import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore')


class StructurePoint:
    """Represents a structural landmark in the track."""

    def __init__(self, time: float, structure_type: str, confidence: float):
        """
        Initialize a structure point.

        Args:
            time: Timestamp in seconds.
            structure_type: Type of structure (intro, drop, breakdown, build, bridge, outro).
            confidence: Confidence score (0-1).
        """
        self.time = time
        self.structure_type = structure_type
        self.confidence = confidence

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'time': self.time,
            'type': self.structure_type,
            'confidence': self.confidence,
        }

    def __repr__(self) -> str:
        return f"StructurePoint({self.time:.2f}s, {self.structure_type}, conf={self.confidence:.2f})"


class StructureSegmenter:
    """Detects musical structure sections in audio tracks."""

    STRUCTURE_TYPES = ['intro', 'build', 'drop', 'breakdown', 'bridge', 'outro']

    def __init__(self, hop_length: int = 512, fft_size: int = 2048):
        """
        Initialize segmenter.

        Args:
            hop_length: Hop length for STFT.
            fft_size: FFT size.
        """
        self.hop_length = hop_length
        self.fft_size = fft_size

    def _compute_novelty_curve(
        self,
        y: np.ndarray,
        sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute novelty curve to detect structural changes.

        Returns:
            (novelty_curve, time_frames) - novelty scores and corresponding times.
        """
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.fft_size, hop_length=self.hop_length
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # Compute spectral flux (change detection)
        flux = np.sqrt(np.sum(np.diff(S_db, axis=1) ** 2, axis=0))
        flux = flux / (np.max(flux) + 1e-8)  # Normalize

        # Smooth with median filter
        flux_smooth = np.median(flux)
        novelty = (flux - flux_smooth) / (np.std(flux) + 1e-8)
        novelty = np.maximum(novelty, 0)  # Half-wave rectify
        novelty = novelty / (np.max(novelty) + 1e-8)  # Normalize to [0,1]

        # Time frames
        frames = np.arange(len(novelty))
        times = librosa.frames_to_time(frames, sr=sr, hop_length=self.hop_length)

        return novelty, times

    def _compute_energy_curve(
        self,
        y: np.ndarray,
        sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute energy curve for structural analysis.

        Returns:
            (energy_curve, time_frames).
        """
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.fft_size, hop_length=self.hop_length
        )
        energy = np.sum(S, axis=0)
        energy = energy / (np.max(energy) + 1e-8)

        frames = np.arange(len(energy))
        times = librosa.frames_to_time(frames, sr=sr, hop_length=self.hop_length)

        return energy, times

    def _detect_peaks(
        self,
        curve: np.ndarray,
        times: np.ndarray,
        threshold: float = 0.3,
        min_distance: float = 10.0
    ) -> List[Tuple[float, float]]:
        """
        Detect peaks in a curve.

        Args:
            curve: Signal curve.
            times: Corresponding time values.
            threshold: Peak threshold (relative to max).
            min_distance: Minimum distance between peaks (seconds).

        Returns:
            List of (time, value) tuples.
        """
        if len(curve) == 0:
            return []

        # Find peaks above threshold
        peak_threshold = threshold * np.max(curve)
        peaks = []

        for i in range(1, len(curve) - 1):
            if curve[i] > peak_threshold and curve[i] > curve[i - 1] and curve[i] > curve[i + 1]:
                peaks.append((times[i], curve[i]))

        # Remove duplicates within min_distance
        if peaks:
            filtered_peaks = [peaks[0]]
            for time, value in peaks[1:]:
                if time - filtered_peaks[-1][0] > min_distance:
                    filtered_peaks.append((time, value))
            peaks = filtered_peaks

        return peaks

    def _classify_structure(
        self,
        y_drums: np.ndarray,
        y_full: np.ndarray,
        sr: int
    ) -> List[StructurePoint]:
        """
        Classify structural segments based on audio characteristics.

        Args:
            y_drums: Drum stem (if available) or percussive component.
            y_full: Full audio track.
            sr: Sample rate.

        Returns:
            List of StructurePoint objects.
        """
        novelty, times = self._compute_novelty_curve(y_full, sr)
        energy_full, _ = self._compute_energy_curve(y_full, sr)
        energy_drums, _ = self._compute_energy_curve(y_drums, sr)

        # Detect peaks
        peaks_novelty = self._detect_peaks(novelty, times, threshold=0.4, min_distance=15.0)
        peaks_energy = self._detect_peaks(energy_full, times, threshold=0.5, min_distance=20.0)

        structure_points = []

        # Identify intro (low energy at start)
        if len(times) > 0:
            intro_energy = np.mean(energy_full[:len(energy_full) // 10])
            if intro_energy < 0.3:
                structure_points.append(StructurePoint(
                    time=0.0,
                    structure_type='intro',
                    confidence=0.8
                ))

        # Process detected peaks
        for peak_time, peak_value in peaks_novelty:
            # Get local energy change
            idx = np.argmin(np.abs(times - peak_time))
            energy_before = np.mean(energy_full[max(0, idx - 20):idx]) if idx > 20 else 0.5
            energy_after = np.mean(energy_full[idx:min(len(energy_full), idx + 20)])

            # Classify structure type based on characteristics
            drums_intensity = np.mean(energy_drums[max(0, idx - 20):min(len(energy_drums), idx + 20)])

            structure_type = 'build'  # Default
            confidence = peak_value

            # Main drop: energy increase + high drum intensity
            if energy_after > energy_before * 1.3 and drums_intensity > 0.5:
                structure_type = 'drop'
                confidence = min(1.0, peak_value + 0.2)

            # Breakdown: energy decrease
            elif energy_after < energy_before * 0.7:
                structure_type = 'breakdown'
                confidence = min(1.0, (1.0 - peak_value) * 0.8 + 0.2)

            # Build-up
            elif 0.05 < (energy_after - energy_before) < 0.3:
                structure_type = 'build'
                confidence = peak_value

            structure_points.append(StructurePoint(
                time=peak_time,
                structure_type=structure_type,
                confidence=min(1.0, confidence)
            ))

        # Identify outro (low energy at end)
        outro_start_idx = int(0.85 * len(energy_full))
        if outro_start_idx < len(energy_full):
            outro_energy = np.mean(energy_full[outro_start_idx:])
            if outro_energy < 0.3:
                structure_points.append(StructurePoint(
                    time=times[outro_start_idx],
                    structure_type='outro',
                    confidence=0.8
                ))

        # Sort by time
        structure_points.sort(key=lambda x: x.time)

        return structure_points

    def detect_structure(
        self,
        y: np.ndarray,
        sr: int,
        y_drums: Optional[np.ndarray] = None
    ) -> List[StructurePoint]:
        """
        Detect musical structure and sections.

        Args:
            y: Full audio track (mono).
            sr: Sample rate.
            y_drums: Optional drum stem (mono). If not provided, uses percussive separation.

        Returns:
            List of StructurePoint objects sorted by time.
        """
        # Extract drums if not provided
        if y_drums is None:
            y_drums, _ = librosa.effects.hpss(y, margin=4.0)

        return self._classify_structure(y_drums, y, sr)


def detect_structure(
    y: np.ndarray,
    sr: int,
    y_drums: Optional[np.ndarray] = None
) -> List[StructurePoint]:
    """
    Convenience function to detect musical structure.

    Args:
        y: Audio time series (mono).
        sr: Sample rate.
        y_drums: Optional drum stem.

    Returns:
        List of StructurePoint objects.
    """
    segmenter = StructureSegmenter()
    return segmenter.detect_structure(y, sr, y_drums=y_drums)
