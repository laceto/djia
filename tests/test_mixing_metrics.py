"""Tests for mixing metrics scoring."""

import pytest
from src.mixing_metrics import (
    calculate_bpm_compatibility,
    calculate_key_compatibility,
)


class TestBPMCompatibility:
    def test_identical_bpm(self):
        assert calculate_bpm_compatibility(128, 128) == 1.0

    def test_within_tolerance(self):
        assert calculate_bpm_compatibility(128, 130) > 0.7

    def test_very_different_bpm(self):
        assert calculate_bpm_compatibility(128, 180) < 0.5


class TestKeyCompatibility:
    def test_same_key(self):
        assert calculate_key_compatibility("C", "C") == 1.0

    def test_perfect_fifth(self):
        assert calculate_key_compatibility("C", "G") == 0.9

    def test_perfect_fourth(self):
        assert calculate_key_compatibility("C", "F") == 0.9

    def test_incompatible_key(self):
        assert calculate_key_compatibility("C", "F#") < 0.5
