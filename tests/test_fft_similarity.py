import sys
from pathlib import Path

import numpy as np
from pydub import AudioSegment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import _fft_similarity_score, _segment_to_samples, _stft_magnitude


def _sine_segment(freq: float, duration: float = 1.0, rate: int = 16000, amplitude: float = 0.6) -> AudioSegment:
    """Create a mono sine wave segment for testing."""
    t = np.arange(int(rate * duration))
    wave = amplitude * np.sin(2 * np.pi * freq * t / rate)
    samples = (wave * (2**15 - 1)).astype(np.int16)
    return AudioSegment(
        samples.tobytes(),
        frame_rate=rate,
        sample_width=2,
        channels=1,
    )


def test_segment_to_samples_is_normalized():
    seg = _sine_segment(440.0, duration=0.25, amplitude=0.9)
    samples = _segment_to_samples(seg)
    assert samples.size > 0
    assert samples.max() <= 1.0 + 1e-6
    assert samples.min() >= -1.0 - 1e-6
    assert np.isclose(np.max(np.abs(samples)), 1.0, atol=1e-2)


def test_stft_magnitude_frames_are_unit_norm():
    seg = _sine_segment(440.0, duration=0.5)
    samples = _segment_to_samples(seg)
    mags = _stft_magnitude(samples, fft_size=512, hop=128)
    norms = np.linalg.norm(mags, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_fft_similarity_prefers_matching_tones():
    reference = _sine_segment(440.0, duration=1.0)
    similar = _sine_segment(440.0, duration=1.0, amplitude=0.4)
    different = _sine_segment(880.0, duration=1.0)

    high_score, high_sim = _fft_similarity_score(reference, similar)
    low_score, low_sim = _fft_similarity_score(reference, different)

    assert high_score > low_score
    assert high_sim > low_sim
    assert high_score > 80
    assert low_score < high_score - 5
