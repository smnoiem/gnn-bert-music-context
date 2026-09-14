from __future__ import annotations

import numpy as np


def load_audio(path: str, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono waveform at the requested sample rate."""
    import librosa

    return librosa.load(path, sr=sample_rate, mono=True)


def segment_audio(
    waveform: np.ndarray,
    sample_rate: int = 22050,
    segment_seconds: float = 5.0,
    minimum_seconds: float = 1.0,
) -> list[np.ndarray]:
    """Split a waveform into fixed-duration segments.

    A trailing segment shorter than ``minimum_seconds`` is discarded so that
    very short fragments do not become low-quality graph nodes.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    if minimum_seconds < 0 or minimum_seconds > segment_seconds:
        raise ValueError("minimum_seconds must be between zero and segment_seconds")

    width = max(1, round(segment_seconds * sample_rate))
    minimum_samples = round(minimum_seconds * sample_rate)
    return [
        waveform[start : start + width]
        for start in range(0, len(waveform), width)
        if len(waveform[start : start + width]) >= minimum_samples
    ]


def extract_mfcc(
    segment: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 20,
) -> np.ndarray:
    """Extract one mean MFCC vector from an audio segment."""
    import librosa

    if n_mfcc <= 0:
        raise ValueError("n_mfcc must be positive")
    values = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
    return values.mean(axis=1).astype(np.float32)


def extract_chroma(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract one mean chroma vector from an audio segment."""
    import librosa

    values = librosa.feature.chroma_stft(y=segment, sr=sample_rate)
    return values.mean(axis=1).astype(np.float32)


def normalize_features(features: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Normalize feature columns while avoiding division by zero."""
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean = features.mean(axis=0, keepdims=True)
    standard_deviation = features.std(axis=0, keepdims=True)
    return ((features - mean) / (standard_deviation + epsilon)).astype(np.float32)


def extract_segment_features(
    segments: list[np.ndarray],
    sample_rate: int,
    n_mfcc: int = 20,
) -> np.ndarray:
    """Create normalized MFCC+chroma vectors for graph nodes."""
    if not segments:
        return np.zeros((1, n_mfcc + 12), dtype=np.float32)

    features = [
        np.concatenate(
            [extract_mfcc(segment, sample_rate, n_mfcc), extract_chroma(segment, sample_rate)]
        )
        for segment in segments
    ]
    return normalize_features(np.asarray(features, dtype=np.float32))


def mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
) -> np.ndarray:
    """Return a log-mel spectrogram for the mel-CNN baseline."""
    import librosa

    if n_mels <= 0:
        raise ValueError("n_mels must be positive")
    value = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
    return librosa.power_to_db(value, ref=np.max).astype(np.float32)


def segment_features(
    waveform: np.ndarray,
    sample_rate: int,
    seconds: float = 5.0,
) -> np.ndarray:
    """Compatibility wrapper returning normalized MFCC+chroma segment features."""
    segments = segment_audio(waveform, sample_rate, seconds)
    return extract_segment_features(segments, sample_rate)
