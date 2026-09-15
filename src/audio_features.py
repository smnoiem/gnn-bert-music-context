from __future__ import annotations

import numpy as np


DEFAULT_N_MFCC = 20
CHROMA_BINS = 12
SPECTRAL_CONTRAST_BANDS = 7
TONNETZ_COMPONENTS = 6
AUDIO_FEATURE_DIM = (
    DEFAULT_N_MFCC * 6
    + CHROMA_BINS * 2
    + SPECTRAL_CONTRAST_BANDS * 2
    + TONNETZ_COMPONENTS * 2
    + 6 * 2
)


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
    n_mfcc: int = DEFAULT_N_MFCC,
) -> np.ndarray:
    """Extract MFCC, delta, and delta-delta statistics from a segment."""
    import librosa

    if n_mfcc <= 0:
        raise ValueError("n_mfcc must be positive")
    values = librosa.feature.mfcc(
        y=segment, sr=sample_rate, n_mfcc=n_mfcc, n_fft=2048, hop_length=512
    )
    features = [values, librosa.feature.delta(values), librosa.feature.delta(values, order=2)]
    return np.concatenate(
        [np.concatenate([value.mean(axis=1), value.std(axis=1)]) for value in features]
    ).astype(np.float32)


def extract_chroma(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract mean and variation of the pitch-class energy distribution."""
    import librosa

    values = librosa.feature.chroma_stft(
        y=segment, sr=sample_rate, n_fft=2048, hop_length=512
    )
    return np.concatenate([values.mean(axis=1), values.std(axis=1)]).astype(np.float32)


def extract_spectral_features(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract spectral shape, harmonic, rhythm, and energy statistics."""
    import librosa

    magnitude = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512))
    frame_kwargs = {"hop_length": 512}
    chroma = librosa.feature.chroma_stft(
        S=magnitude, sr=sample_rate, n_fft=2048, hop_length=512
    )
    features = [
        librosa.feature.spectral_contrast(
            S=magnitude, sr=sample_rate, n_fft=2048, hop_length=512
        ),
        librosa.feature.tonnetz(chroma=chroma, sr=sample_rate),
        librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate, **frame_kwargs),
        librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate, **frame_kwargs),
        librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate, **frame_kwargs),
        librosa.feature.zero_crossing_rate(
            segment, frame_length=2048, **frame_kwargs
        ),
        librosa.feature.rms(S=magnitude, **frame_kwargs),
        librosa.feature.spectral_flatness(S=magnitude, **frame_kwargs),
    ]
    return np.concatenate(
        [np.concatenate([value.mean(axis=1), value.std(axis=1)]) for value in features]
    ).astype(np.float32)


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
    n_mfcc: int = DEFAULT_N_MFCC,
) -> np.ndarray:
    """Create normalized, information-rich audio vectors for graph nodes.

    Each segment contains MFCC means/stds and their first two deltas, chroma
    means/stds, and statistics for spectral contrast, tonnetz, centroid,
    bandwidth, rolloff, zero-crossing rate, RMS energy, and spectral flatness.
    """
    if not segments:
        feature_count = (n_mfcc * 6) + (CHROMA_BINS * 2) + (
            SPECTRAL_CONTRAST_BANDS * 2
        ) + (TONNETZ_COMPONENTS * 2) + (6 * 2)
        return np.zeros((1, feature_count), dtype=np.float32)

    features = [
        np.concatenate(
            [
                extract_mfcc(segment, sample_rate, n_mfcc),
                extract_chroma(segment, sample_rate),
                extract_spectral_features(segment, sample_rate),
            ]
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
