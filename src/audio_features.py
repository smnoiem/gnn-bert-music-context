from __future__ import annotations

import numpy as np


def load_audio(path: str, sample_rate: int = 22050):
    import librosa
    return librosa.load(path, sr=sample_rate, mono=True)


def mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    import librosa
    value = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(value, ref=np.max).astype(np.float32)


def segment_features(y: np.ndarray, sr: int, seconds: float = 5.0) -> np.ndarray:
    """Return per-window normalized chroma + MFCC mean features."""
    import librosa
    width, pieces = max(1, int(seconds * sr)), []
    for start in range(0, len(y), width):
        clip = y[start:start + width]
        if len(clip) < sr: continue
        chroma = librosa.feature.chroma_stft(y=clip, sr=sr).mean(axis=1)
        mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=20).mean(axis=1)
        pieces.append(np.concatenate([chroma, mfcc]))
    if not pieces: pieces = [np.zeros(32, dtype=np.float32)]
    features = np.asarray(pieces, dtype=np.float32)
    return (features - features.mean(0)) / (features.std(0) + 1e-6)
