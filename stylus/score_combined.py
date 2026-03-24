"""
Spectral scores for audio style transfer evaluation.

  - mel_style_score   : style transfer (cosine on Mel)
  - mfcc_content_score: content preservation (cosine on MFCC)
  - combined_score    : λ·style + (1-λ)·content
"""

import numpy as np
import librosa


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def mel_style_score(output: np.ndarray, style_ref: np.ndarray, sr: int) -> float:
    """Cosine similarity between mean MEL spectra (style transfer)."""

    def embed(audio):
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        return librosa.power_to_db(mel).mean(axis=1)

    return _cosine(embed(output), embed(style_ref))


def mfcc_content_score(output: np.ndarray, content_ref: np.ndarray, sr: int) -> float:
    """Cosine similarity of mean MFCCs (content preservation)."""

    def embed(audio):
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20).mean(axis=1)

    return _cosine(embed(output), embed(content_ref))


def combined_score(
    output: np.ndarray,
    style_ref: np.ndarray,
    content_ref: np.ndarray,
    sr: int,
    lam: float = 0.5,
) -> float:
    """λ·mel_style_score + (1-λ)·mfcc_content_score."""
    return lam * mel_style_score(output, style_ref, sr) + (
        1 - lam
    ) * mfcc_content_score(output, content_ref, sr)
