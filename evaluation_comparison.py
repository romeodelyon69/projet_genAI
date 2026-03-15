#!/usr/bin/env python3
"""
evaluation_comparison.py
────────────────────────────────────────────────────────────────────────────
Compare 3 style-transfer models on a subset of musicTI_dataset:
  1. Stylus           (Stable Diffusion 1.5 + mel spectrograms)
  2. StylusAudioLDM2  (AudioLDM2-music + attention injection)
  3. MusicLDM         (MusicLDM + CLAP audio encoder)

For each (content, style) pair, computes:
  - clap_style      : cos(e_out, e_style)
  - clap_content    : cos(e_out, e_content)
  - clap_directional: cos(e_out−e_content, e_style−e_content)
  - combined        : λ·mel_style_score + (1−λ)·mfcc_content_score

Prints per-pair results and a final averaged comparison table.
"""

import os
import sys
import itertools
import traceback

import numpy as np
import librosa
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "stylus"))
sys.path.insert(0, os.path.join(ROOT, "musicLDM"))

# ─── Scoring ─────────────────────────────────────────────────────────────────
from score_clap import clap_scores
from score_combined import combined_score

# ─── Models ──────────────────────────────────────────────────────────────────
from stylus import StylusConfig, StylusPipeline
from stylus_audioldm2_v5 import StylusAudioLDM2Config, StylusAudioLDM2Pipeline
from diffusers import MusicLDMPipeline, DDIMScheduler
import musicldm_style_transferClaude as mldm

# ─────────────────────────────────────────────────────────────────────────────
# Dataset subset
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(ROOT, "musicTI_dataset")

# 3 content × 3 style = 9 pairs (1 file per category)
CONTENT_CATEGORIES = ["hiphop", "violin", "piano"]
STYLE_CATEGORIES = ["harmonica", "bird", "chime"]
FILES_PER_CAT = 1  # first file of each category

EVAL_OUTDIR = os.path.join(ROOT, "evaluation_outputs")

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
STYLUS_ALPHA = 0.9
STYLUS_GAMMA = 0.1
STYLUS_STEPS = 50
STYLUS_DURATION = 5.0  # seconds

AUDIOLDM2_ALPHA = 0.9
AUDIOLDM2_GAMMA = 0.3
AUDIOLDM2_STEPS = 50

MUSICLDM_STRENGTH = 0.7
MUSICLDM_GUIDANCE_SCALE = 10
MUSICLDM_STEPS = 50

LAM = 0.5  # λ for combined_score
SCORE_SR = 22050  # common sample-rate for all scoring functions

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def list_audio_files(directory: str, n: int = 1) -> list[str]:
    """Return the first n audio files (sorted) in a directory."""
    exts = {".wav", ".mp3", ".flac", ".ogg"}
    files = sorted(
        f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in exts
    )
    return [os.path.join(directory, f) for f in files[:n]]


def load_for_scoring(
    path: str, target_sr: int = SCORE_SR, duration: float = 5.0
) -> np.ndarray:
    wav, _ = librosa.load(path, sr=target_sr, mono=True, duration=duration)
    target_len = int(target_sr * duration)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    return wav.astype(np.float32)


def resample_to(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.copy()
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def save_wav(path: str, audio: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.9
    sf.write(path, audio.astype(np.float32), sr)


def score_output(
    output: np.ndarray,
    output_sr: int,
    style_path: str,
    content_path: str,
    duration: float = 5.0,
) -> dict:
    """Resample to SCORE_SR, align lengths, compute all 4 scores."""
    out_rs = resample_to(output, output_sr, SCORE_SR)
    style = load_for_scoring(style_path, SCORE_SR, duration)
    content = load_for_scoring(content_path, SCORE_SR, duration)

    L = min(len(out_rs), len(style), len(content))
    out_rs, style, content = out_rs[:L], style[:L], content[:L]

    clap = clap_scores(out_rs, style, content, SCORE_SR)
    comb = combined_score(out_rs, style, content, SCORE_SR, lam=LAM)

    result = {"combined": comb}
    if clap:
        result.update(
            {
                "clap_style": clap["style"],
                "clap_content": clap["content"],
                "clap_directional": clap["directional"],
            }
        )
    else:
        result.update(
            {
                "clap_style": float("nan"),
                "clap_content": float("nan"),
                "clap_directional": float("nan"),
            }
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Stylus (SD 1.5 + mel spectrograms)
# ─────────────────────────────────────────────────────────────────────────────


def make_stylus_pipeline() -> StylusPipeline:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    cfg = StylusConfig(
        alpha=STYLUS_ALPHA,
        gamma=STYLUS_GAMMA,
        num_inference_steps=STYLUS_STEPS,
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    pipe = StylusPipeline(cfg)
    pipe.load_model()
    return pipe


def run_stylus(
    pipe: StylusPipeline,
    content_path: str,
    style_path: str,
    out_dir: str,
) -> tuple[np.ndarray, int]:
    sr = pipe.cfg.sample_rate

    def _load(p):
        wav, _ = librosa.load(p, sr=sr, mono=True, duration=STYLUS_DURATION)
        target = int(sr * STYLUS_DURATION)
        if len(wav) < target:
            wav = np.pad(wav, (0, target - len(wav)))
        return wav.astype(np.float32)

    style_audio = _load(style_path)
    content_audio = _load(content_path)

    audio_out = pipe.transfer(style_audio, content_audio, save_dir=out_dir)
    save_wav(os.path.join(out_dir, "output.wav"), audio_out, sr)
    return audio_out, sr


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: StylusAudioLDM2
# ─────────────────────────────────────────────────────────────────────────────


def make_audioldm2_pipeline() -> StylusAudioLDM2Pipeline:
    cfg = StylusAudioLDM2Config(
        alpha=AUDIOLDM2_ALPHA,
        gamma=AUDIOLDM2_GAMMA,
        num_inference_steps=AUDIOLDM2_STEPS,
        skip_roundtrip_check=True,
    )
    pipe = StylusAudioLDM2Pipeline(cfg)
    pipe.load_model()
    return pipe


def run_audioldm2(
    pipe: StylusAudioLDM2Pipeline,
    content_path: str,
    style_path: str,
    out_dir: str,
) -> tuple[np.ndarray, int]:
    audio_out = pipe.transfer(
        style_path=style_path,
        content_path=content_path,
        output_dir=out_dir,
    )
    sr = pipe.proc.sample_rate
    return audio_out, sr


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: MusicLDM
# ─────────────────────────────────────────────────────────────────────────────


def make_musicldm_pipeline():
    pipe = MusicLDMPipeline.from_pretrained(
        mldm.MODEL_ID,
        torch_dtype=torch.float32,
    ).to(mldm.DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def run_musicldm(
    pipe,
    content_path: str,
    style_path: str,
    out_dir: str,
) -> tuple[np.ndarray, int]:
    sr = mldm.SAMPLE_RATE
    do_cfg = MUSICLDM_GUIDANCE_SCALE > 1.0

    def _pad(wav):
        if len(wav) < mldm.CHUNK_SAMPLES:
            rep = int(np.ceil(mldm.CHUNK_SAMPLES / len(wav)))
            return np.tile(wav, rep)[: mldm.CHUNK_SAMPLES]
        return wav[: mldm.CHUNK_SAMPLES]

    wav_content = _pad(mldm.load_audio(content_path))
    wav_style = _pad(mldm.load_audio(style_path))

    p_embeds, p_mask = mldm.encode_audio_as_prompt(wav_style, pipe, do_cfg)
    mel = mldm.audio_to_mel(wav_content)
    z0 = mldm.vae_encode(mel, pipe.vae)
    z_n, t_start = mldm.add_noise(z0, pipe.scheduler, MUSICLDM_STRENGTH, MUSICLDM_STEPS)
    z_out = mldm.guided_denoise(
        z_n, t_start, pipe, MUSICLDM_GUIDANCE_SCALE, MUSICLDM_STEPS, p_embeds, p_mask
    )
    wav_out = mldm.vae_decode(z_out, pipe.vae, pipe)

    save_wav(os.path.join(out_dir, "output.wav"), wav_out, sr)
    return wav_out, sr


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

MODELS = ["Stylus", "StylusAudioLDM2", "MusicLDM"]
SCORE_KEYS = ["clap_style", "clap_content", "clap_directional", "combined"]


def print_table(accum: dict):
    col_w = 18
    header = f"\n{'Model':<22}" + "".join(f"{k:>{col_w}}" for k in SCORE_KEYS)
    sep = "─" * (22 + col_w * len(SCORE_KEYS))
    print("\n" + "=" * len(sep))
    print("  COMPARATIVE EVALUATION — Average scores over dataset")
    print("=" * len(sep))
    print(header)
    print(sep)
    for model_name in MODELS:
        row = f"{model_name:<22}"
        for k in SCORE_KEYS:
            vals = [v for v in accum[model_name][k] if not np.isnan(v)]
            mean = np.mean(vals) if vals else float("nan")
            row += f"{mean:>{col_w}.4f}"
        print(row)
    print("=" * len(sep))


def main():
    # ── Build pairs ──────────────────────────────────────────────────────────
    pairs = []
    for content_cat, style_cat in itertools.product(
        CONTENT_CATEGORIES, STYLE_CATEGORIES
    ):
        content_dir = os.path.join(DATASET_ROOT, "content", content_cat)
        style_dir = os.path.join(DATASET_ROOT, "timbre", style_cat)
        c_files = list_audio_files(content_dir, FILES_PER_CAT)
        s_files = list_audio_files(style_dir, FILES_PER_CAT)
        for cf, sf_ in itertools.product(c_files, s_files):
            pairs.append((cf, sf_))

    print(f"\nEvaluating {len(pairs)} pairs × {len(MODELS)} models")
    print("=" * 60)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[LOADING MODELS]")
    print("  1/3  Stylus (SD1.5)...")
    stylus_pipe = make_stylus_pipeline()
    print("  2/3  StylusAudioLDM2...")
    audioldm2_pipe = make_audioldm2_pipeline()
    print("  3/3  MusicLDM...")
    musicldm_pipe = make_musicldm_pipeline()
    print("All models loaded.\n")

    # ── Accumulators ─────────────────────────────────────────────────────────
    accum = {m: {k: [] for k in SCORE_KEYS} for m in MODELS}

    runners = [
        ("Stylus", run_stylus, stylus_pipe),
        ("StylusAudioLDM2", run_audioldm2, audioldm2_pipe),
        ("MusicLDM", run_musicldm, musicldm_pipe),
    ]

    for i, (content_path, style_path) in enumerate(pairs):
        content_name = os.path.splitext(os.path.basename(content_path))[0]
        style_name = os.path.splitext(os.path.basename(style_path))[0]
        pair_tag = f"{content_name}_x_{style_name}"

        print(f"\n{'─' * 60}")
        print(
            f"[Pair {i + 1}/{len(pairs)}]  content={content_name}  style={style_name}"
        )

        for model_name, runner, pipe in runners:
            out_dir = os.path.join(EVAL_OUTDIR, pair_tag, model_name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"  [{model_name}] running...", end=" ", flush=True)
            try:
                audio_out, sr = runner(pipe, content_path, style_path, out_dir)
                scores = score_output(audio_out, sr, style_path, content_path)
                print(
                    f"done | combined={scores['combined']:.3f}  "
                    f"clap_dir={scores['clap_directional']:.3f}"
                )
                for k in SCORE_KEYS:
                    accum[model_name][k].append(scores[k])
            except Exception:
                print("ERROR")
                traceback.print_exc()
                for k in SCORE_KEYS:
                    accum[model_name][k].append(float("nan"))

    # ── Final comparison table ────────────────────────────────────────────────
    print_table(accum)


if __name__ == "__main__":
    main()
