"""
Best-of-N analysis for MusicLDM style transfer.

MusicLDM is stochastic (random noise in add_noise) → each run with the same
inputs produces a different output. This makes Best-of-N meaningful:
generate N samples, keep the best one according to a score function.

Workflow
--------
1. generate_samples(N=64)  — runs the MusicLDM pipeline N times, saves WAVs,
                             computes combined_score + clap_scores for each.
2. bon_scaling_analysis()  — Monte Carlo estimate of E[best-of-N] for each
                             score key and N ∈ [1,2,4,8,16,32,64].
3. plot_bon_curves()       — one subplot per score key, E[best-of-N] vs N.

Score keys
----------
  "combined"         → λ·Mel_style + (1-λ)·MFCC_content  (spectral, no ML)
  "clap_style"       → cos(e_out, e_style)
  "clap_content"     → cos(e_out, e_content)
  "clap_directional" → cos(e_out−e_content, e_style−e_content)  ← main metric

Usage
-----
Edit the variables at the bottom of this file and run:
    python3 musicLDM/best_of_n.py
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import torch

# ── make sure musicldm_style_transferClaude functions are importable ─────────
sys.path.insert(0, os.path.dirname(__file__))
from musicldm_style_transferClaude import (
    SAMPLE_RATE,
    CHUNK_SAMPLES,
    CHUNK_SECONDS,
    CLAP_SR,
    DEVICE,
    load_audio,
    audio_to_mel,
    vae_encode,
    vae_decode,
    encode_audio_as_prompt,
    add_noise,
    guided_denoise,
)

# ── score functions from stylus/ (shared) ────────────────────────────────────
STYLUS_DIR = os.path.join(os.path.dirname(__file__), "..", "stylus")
sys.path.insert(0, os.path.abspath(STYLUS_DIR))
from score_combined import combined_score
from score_clap import clap_scores

from diffusers import MusicLDMPipeline, DDIMScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SCORE_KEYS = ["combined", "clap_style", "clap_content", "clap_directional"]


def _loop_to_10s(wav: np.ndarray) -> np.ndarray:
    if len(wav) < CHUNK_SAMPLES:
        repeats = int(np.ceil(CHUNK_SAMPLES / len(wav)))
        wav = np.tile(wav, repeats)
    return wav[:CHUNK_SAMPLES]


def _extract_score(result: dict, score_key: str) -> float | None:
    if score_key == "combined":
        return result["combined"]
    sub = score_key[len("clap_") :]  # "style", "content", "directional"
    clap = result["clap"]
    return clap[sub] if clap is not None else None


def _load_pipe(model_id: str = "ucsd-reach/musicldm") -> MusicLDMPipeline:
    pipe = MusicLDMPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(
        DEVICE
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def _run_one(
    pipe,
    z0: torch.Tensor,
    p_embeds,
    p_mask,
    guidance_scale: float,
    strength: float,
    n_steps: int,
) -> np.ndarray:
    """Run one stochastic forward+reverse pass and return audio (np.ndarray)."""
    z_n, t_start = add_noise(z0, pipe.scheduler, strength, n_steps)
    z_out = guided_denoise(
        z_n, t_start, pipe, guidance_scale, n_steps, p_embeds, p_mask
    )
    return vae_decode(z_out, pipe.vae, pipe)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate N samples
# ─────────────────────────────────────────────────────────────────────────────


def generate_samples(
    content_path: str,
    style_path: str,
    save_dir: str = "./bon_musicldm",
    n_samples: int = 64,
    strength: float = 0.7,
    guidance_scale: float = 10.0,
    n_steps: int = 100,
    lam: float = 0.5,
    model_id: str = "ucsd-reach/musicldm",
) -> list[dict]:
    """
    Generate n_samples audio outputs from MusicLDM (stochastic = different each run).
    Each sample is scored with combined_score and clap_scores.

    Returns a list of dicts:
        { idx, wav_path, combined, clap }
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'═' * 62}")
    print(f"  MusicLDM Best-of-N  |  N={n_samples}")
    print(f"  content : {content_path}")
    print(f"  style   : {style_path}")
    print(f"  strength={strength}  guidance={guidance_scale}  steps={n_steps}")
    print(f"{'═' * 62}\n")

    # Load model once
    print("[1/4] Loading MusicLDM...")
    pipe = _load_pipe(model_id)
    vae = pipe.vae
    print("      Done.\n")

    # Load and prepare audio
    print("[2/4] Loading audio...")
    wav_content = _loop_to_10s(load_audio(content_path))
    wav_style = _loop_to_10s(load_audio(style_path))
    sf.write(
        os.path.join(save_dir, "00_content.wav"),
        wav_content / (np.abs(wav_content).max() + 1e-8),
        SAMPLE_RATE,
    )
    sf.write(
        os.path.join(save_dir, "00_style.wav"),
        wav_style / (np.abs(wav_style).max() + 1e-8),
        SAMPLE_RATE,
    )
    print(
        f"      content: {len(wav_content) / SAMPLE_RATE:.1f}s  "
        f"style: {len(wav_style) / SAMPLE_RATE:.1f}s\n"
    )

    # Pre-compute fixed quantities (computed once, reused N times)
    print("[3/4] Pre-computing fixed latents and style embedding...")
    mel = audio_to_mel(wav_content)
    z0 = vae_encode(mel, vae)

    do_cfg = guidance_scale > 1.0
    p_embeds, p_mask = encode_audio_as_prompt(wav_style, pipe, do_cfg)
    print(f"      Style embedding shape: {p_embeds.shape}\n")

    # Generate N samples
    print(f"[4/4] Generating {n_samples} samples...")
    results = []
    for i in range(n_samples):
        print(f"  [{i + 1:02d}/{n_samples}]", end="  ", flush=True)

        wav_out = _run_one(
            pipe, z0, p_embeds, p_mask, guidance_scale, strength, n_steps
        )

        wav_path = os.path.join(save_dir, f"sample_{i + 1:02d}.wav")
        wav_norm = wav_out / (np.abs(wav_out).max() + 1e-8) * 0.9
        sf.write(wav_path, wav_norm, SAMPLE_RATE)

        c_score = combined_score(wav_out, wav_style, wav_content, SAMPLE_RATE, lam)
        clap = clap_scores(wav_out, wav_style, wav_content, SAMPLE_RATE)

        results.append(
            {
                "idx": i,
                "wav_path": wav_path,
                "combined": c_score,
                "clap": clap,
            }
        )

        clap_dir = f"{clap['directional']:.4f}" if clap else "N/A"
        print(f"combined={c_score:.4f}  clap_dir={clap_dir}")

    print(f"\n{'═' * 62}")
    print(f"  {n_samples} samples saved to {save_dir}/")
    print(f"{'═' * 62}\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Best-of-N Monte Carlo analysis
# ─────────────────────────────────────────────────────────────────────────────


def bon_scaling_analysis(
    results: list[dict],
    score_key: str = "combined",
    ns: list[int] | None = None,
    n_trials: int = 500,
    rng_seed: int = 0,
) -> dict:
    """
    For each N in ns, estimate E[best-of-N] by Monte Carlo:
      draw N samples (without replacement), take the max score, repeat n_trials times.

    Returns { N: {"mean": float, "std": float} }
    """
    if ns is None:
        ns = [1, 2, 4, 8, 16, 32, 64]

    scores_arr = np.array(
        [s for r in results if (s := _extract_score(r, score_key)) is not None],
        dtype=float,
    )

    if len(scores_arr) == 0:
        raise RuntimeError(
            f"No valid scores for key '{score_key}'. "
            "Is CLAP installed? (pip install laion-clap)"
        )

    rng = np.random.default_rng(rng_seed)

    print(
        f"  Best-of-N  |  score={score_key}  |  {len(scores_arr)} samples  "
        f"|  n_trials={n_trials}"
    )
    print(f"  {'N':>4}  {'E[best-of-N]':>14}  {'std':>10}")
    print(f"  {'-' * 32}")

    analysis = {}
    for n in ns:
        if n > len(scores_arr):
            print(f"  {n:>4}  (N > n_samples, skipped)")
            continue
        trials = [
            float(np.max(rng.choice(scores_arr, size=n, replace=False)))
            for _ in range(n_trials)
        ]
        m, s = float(np.mean(trials)), float(np.std(trials))
        analysis[n] = {"mean": m, "std": s, "trials": trials}
        print(f"  {n:>4}  {m:>14.4f}  {s:>10.4f}")

    return analysis


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot
# ─────────────────────────────────────────────────────────────────────────────


def plot_bon_curves(
    all_analyses: dict[str, dict],
    save_path: str = "bon_curves.png",
):
    """
    One subplot per score key.
    all_analyses = { score_key: bon_scaling_analysis(...) }
    """
    import matplotlib.pyplot as plt

    keys = [k for k in SCORE_KEYS if k in all_analyses and all_analyses[k]]
    if not keys:
        print("No data to plot.")
        return

    n_cols = 2
    n_rows = int(np.ceil(len(keys) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False
    )

    for ax, key in zip(axes.flat, keys):
        analysis = all_analyses[key]
        ns = sorted(analysis.keys())
        means = [analysis[n]["mean"] for n in ns]
        stds = [analysis[n]["std"] for n in ns]

        ax.plot(
            ns, means, marker="o", linewidth=2, color="#2563EB", label="E[best-of-N]"
        )
        ax.fill_between(
            ns,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2,
            color="#2563EB",
            label="±1 std",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("N  (number of samples)", fontsize=11)
        ax.set_ylabel("E[best-of-N]", fontsize=11)
        ax.set_title(f"Best-of-N — {key}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Hide unused subplots
    for ax in axes.flat[len(keys) :]:
        ax.set_visible(False)

    fig.suptitle("MusicLDM — Test-Time Scaling (Best-of-N)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {save_path}")

    try:
        plt.show()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Parameters ───────────────────────────────────────────────────────────
    content_path = "./music4.wav"
    style_path = "./musicTI_dataset/timbre/bird/bird1.wav"
    save_dir = "./bon_musicldm"
    n_samples = 64  # number of stochastic samples to generate
    strength = 0.7  # 0.4 = stay close to content, 0.9 = heavy restyle
    guidance_scale = 10.0
    n_steps = 100
    lam = 0.5  # weight for combined_score
    ns = [1, 2, 4, 8, 16, 32, 64]
    n_trials = 500  # Monte Carlo trials per N

    # ── Run ──────────────────────────────────────────────────────────────────
    results = generate_samples(
        content_path=content_path,
        style_path=style_path,
        save_dir=save_dir,
        n_samples=n_samples,
        strength=strength,
        guidance_scale=guidance_scale,
        n_steps=n_steps,
        lam=lam,
    )

    # ── Best-of-N analysis for all score keys ────────────────────────────────
    print(f"\n{'═' * 62}")
    print("  Best-of-N scaling analysis")
    print(f"{'═' * 62}")

    all_analyses = {}
    for key in SCORE_KEYS:
        print()
        try:
            all_analyses[key] = bon_scaling_analysis(
                results, score_key=key, ns=ns, n_trials=n_trials
            )
        except RuntimeError as e:
            print(f"  Skipped {key}: {e}")
            all_analyses[key] = {}

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_path = os.path.join(save_dir, "bon_curves.png")
    plot_bon_curves(all_analyses, save_path=plot_path)
