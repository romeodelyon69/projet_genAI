"""
Test-time scaling analysis: Best-of-N on audio style transfer.

Workflow:
  1. generate_candidates()   — generates 64 audios (grid 8α × 8γ) and computes all scores
  2. bon_scaling_analysis()  — for N ∈ [1,2,4,8,16,32], estimates E[best-of-N] by Monte Carlo
  3. plot_bon_curve()        — plots and saves the curve

Selectable score via score_key:
  "combined"         → λ·Mel_style + (1-λ)·MFCC_content  (score_combined.py)
  "clap_directional" → cos(e_out - e_content, e_style - e_content)  ← main metric
  "clap_style"       → cos(e_out, e_style)
  "clap_content"     → cos(e_out, e_content)
"""

import os
import itertools
import numpy as np
import soundfile as sf

from score_combined import combined_score
from score_clap import clap_scores


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def save_audio(path: str, audio: np.ndarray, sr: int):
    audio_norm = audio / (np.abs(audio).max() + 1e-8) * 0.9
    sf.write(path, audio_norm, sr)


def _extract_score(result: dict, score_key: str) -> float | None:
    """Extracts the scalar score from a result given a score key."""
    if score_key == "combined":
        return result["combined"]
    elif score_key.startswith("clap_"):
        sub = score_key[len("clap_") :]  # "directional", "style" ou "content"
        clap = result["clap"]
        return clap[sub] if clap is not None else None
    else:
        raise ValueError(
            f"score_key '{score_key}' invalide. "
            "Choisir parmi : 'combined', 'clap_directional', 'clap_style', 'clap_content'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generating 64 candidates
# ─────────────────────────────────────────────────────────────────────────────


def generate_candidates(
    pipeline,
    style_audio: np.ndarray,
    content_audio: np.ndarray,
    save_dir: str = "./bon_candidates",
    lam: float = 0.5,
    alphas: list | None = None,
    gammas: list | None = None,
) -> list[dict]:
    """
    Generates stylized audios over an α × γ grid.
    Computes combined_score and clap_scores for each candidate.

    Parameters
    ----------
    pipeline    : already loaded StylusPipeline (load_model() called)
    alphas      : style guidance values (default: linspace(0.25, 1.0, 8))
    gammas      : query preservation values (default: linspace(0.02, 0.30, 8))
    lam         : λ for combined_score

    Returns a list of 64 dicts:
      { alpha, gamma, wav_path, combined, clap }
      (audio is NOT kept in memory to save RAM)
    """
    if alphas is None:
        alphas = list(np.linspace(0.25, 1.0, 8).round(3))
    if gammas is None:
        gammas = list(np.linspace(0.02, 0.30, 8).round(3))

    os.makedirs(save_dir, exist_ok=True)
    sr = pipeline.cfg.sample_rate
    grid = list(itertools.product(alphas, gammas))

    print(f"\n{'═' * 62}")
    print(
        f"  Génération de {len(grid)} candidats (grille {len(alphas)}α × {len(gammas)}γ)"
    )
    print(f"{'═' * 62}\n")

    results = []

    for i, (alpha, gamma) in enumerate(grid):
        print(
            f"  [{i + 1:02d}/{len(grid)}] α={alpha:.3f}  γ={gamma:.3f}",
            end="  ",
            flush=True,
        )

        pipeline.cfg.alpha = alpha
        pipeline.cfg.gamma = gamma
        pipeline.store.alpha = alpha
        pipeline.store.gamma = gamma

        audio = pipeline.transfer(style_audio, content_audio, save_dir=None)

        wav_path = os.path.join(
            save_dir, f"cand_{i + 1:02d}_a{alpha:.3f}_g{gamma:.3f}.wav"
        )
        save_audio(wav_path, audio, sr)

        c_score = combined_score(audio, style_audio, content_audio, sr, lam)
        clap = clap_scores(audio, style_audio, content_audio, sr)

        results.append(
            {
                "idx": i,
                "alpha": alpha,
                "gamma": gamma,
                "wav_path": wav_path,
                "combined": c_score,
                "clap": clap,
            }
        )

        clap_dir_str = f"{clap['directional']:.4f}" if clap else "N/A"
        print(f"combined={c_score:.4f}  clap_dir={clap_dir_str}")

    print(f"\n{'═' * 62}")
    print(f"  {len(results)} candidats générés → {save_dir}")
    print(f"{'═' * 62}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Best-of-N analysis (Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────


def bon_scaling_analysis(
    candidates: list[dict],
    score_key: str = "combined",
    ns: list[int] | None = None,
    n_trials: int = 200,
    rng_seed: int = 0,
) -> dict:
    """
    For each N ∈ ns, estimates E[best-of-N] by Monte Carlo:
      - Repeats n_trials times: draws N random candidates, takes the max of their scores.
      - Mean and standard deviation over trials.

    Parameters
    ----------
    candidates  : list returned by generate_candidates()
    score_key   : 'combined' | 'clap_directional' | 'clap_style' | 'clap_content'
    ns          : N values to analyse (default: [1,2,4,8,16,32])
    n_trials    : number of Monte Carlo draws per N value
    rng_seed    : random seed for reproducibility

    Returns a dict:
      {
        N: {"mean": float, "std": float, "trials": list[float]}
        for N in ns
      }
    """
    if ns is None:
        ns = [1, 2, 4, 8, 16, 32]

    # Extraire tous les scores scalaires (on ignore les candidats sans score)
    scored = []
    for r in candidates:
        s = _extract_score(r, score_key)
        if s is not None:
            scored.append(s)

    if not scored:
        raise RuntimeError(
            f"Aucun candidat n'a un score valide pour score_key='{score_key}'. "
            "Vérifiez que CLAP est installé si vous utilisez un score CLAP."
        )

    scores_arr = np.array(scored, dtype=float)
    rng = np.random.default_rng(rng_seed)

    print(f"\n{'═' * 62}")
    print(f"  Analyse Best-of-N  |  score={score_key}  |  {len(scores_arr)} candidats")
    print(f"  n_trials={n_trials}  |  ns={ns}")
    print(f"{'─' * 62}")
    print(f"  {'N':>4}  {'E[best-of-N]':>14}  {'std':>10}")
    print(f"{'─' * 62}")

    analysis = {}

    for n in ns:
        if n > len(scores_arr):
            print(f"  {n:>4}  (N > nombre de candidats, ignoré)")
            continue

        trials = [
            float(np.max(rng.choice(scores_arr, size=n, replace=False)))
            for _ in range(n_trials)
        ]
        mean_val = float(np.mean(trials))
        std_val = float(np.std(trials))

        analysis[n] = {"mean": mean_val, "std": std_val, "trials": trials}
        print(f"  {n:>4}  {mean_val:>14.4f}  {std_val:>10.4f}")

    print(f"{'═' * 62}\n")
    return analysis


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot
# ─────────────────────────────────────────────────────────────────────────────


def plot_bon_curve(
    analysis: dict,
    score_key: str = "combined",
    save_path: str = "bon_curve.png",
):
    """
    Plots E[best-of-N] ± std as a function of N (log-2 scale on x-axis).
    Saves the figure and displays it if a display is available.

    Parameters
    ----------
    analysis   : dict returned by bon_scaling_analysis()
    score_key  : used for the title and Y-axis label
    save_path  : figure save path
    """
    import matplotlib.pyplot as plt

    ns = sorted(analysis.keys())
    means = [analysis[n]["mean"] for n in ns]
    stds = [analysis[n]["std"] for n in ns]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(ns, means, marker="o", linewidth=2, color="#2563EB", label="E[best-of-N]")
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
    ax.set_xlabel("N  (number of candidates per group)", fontsize=12)
    ax.set_ylabel(f"Mean best-of-N score\n({score_key})", fontsize=12)
    ax.set_title(f"Test-Time Scaling — Best-of-N\n(score: {score_key})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Figure saved: {save_path}")

    try:
        plt.show()
    except Exception:
        pass  # pas de display (serveur headless)
