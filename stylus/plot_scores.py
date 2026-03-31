"""
Visualization of scores on the α × γ grid.

Usage:
    python plot_scores.py <candidates_folder> <style.wav> <content.wav> [--lam 0.5]

The folder must contain files named according to generate_candidates() convention:
    cand_{i:02d}_a{alpha:.3f}_g{gamma:.3f}.wav

Produces a figure with 4 heatmaps:
    - combined_score
    - clap_style_score
    - clap_content_score
    - clap_directional_score
"""

import os
import re
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import soundfile as sf

from score_combined import combined_score
from scores_clap import clap_scores


# ─────────────────────────────────────────────────────────────────────────────
# Folder parsing
# ─────────────────────────────────────────────────────────────────────────────

_FOLDER_RE = re.compile(r"^a([\d.]+)_g([\d.]+)$")


def parse_candidates(folder: str) -> list[dict]:
    """
    Reads subfolders named a{alpha}_g{gamma}/ and finds the stylized_*.wav
    file inside each one (layout produced by stylus_audioldm2_v4.py).
    Also falls back to legacy cand_*.wav files at the folder root.
    """
    entries = []
    for name in sorted(os.listdir(folder)):
        subdir = os.path.join(folder, name)
        # New layout: a{alpha}_g{gamma}/ subfolders
        m = _FOLDER_RE.match(name)
        if m and os.path.isdir(subdir):
            wavs = [
                f
                for f in os.listdir(subdir)
                if f.startswith("stylized_") and f.endswith(".wav")
            ]
            if wavs:
                entries.append(
                    {
                        "path": os.path.join(subdir, sorted(wavs)[0]),
                        "alpha": float(m.group(1)),
                        "gamma": float(m.group(2)),
                    }
                )
        # Legacy layout: cand_NN_a{alpha}_g{gamma}.wav at root
        elif re.match(r"cand_\d+_a([\d.]+)_g([\d.]+)\.wav$", name):
            mg = re.match(r"cand_\d+_a([\d.]+)_g([\d.]+)\.wav$", name)
            entries.append(
                {
                    "path": os.path.join(folder, name),
                    "alpha": float(mg.group(1)),
                    "gamma": float(mg.group(2)),
                }
            )
    if not entries:
        raise FileNotFoundError(
            f"Aucun fichier candidat trouvé dans '{folder}'. "
            "Attendu : sous-dossiers a{{alpha}}_g{{gamma}}/ avec stylized_*.wav, "
            "ou fichiers cand_*.wav à la racine."
        )
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Score computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_all_scores(
    entries: list[dict],
    style_audio: np.ndarray,
    content_audio: np.ndarray,
    sr: int,
    lam: float = 0.5,
) -> list[dict]:
    """
    For each entry, loads the audio and computes the 4 scores.
    Returns the enriched list with scores.
    """
    for i, e in enumerate(entries):
        print(
            f"  [{i + 1:02d}/{len(entries)}] scoring {os.path.basename(e['path'])}",
            flush=True,
        )
        audio, _ = sf.read(e["path"])
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        e["combined"] = combined_score(audio, style_audio, content_audio, sr, lam)

        clap = clap_scores(audio, style_audio, content_audio, sr)
        if clap is not None:
            e["clap_style"] = clap["style"]
            e["clap_content"] = clap["content"]
            e["clap_directional"] = clap["directional"]
        else:
            e["clap_style"] = np.nan
            e["clap_content"] = np.nan
            e["clap_directional"] = np.nan

        print(
            f"         combined={e['combined']:.4f}  "
            f"clap_dir={e['clap_directional']:.4f}  "
            f"clap_sty={e['clap_style']:.4f}  "
            f"clap_cnt={e['clap_content']:.4f}"
        )
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# α × γ matrix construction
# ─────────────────────────────────────────────────────────────────────────────


def build_grid(
    entries: list[dict], score_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds an (n_alpha × n_gamma) matrix for the requested score.
    Returns (alphas_sorted, gammas_sorted, matrix).
    """
    alphas = sorted(set(e["alpha"] for e in entries))
    gammas = sorted(set(e["gamma"] for e in entries))

    a_idx = {a: i for i, a in enumerate(alphas)}
    g_idx = {g: i for i, g in enumerate(gammas)}

    mat = np.full((len(alphas), len(gammas)), np.nan)
    for e in entries:
        mat[a_idx[e["alpha"]], g_idx[e["gamma"]]] = e[score_key]

    return np.array(alphas), np.array(gammas), mat


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────


def plot_heatmaps(
    entries: list[dict],
    save_path: str | None = None,
):
    score_keys = [
        ("combined", "Mel + MFCC score\n(λ·Mel_style + (1-λ)·MFCC_timbre)"),
        ("clap_style", "CLAP style score"),
        ("clap_content", "CLAP content score"),
        (
            "clap_directional",
            "CLAP directional score",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, score_keys):
        alphas, gammas, mat = build_grid(entries, key)

        # Masked array pour NaN (CLAP non disponible)
        masked = np.ma.masked_invalid(mat)
        vmin, vmax = np.nanmin(mat), np.nanmax(mat)

        im = ax.imshow(
            masked,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="RdYlGn",
            interpolation="nearest",
        )

        # Axes labels
        ax.set_xticks(range(len(gammas)))
        ax.set_xticklabels(
            [f"{g:.3f}" for g in gammas], rotation=45, ha="right", fontsize=8
        )
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"{a:.3f}" for a in alphas], fontsize=8)
        ax.set_xlabel("γ  (query preservation)", fontsize=10)
        ax.set_ylabel("α  (style guidance)", fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)

        # Values in cells
        for i in range(len(alphas)):
            for j in range(len(gammas)):
                val = mat[i, j]
                if not np.isnan(val):
                    # Text color based on cell brightness
                    norm_val = (val - vmin) / (vmax - vmin + 1e-8)
                    txt_color = (
                        "black"
                        if 0.25 < norm_val < 0.75
                        else "white"
                        if norm_val <= 0.25
                        else "black"
                    )
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=txt_color,
                    )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Marquer le meilleur
        best_i, best_j = np.unravel_index(np.nanargmax(mat), mat.shape)
        ax.add_patch(
            plt.Rectangle(
                (best_j - 0.5, best_i - 0.5),
                1,
                1,
                linewidth=2.5,
                edgecolor="blue",
                facecolor="none",
            )
        )

    fig.suptitle("Grid search α × γ — Style transfer scores", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.30, wspace=0.22)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved: {save_path}")

    try:
        plt.show()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Paramètres ────────────────────────────────────────────────────────────
    folder = "./stylus-audio_LDM2_outputs/grid_search_output"  # folder generated by generate_candidates()
    style = "musicTI_dataset/audios/timbre/chime/chime1.wav"
    content = "musicTI_dataset/audios/content/violin/violin1.wav"
    lam = 0.5  # λ for combined_score
    sr = 22050
    duration = 5.0
    out = None  # None → sauvegarde dans folder/scores_heatmap.png

    # ── Loading references ──────────────────────────────────────────────────
    print(f"Loading style  : {style}")
    print(f"Loading content: {content}")
    style_audio, _ = librosa.load(style, sr=sr, mono=True, duration=duration)
    content_audio, _ = librosa.load(content, sr=sr, mono=True, duration=duration)
    style_audio = style_audio.astype(np.float32)
    content_audio = content_audio.astype(np.float32)

    # ── Parse + score ──────────────────────────────────────────────────────
    print(f"\nParsing candidates in : {folder}")
    entries = parse_candidates(folder)
    print(f"  {len(entries)} candidats trouvés\n")

    print("Computing scores...")
    entries = compute_all_scores(entries, style_audio, content_audio, sr, lam)

    # ── Plot ───────────────────────────────────────────────────────────────
    out_path = out or os.path.join(folder, "scores_heatmap.png")
    plot_heatmaps(entries, save_path=out_path)


if __name__ == "__main__":
    main()
