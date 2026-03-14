"""
test_mel_spectro.py

Lance deux transferts de style avec des paramètres extrêmes et compare
les mel-spectrogrammes résultants :

  Config A : alpha=0, gamma=1  → content pur (référence "sans style")
             - alpha=0 : out = out_content (K/V style ignorés)
             - gamma=1 : Q = Q_content (structure maximale)

  Config B : alpha=1, gamma=0  → style pur (injection maximale)
             - alpha=1 : out = out_style  (K/V content ignorés)
             - gamma=0 : Q = Q_current   (query libre, pas de preservation)

Si le pipeline fonctionne, les deux mels doivent être VISUELLEMENT différents :
  - Config A ≈ mel_content (même structure, pas de texture style)
  - Config B ≈ texture du style sur la structure du content

Le script produit :
  ./test_outputs/mel_A_alpha0_gamma1.png
  ./test_outputs/mel_B_alpha1_gamma0.png
  ./test_outputs/mel_comparison_AB.png   ← les deux côte à côte + diff
  ./test_outputs/diff_stats.txt          ← statistiques de différence
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stylus import StylusConfig, StylusPipeline, AudioProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres — adapter les chemins
# ─────────────────────────────────────────────────────────────────────────────

STYLE_PATH   = "bird.wav"
CONTENT_PATH = "color.wav"
OUT_DIR      = "./test_outputs"
DURATION     = 5.0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID     = "runwayml/stable-diffusion-v1-5"
STEPS        = 50


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path, sr, duration):
    audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    target = int(sr * duration)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio.astype(np.float32)


def run_transfer(style_audio, content_audio, alpha, gamma, label, shared_pipeline=None):
    """Lance un transfert et retourne (mel_out, pipeline)."""
    cfg = StylusConfig(
        alpha=alpha,
        gamma=gamma,
        num_inference_steps=STEPS,
        device=DEVICE,
        model_id=MODEL_ID,
        dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
    )
    if shared_pipeline is None:
        pipeline = StylusPipeline(cfg)
        pipeline.load_model()
    else:
        # Réutiliser le modèle chargé, juste mettre à jour alpha/gamma
        pipeline = shared_pipeline
        pipeline.cfg.alpha = alpha
        pipeline.cfg.gamma = gamma
        pipeline.store.alpha = alpha
        pipeline.store.gamma = gamma

    print(f"\n{'='*60}")
    print(f"  Config {label} : alpha={alpha}, gamma={gamma}")
    print(f"{'='*60}")

    save_dir = os.path.join(OUT_DIR, label)
    audio_out = pipeline.transfer(style_audio, content_audio, save_dir=save_dir)

    # Récupérer le mel stylisé depuis le fichier sauvegardé
    # (on relit le mel depuis le PNG sauvegardé par transfer())
    # Plus propre : recalculer depuis l'audio
    proc = AudioProcessor(cfg)
    mel_out, _ = proc.audio_to_mel_and_phase(audio_out)

    # Sauvegarder l'audio
    audio_norm = audio_out / (np.abs(audio_out).max() + 1e-8) * 0.9
    sf.write(os.path.join(OUT_DIR, f"audio_{label}.wav"), audio_norm, cfg.sample_rate)

    return mel_out, pipeline


def save_comparison(mel_c, mel_s, mel_A, mel_B, cfg_audio, out_path):
    """
    4 panneaux : Content | Style | Config A (alpha=0,γ=1) | Config B (alpha=1,γ=0)
    + panneau de différence A-B normalisée
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 9))

    hop = 512
    sr  = 22050
    fmin, fmax = 0.0, 8000.0

    mels = [mel_c, mel_s, mel_A, mel_B]
    vmin = min(m.min() for m in mels)
    vmax = max(m.max() for m in mels)

    panels = [
        (axes[0, 0], mel_c, "Content (référence)"),
        (axes[0, 1], mel_s, "Style (référence)"),
        (axes[0, 2], mel_A, "Config A : α=0, γ=1\n(content pur — sans injection)"),
        (axes[1, 0], mel_B, "Config B : α=1, γ=0\n(style pur — injection maximale)"),
    ]
    for ax, mel, title in panels:
        dur = mel.shape[1] * hop / sr
        im = ax.imshow(mel, origin='lower', aspect='auto', cmap='magma',
                       extent=[0, dur, fmin, fmax],
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Temps (s)"); ax.set_ylabel("Fréquence (Hz)")

    fig.colorbar(im, ax=axes[0, :], format="%+2.0f dB", shrink=0.6, label="dB")

    # Différence absolue A - B
    # Aligner les tailles si nécessaire
    T = min(mel_A.shape[1], mel_B.shape[1])
    diff = np.abs(mel_A[:, :T] - mel_B[:, :T])
    dur_diff = T * hop / sr

    ax_diff = axes[1, 1]
    im_diff = ax_diff.imshow(diff, origin='lower', aspect='auto', cmap='hot',
                              extent=[0, dur_diff, fmin, fmax], interpolation='nearest')
    ax_diff.set_title("Différence |A − B| (dB)\n"
                      "→ zones rouges = l'injection a eu un effet", fontsize=11, fontweight='bold')
    ax_diff.set_xlabel("Temps (s)"); ax_diff.set_ylabel("Fréquence (Hz)")
    fig.colorbar(im_diff, ax=ax_diff, format="%+.1f dB", shrink=0.8, label="|ΔdB|")

    # Stats textuelles
    ax_stats = axes[1, 2]
    ax_stats.axis('off')
    mean_diff = diff.mean()
    max_diff  = diff.max()
    std_diff  = diff.std()

    # Corrélation de Pearson entre A et B (sur les mels aplatis)
    flat_A = mel_A[:, :T].flatten()
    flat_B = mel_B[:, :T].flatten()
    corr = np.corrcoef(flat_A, flat_B)[0, 1]

    # Corrélation A vs content (pour voir si A ≈ content)
    flat_C = mel_c[:, :T].flatten() if mel_c.shape[1] >= T else np.pad(mel_c.flatten(), (0, T*mel_c.shape[0] - len(mel_c.flatten())))
    corr_A_C = np.corrcoef(flat_A, flat_C[:len(flat_A)])[0, 1] if len(flat_A) == len(flat_C) else float('nan')

    stats_text = (
        f"Statistiques de différence |A − B|\n"
        f"{'─'*35}\n"
        f"  Moyenne    : {mean_diff:+.2f} dB\n"
        f"  Max        : {max_diff:+.2f} dB\n"
        f"  Std        : {std_diff:.2f} dB\n\n"
        f"Corrélation de Pearson\n"
        f"{'─'*35}\n"
        f"  A ↔ B      : {corr:.4f}\n"
        f"  A ↔ Content: {corr_A_C:.4f}\n\n"
        f"Interprétation\n"
        f"{'─'*35}\n"
        f"  Si corr(A,B) ≈ 1 → injection sans effet\n"
        f"  Si corr(A,B) < 0.95 → injection visible\n"
        f"  Si corr(A,C) > corr(A,B) → A ≈ content ✓"
    )
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8),
                  color='white')
    ax_stats.set_facecolor('#0a0a1a')

    plt.suptitle("Test d'injection : impact de α et γ sur le mel-spectrogramme",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(fig)
    print(f"\nComparaison sauvegardée : {out_path}")

    return {
        "mean_diff_db": float(mean_diff),
        "max_diff_db":  float(max_diff),
        "std_diff_db":  float(std_diff),
        "corr_A_B":     float(corr),
        "corr_A_content": float(corr_A_C),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Config de base pour charger le modèle une seule fois
    base_cfg = StylusConfig(
        alpha=0.5, gamma=0.8,
        num_inference_steps=STEPS,
        device=DEVICE, model_id=MODEL_ID,
        dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
    )
    proc = AudioProcessor(base_cfg)

    print("Chargement audio...")
    style_audio   = load_audio(STYLE_PATH,   base_cfg.sample_rate, DURATION)
    content_audio = load_audio(CONTENT_PATH, base_cfg.sample_rate, DURATION)

    mel_s, _ = proc.audio_to_mel_and_phase(style_audio)
    mel_c, _ = proc.audio_to_mel_and_phase(content_audio)

    # Charger le modèle une seule fois
    pipeline = StylusPipeline(base_cfg)
    pipeline.load_model()

    # ── Config A : alpha=0, gamma=1 (content pur) ─────────────────────────────
    mel_A, pipeline = run_transfer(
        style_audio, content_audio,
        alpha=0.0, gamma=1.0, label="A_alpha0_gamma1",
        shared_pipeline=pipeline,
    )

    # ── Config B : alpha=1, gamma=0 (style pur) ───────────────────────────────
    mel_B, pipeline = run_transfer(
        style_audio, content_audio,
        alpha=1.0, gamma=0.0, label="B_alpha1_gamma0",
        shared_pipeline=pipeline,
    )

    # ── Comparaison visuelle ───────────────────────────────────────────────────
    stats = save_comparison(
        mel_c, mel_s, mel_A, mel_B,
        cfg_audio=base_cfg,
        out_path=os.path.join(OUT_DIR, "mel_comparison_AB.png"),
    )

    # ── Stats textuelles ───────────────────────────────────────────────────────
    stats_path = os.path.join(OUT_DIR, "diff_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Test alpha/gamma - Statistiques de difference\n")
        f.write("="*40 + "\n\n")
        f.write("Config A : alpha=0.0, gamma=1.0  (content pur)\n")
        f.write("Config B : alpha=1.0, gamma=0.0  (style pur)\n\n")
        for k, v in stats.items():
            f.write(f"  {k:25s}: {v:.4f}\n")
        f.write("\nInterpretation :\n")
        if stats["corr_A_B"] > 0.98:
            f.write("  [!] corr(A,B) > 0.98 -> injection quasi sans effet\n")
            f.write("      Piste : verifier que K/V sont bien captures\n")
            f.write("      (store._ks non vide apres inversion style ?)\n")
        elif stats["corr_A_B"] > 0.95:
            f.write("  [~] corr(A,B) in [0.95, 0.98] -> effet faible\n")
            f.write("      L'injection a un impact mais limite par le VAE SD1.5\n")
        else:
            f.write("  [OK] corr(A,B) < 0.95 -> injection clairement visible\n")

    print(f"Stats sauvegardées : {stats_path}")
    print("\n── Résumé ──────────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:25s}: {v:.4f}")

    # Diagnostic supplémentaire : vérifier que le store n'est pas vide
    print("\n── Diagnostic store ────────────────────────────")
    print(f"  K style stockés  : {len(pipeline.store._ks)} entrées")
    print(f"  V style stockés  : {len(pipeline.store._vs)} entrées")
    print(f"  Q content stockés: {len(pipeline.store._qs)} entrées")
    if len(pipeline.store._ks) == 0:
        print("  ⚠ STORE VIDE — les features ne sont pas capturées !")
        print("    Vérifier que _install_processors() a trouvé des couches.")


if __name__ == "__main__":
    main()
