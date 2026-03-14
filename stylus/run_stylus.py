import os
import numpy as np
import torch
import soundfile as sf
import librosa

from stylus import StylusConfig, StylusPipeline
from best_of_n import generate_candidates


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────────────────────


def load_audio(path: str, target_sr: int, duration: float = 5.0) -> np.ndarray:
    audio, sr = librosa.load(path, sr=target_sr, mono=True, duration=duration)
    target_len = int(target_sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    return audio.astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sr: int):
    audio_norm = audio / (np.abs(audio).max() + 1e-8) * 0.9
    sf.write(path, audio_norm, sr)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Fichiers audio ────────────────────────────────────────────────────────
    style = "musicTI_dataset/audios/timbre/harmonica/harmonica2.wav"
    content = "musicTI_dataset/audios/content/hiphop/hiphop1.wav"

    dir = style.split("/")[-1].split(".")[0] + "_" + content.split("/")[-1].split(".")[0]
    save_dir = "./test_outputs/" + dir + "/" + "grid_search"
    duration = 5.0
    
    # Ajustement proportionnel de la taille de l'image (512px pour 5s)
    target_length = int((duration / 5.0) * 512)
    target_length = (target_length // 64) * 64  # Doit être un multiple de 64

    # ── Scores ────────────────────────────────────────────────────────────────
    lam = 0.5  # λ pour combined_score

    # ── Grille 8×8 = 64 candidats ─────────────────────────────────────────────
    alphas = list(np.linspace(0.25, 1.0, 6).round(3))  # style guidance
    gammas = list(np.linspace(0.02, 0.40, 6).round(3))  # query preservation

    # ── Config modèle ─────────────────────────────────────────────────────────
    steps = 50
    up_blocks = [2, 3]
    model_id = "runwayml/stable-diffusion-v1-5"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    fp32 = False

    cfg = StylusConfig(
        alpha=alphas[0],
        gamma=gammas[0],
        num_inference_steps=steps,
        target_up_block_indices=up_blocks,
        model_id=model_id,
        device=device,
        dtype=torch.float32 if fp32 or device == "cpu" else torch.float16,
        target_length=target_length,
    )

    # ── Chargement audio ──────────────────────────────────────────────────────
    print(f"Loading style  : {style}")
    print(f"Loading content: {content}")
    style_audio = load_audio(style, cfg.sample_rate, duration)
    content_audio = load_audio(content, cfg.sample_rate, duration)

    # ── Pipeline (modèle chargé une seule fois) ───────────────────────────────
    pipeline = StylusPipeline(cfg)
    pipeline.load_model()

    # ── Grid search sur 64 candidats (α × γ) ─────────────────────────────────
    candidates = generate_candidates(
        pipeline,
        style_audio,
        content_audio,
        save_dir=save_dir,
        lam=lam,
        alphas=alphas,
        gammas=gammas,
    )


if __name__ == "__main__":
    main()
