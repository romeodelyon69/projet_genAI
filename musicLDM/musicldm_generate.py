"""
MusicLDM — Génération de musique depuis un prompt texte
--------------------------------------------------------
Usage:
    python musicldm_generate.py --prompt "jazz piano, soft, nocturne" --duration 10
"""

import argparse
import os
import torch
import soundfile as sf
import numpy as np

from diffusers import MusicLDMPipeline

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ucsd-reach/musicldm"


def save_audio(path: str, wav, sr: int):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float().numpy()
    wav = np.squeeze(wav)
    wav = wav / (np.abs(wav).max() + 1e-8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, wav, sr)
    print(f"  → {path}")


def main(prompt: str, output_path: str, duration: float,
         guidance_scale: float, n_steps: int, n_candidates: int):

    print(f"\n{'='*55}")
    print(f"  MusicLDM — Génération")
    print(f"  Prompt   : \"{prompt}\"")
    print(f"  Durée    : {duration}s")
    print(f"  Guidance : {guidance_scale}  |  Steps : {n_steps}")
    print(f"  Device   : {DEVICE}")
    print(f"{'='*55}\n")

    print("[1/2] Chargement MusicLDM...")
    pipe = MusicLDMPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    print("      Modèle chargé.\n")

    print(f"[2/2] Génération ({n_candidates} candidat(s))...")
    output = pipe(
        prompt                   = prompt,
        audio_length_in_s        = duration,
        guidance_scale           = guidance_scale,
        num_inference_steps      = n_steps,
        num_waveforms_per_prompt = n_candidates,
    )

    sr = pipe.vocoder.config.sampling_rate
    audios = output.audios   # [n_candidates, T]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base, ext = os.path.splitext(output_path)
    ext = ext or ".wav"

    for i, wav in enumerate(audios):
        path = f"{base}_{i+1}{ext}" if n_candidates > 1 else f"{base}{ext}"
        save_audio(path, wav, sr)

    print(f"\n  Génération terminée — SR: {sr}Hz")
    print(f"""
  Conseils :
    --guidance  2-4  → plus créatif, moins fidèle au prompt
                7-10 → très fidèle au prompt
    --duration  5-30 → MusicLDM est entraîné sur des segments courts
    --candidates 3   → génère 3 variantes, choisir la meilleure
""")


if __name__ == "__main__":
    prompt = "melodic, piano"  # prompt par défaut
    output_path = "./generated/output.wav"  # chemin de sortie par défaut
    duration = 15.0  # durée par défaut
    guidance_scale = 7.0  # guidance scale par défaut
    n_steps = 50  # nombre de steps par défaut
    n_candidates = 1  # nombre de candidats par défaut
    main(prompt, output_path, duration,
         guidance_scale, n_steps, n_candidates)
