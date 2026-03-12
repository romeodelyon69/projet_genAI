"""
AudioLDM2 — Encode & Decode test (v3 - correct)
-------------------------------------------------
Stratégie :
  - On génère un audio "de référence" via la pipeline normale (texte → audio)
  - On encode ce latent proprement sorti du scheduler → VAE decode → waveform
  - On compare avec la sortie originale de la pipeline
 
Pourquoi ce changement ?
  Le problème des versions précédentes : on construisait notre propre
  MelSpectrogram avec torchaudio, mais le VAE d'AudioLDM2 n'a jamais
  été entraîné sur ce type d'entrée. Il attend des latents produits par
  son propre scheduler de diffusion, pas un Mel encodé manuellement.
 
  La bonne façon de tester le VAE seul : intercepter les latents
  produits par le UNet (déjà débruitées), puis les passer au VAE decode.
 
Usage:
    python audioldm2_encode_decode.py --input chemin/vers/musique.wav
 
Dépendances:
    pip install diffusers transformers accelerate soundfile librosa torch
"""
 
import argparse
import os
import torch
import numpy as np
import soundfile as sf
import librosa
 
from diffusers import AudioLDM2Pipeline
 
 
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_ID    = "cvssp/audioldm2-music"
SAMPLE_RATE = 16000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
 
 
def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=target_sr, mono=True)
    max_samples = target_sr * 10
    if len(waveform) > max_samples:
        print(f"      ⚠ Troncature à 10s pour le test")
        waveform = waveform[:max_samples]
    return waveform
 
 
def save_audio(path: str, waveform: np.ndarray, sr: int = SAMPLE_RATE):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    waveform = waveform / (np.abs(waveform).max() + 1e-8)
    sf.write(path, waveform, sr)
    print(f"  → Sauvegardé : {path}")
 
 
def compute_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    snr  = 10 * np.log10(np.var(a[:n]) / (np.var(diff) + 1e-8))
    rmse = np.sqrt(np.mean(diff ** 2))
    return {"SNR (dB)": round(snr, 2), "RMSE": round(float(rmse), 5)}
 
 
# ─────────────────────────────────────────────
# Hook pour intercepter les latents du UNet
# ─────────────────────────────────────────────
class LatentCapture:
    """
    Capture les latents finaux via l ancien API callback (step, timestep, latents).
    Compatible avec toutes les versions de diffusers.
    """
    def __init__(self):
        self.latents = None
 
    def hook(self, step: int, timestep: int, latents):
        self.latents = latents.clone()
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(input_path: str, output_dir: str):
    print(f"\n{'='*55}")
    print(f"  AudioLDM2 — Test VAE Encode / Decode")
    print(f"{'='*55}\n")
 
    os.makedirs(output_dir, exist_ok=True)
 
    # ── 1. Chargement du modèle ──────────────────────────────
    print(f"[1/4] Chargement du modèle {MODEL_ID} sur {DEVICE}...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,   # float32 sur CPU pour la stabilité
    ).to(DEVICE)
 
    vae = pipe.vae
    print(f"      VAE latent dim     : {vae.config.latent_channels} channels")
    print(f"      VAE scaling factor : {vae.config.scaling_factor:.4f}")
 
    # ── 2. Génération d'un audio de référence via la pipeline ─
    print(f"\n[2/4] Génération d'un audio de référence (10s)...")
    print(f"      (on utilise la pipeline normale pour avoir des latents corrects)")
 
    capture = LatentCapture()
 
    # Génération avec peu de steps pour aller vite — on veut juste des latents propres
    result = pipe(
        prompt              = "music",
        audio_length_in_s   = 10.0,
        num_inference_steps = 20,
        guidance_scale      = 3.5,
        callback            = capture.hook,
        callback_steps      = 1,
    )
 
    wav_reference = result.audios[0]           # numpy [T]
    latents_final = capture.latents            # [1, C, H, W] — latents débruitées
 
    print(f"      Audio généré    : {len(wav_reference)/SAMPLE_RATE:.1f}s")
    print(f"      Latents shape   : {latents_final.shape}")
    print(f"      Latents stats   : mean={latents_final.mean():.3f}  std={latents_final.std():.3f}")
 
    out_ref = os.path.join(output_dir, "reference_pipeline.wav")
    save_audio(out_ref, wav_reference)
 
    # ── 3. VAE decode depuis les latents capturés ─────────────
    print(f"\n[3/4] VAE decode → mel → vocodeur (round-trip)...")
 
    with torch.no_grad():
        # Dé-scale les latents (inverse de ce que fait la pipeline)
        z = latents_final / vae.config.scaling_factor
        mel_decoded = vae.decode(z).sample        # [1, 1, n_mels, T]
 
    print(f"      Mel décodé shape : {mel_decoded.shape}")
 
    # Utiliser pipe.mel_spectrogram_to_waveform — la méthode officielle
    wav_roundtrip = pipe.mel_spectrogram_to_waveform(mel_decoded)
    if isinstance(wav_roundtrip, torch.Tensor):
        wav_roundtrip = wav_roundtrip.detach().cpu().numpy()
    if wav_roundtrip.ndim > 1:
        wav_roundtrip = wav_roundtrip[0]
 
    out_rt = os.path.join(output_dir, "roundtrip_vae_decode.wav")
    save_audio(out_rt, wav_roundtrip)
 
    # ── 4. Test encode → decode (vrai round-trip VAE) ─────────
    print(f"\n[4/4] VAE encode → decode (round-trip complet)...")
 
    with torch.no_grad():
        # Re-encode le Mel décodé
        posterior   = vae.encode(mel_decoded).latent_dist
        z_reenc     = posterior.sample() * vae.config.scaling_factor
 
        # Re-decode
        z_reenc_dec = z_reenc / vae.config.scaling_factor
        mel_reenc   = vae.decode(z_reenc_dec).sample
 
    print(f"      Latents re-encodés stats : mean={z_reenc.mean():.3f}  std={z_reenc.std():.3f}")
 
    wav_reenc = pipe.mel_spectrogram_to_waveform(mel_reenc)
    if isinstance(wav_reenc, torch.Tensor):
        wav_reenc = wav_reenc.detach().cpu().numpy()
    if wav_reenc.ndim > 1:
        wav_reenc = wav_reenc[0]
    out_reenc = os.path.join(output_dir, "reencoded_vae_roundtrip.wav")
    save_audio(out_reenc, wav_reenc)
 
    # ── Métriques ─────────────────────────────────────────────
    m1 = compute_metrics(wav_reference, wav_roundtrip)
    m2 = compute_metrics(wav_roundtrip, wav_reenc)
 
    print(f"\n{'─'*55}")
    print(f"  Métriques :")
    print(f"    Pipeline → VAE decode seul (référence vs roundtrip) :")
    for k, v in m1.items():
        print(f"      {k:15s} : {v}")
    print(f"    VAE encode→decode (roundtrip vs re-encodé) :")
    for k, v in m2.items():
        print(f"      {k:15s} : {v}")
    print(f"{'─'*55}")
    print(f"""
  Trois fichiers générés dans {output_dir}/ :
    reference_pipeline.wav      → sortie directe de la pipeline (référence)
    roundtrip_vae_decode.wav    → latents capturés → VAE decode → vocodeur
    reencoded_vae_roundtrip.wav → mel → VAE encode → decode → vocodeur
 
  Si reference ≈ roundtrip → le VAE decode + vocodeur fonctionne bien
  Si roundtrip ≈ reencoded  → le VAE encode/decode est stable
""")
    
if __name__ == "__main__":
    music_path = "./musique/music1.wav"  # chemin par défaut
    main(music_path, "./encode_decode/reconstructed_audioldm2.wav")
