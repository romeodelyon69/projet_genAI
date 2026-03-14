"""
AudioLDM2 — Encode & Decode via DDIM Inversion (v4)
-----------------------------------------------------
Encode un vrai morceau audio en remontant le processus de diffusion
(DDIM Inversion), puis reconstruit depuis les latents inversés.
 
Schéma :
    Audio input
        ↓ librosa → numpy
    Waveform → VAE encode → z0  (latent "propre")
        ↓ DDIM Inversion (forward diffusion guidée)
    zT  (latent bruité)
        ↓ DDIM Denoising (débruitage normal)
    z0' (latent reconstruit)
        ↓ VAE decode → mel → vocodeur
    Audio reconstruit
 
Usage:
    python audioldm2_encode_decode.py --input musique.wav --output_dir ./encode_decode
 
Dépendances:
    pip install diffusers transformers accelerate soundfile librosa torch torchaudio
"""
 
import argparse
import os
import torch
import numpy as np
import soundfile as sf
import librosa
import torchaudio.transforms as T
from tqdm import tqdm
 
from diffusers import AudioLDM2Pipeline, DDIMScheduler
 
 
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_ID     = "cvssp/audioldm2-music"
SAMPLE_RATE  = 16000
DDIM_STEPS   = 50      # nb de steps pour inversion + reconstruction
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
 
 
# ─────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────
def load_audio(path: str, max_seconds: int = 10) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    max_samples = SAMPLE_RATE * max_seconds
    if len(waveform) > max_samples:
        print(f"      ⚠ Troncature à {max_seconds}s")
        waveform = waveform[:max_samples]
    return waveform
 
 
def save_audio(path: str, waveform, sr: int = SAMPLE_RATE):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().float().numpy()
    if waveform.ndim > 1:
        waveform = waveform[0]
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    waveform = waveform / (np.abs(waveform).max() + 1e-8)
    sf.write(path, waveform, sr)
    print(f"  → Sauvegardé : {path}")
 
 
def compute_metrics(a, b) -> dict:
    if isinstance(a, torch.Tensor): a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor): b = b.detach().cpu().numpy()
    n = min(len(a.flatten()), len(b.flatten()))
    a, b = a.flatten()[:n], b.flatten()[:n]
    diff = a - b
    snr  = 10 * np.log10(np.var(a) / (np.var(diff) + 1e-8))
    rmse = np.sqrt(np.mean(diff ** 2))
    return {"SNR (dB)": round(snr, 2), "RMSE": round(float(rmse), 5)}
 
 
# ─────────────────────────────────────────────
# Audio → Mel (paramètres exacts AudioLDM2)
# ─────────────────────────────────────────────
def audio_to_mel(waveform: np.ndarray, target_length: int = 1000) -> torch.Tensor:
    """
    Convertit un waveform numpy en log-Mel spectrogram normalisé
    prêt pour le VAE d'AudioLDM2.
 
    Paramètres calqués sur le preprocessing d'AudioLDM2 :
      n_mels=64, n_fft=1024, hop=160, fmax=8000, norm=slaney
    Normalisation finale : mean=0, std=1 (attendu par le VAE).
    """
    wav_tensor = torch.FloatTensor(waveform).unsqueeze(0)  # [1, T]
 
    mel_transform = T.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft       = 1024,
        win_length  = 1024,
        hop_length  = 160,
        n_mels      = 64,
        f_min       = 0.0,
        f_max       = 8000.0,
        power       = 1.0,
        norm        = "slaney",
        mel_scale   = "slaney",
    )
 
    mel = mel_transform(wav_tensor)                         # [1, 64, T]
    mel = torch.log(torch.clamp(mel, min=1e-5))             # log-Mel
 
    # Padding/crop pour avoir exactement target_length frames
    t = mel.shape[-1]
    if t < target_length:
        mel = torch.nn.functional.pad(mel, (0, target_length - t))
    else:
        mel = mel[..., :target_length]
 
    # Normalisation z-score globale (≈ ce qu'AudioLDM2 fait au preprocessing)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
 
    # Shape attendu par le VAE : [B, 1, T, n_mels] (time-first comme les latents)
    mel = mel.permute(0, 2, 1).unsqueeze(1)                # [1, 1, T, 64]
    return mel.to(DEVICE)
 
 
# ─────────────────────────────────────────────
# VAE encode / decode
# ─────────────────────────────────────────────
def vae_encode(mel: torch.Tensor, vae) -> torch.Tensor:
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        posterior = vae.encode(mel.to(dtype)).latent_dist
        z = posterior.mode()                        # mode plutôt que sample → déterministe
        z = z * vae.config.scaling_factor
    print(f"      z shape : {z.shape}  mean={z.mean():.3f}  std={z.std():.3f}")
    return z
 
 
def vae_decode(z: torch.Tensor, vae) -> torch.Tensor:
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        mel = vae.decode(z.to(dtype) / vae.config.scaling_factor).sample
    return mel
 
 
# ─────────────────────────────────────────────
# DDIM Inversion
# ─────────────────────────────────────────────
def ddim_inversion(z0: torch.Tensor, pipe, n_steps: int = DDIM_STEPS) -> torch.Tensor:
    """
    Remonte de z0 (latent propre) vers zT (bruit gaussien) en suivant
    le scheduler DDIM à l'envers.
 
    À chaque step t → t+1 :
        z_{t+1} = √(ᾱ_{t+1}) * pred_x0 + √(1-ᾱ_{t+1}) * pred_noise
 
    On utilise un UNet unconditioned (guidance_scale=1) pour que
    l'inversion soit cohérente avec la reconstruction.
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(n_steps, device=DEVICE)
    timesteps = scheduler.timesteps.flip(0)         # [T0, T1, ..., TN] → inversion
 
    # Embeddings vides pour inversion sans conditioning
    # (on passe un prompt neutre)
    with torch.no_grad():
        prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
            prompt              = "",
            device              = DEVICE,
            num_waveforms_per_prompt = 1,
            do_classifier_free_guidance = False,
        )
 
    zt = z0.clone()
    dtype = next(pipe.unet.parameters()).dtype
 
    print(f"      DDIM Inversion ({n_steps} steps)...")
    for i, t in enumerate(tqdm(timesteps, desc="      Inversion")):
        with torch.no_grad():
            noise_pred = pipe.unet(
                zt.to(dtype),
                t,
                encoder_hidden_states          = generated_prompt_embeds.to(dtype),
                encoder_hidden_states_1        = prompt_embeds.to(dtype),
                encoder_attention_mask_1       = attention_mask,
            ).sample
 
        # Step inverse DDIM
        alpha_prod_t = scheduler.alphas_cumprod[t]
        if i + 1 < len(timesteps):
            alpha_prod_next = scheduler.alphas_cumprod[timesteps[i + 1]]
        else:
            alpha_prod_next = torch.tensor(1.0)
 
        beta_prod_t = 1 - alpha_prod_t
 
        # Pred x0
        pred_x0 = (zt - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        pred_x0 = pred_x0.clamp(-4, 4)
 
        # Mise à jour zT
        zt = alpha_prod_next.sqrt() * pred_x0 + (1 - alpha_prod_next).sqrt() * noise_pred
 
    print(f"      zT stats : mean={zt.mean():.3f}  std={zt.std():.3f}")
    return zt
 
 
# ─────────────────────────────────────────────
# DDIM Denoising (reconstruction)
# ─────────────────────────────────────────────
def ddim_denoise(zt: torch.Tensor, pipe, n_steps: int = DDIM_STEPS) -> torch.Tensor:
    """
    Reconstruit z0 depuis zT en suivant le scheduler DDIM en avant.
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(n_steps, device=DEVICE)
 
    with torch.no_grad():
        prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
            prompt              = "",
            device              = DEVICE,
            num_waveforms_per_prompt = 1,
            do_classifier_free_guidance = False,
        )
 
    z = zt.clone()
    dtype = next(pipe.unet.parameters()).dtype
 
    print(f"      DDIM Denoising ({n_steps} steps)...")
    for t in tqdm(scheduler.timesteps, desc="      Denoising"):
        with torch.no_grad():
            noise_pred = pipe.unet(
                z.to(dtype),
                t,
                encoder_hidden_states          = generated_prompt_embeds.to(dtype),
                encoder_hidden_states_1        = prompt_embeds.to(dtype),
                encoder_attention_mask_1       = attention_mask,
            ).sample
 
        z = scheduler.step(noise_pred, t, z).prev_sample
 
    print(f"      z0' stats : mean={z.mean():.3f}  std={z.std():.3f}")
    return z
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(input_path: str, output_dir: str):
    print(f"\n{'='*55}")
    print(f"  AudioLDM2 — DDIM Inversion Encode/Decode")
    print(f"{'='*55}\n")
 
    os.makedirs(output_dir, exist_ok=True)
 
    # ── 1. Chargement modèle ────────────────────────────────
    print(f"[1/5] Chargement du modèle {MODEL_ID}...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.float32,
    ).to(DEVICE)
 
    # Remplacer le scheduler par DDIM (nécessaire pour l'inversion)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(DDIM_STEPS)
 
    vae = pipe.vae
    print(f"      VAE channels   : {vae.config.latent_channels}")
    print(f"      Scheduler      : {pipe.scheduler.__class__.__name__}")
 
    # ── 2. Chargement audio ─────────────────────────────────
    print(f"\n[2/5] Chargement audio : {input_path}")
    waveform = load_audio(input_path)
    print(f"      Durée : {len(waveform)/SAMPLE_RATE:.2f}s")
 
    # Sauvegarde de l'original tronqué pour comparaison
    out_orig = os.path.join(output_dir, "original.wav")
    save_audio(out_orig, waveform)
 
    # ── 3. Audio → Mel → VAE encode ────────────────────────
    print(f"\n[3/5] Audio → Mel → VAE encode (z0)...")
    mel = audio_to_mel(waveform)
    print(f"      Mel shape : {mel.shape}  — [B, 1, T, n_mels]")
 
    z0 = vae_encode(mel, vae)
 
    # Vérification : decode immédiat sans diffusion
    mel_direct = vae_decode(z0, vae)
    wav_direct = pipe.mel_spectrogram_to_waveform(mel_direct)
    if isinstance(wav_direct, torch.Tensor):
        wav_direct = wav_direct.detach().cpu().numpy()
    if wav_direct.ndim > 1:
        wav_direct = wav_direct[0]
    save_audio(os.path.join(output_dir, "vae_direct_roundtrip.wav"), wav_direct)
 
    # ── 4. DDIM Inversion : z0 → zT ────────────────────────
    print(f"\n[4/5] DDIM Inversion (z0 → zT)...")
    zT = ddim_inversion(z0, pipe, n_steps=DDIM_STEPS)
 
    # ── 5. DDIM Denoising : zT → z0' → audio ───────────────
    print(f"\n[5/5] DDIM Denoising (zT → z0') + synthèse audio...")
    z0_reconstructed = ddim_denoise(zT, pipe, n_steps=DDIM_STEPS)
 
    mel_reconstructed = vae_decode(z0_reconstructed, vae)
    wav_reconstructed = pipe.mel_spectrogram_to_waveform(mel_reconstructed)
    if isinstance(wav_reconstructed, torch.Tensor):
        wav_reconstructed = wav_reconstructed.detach().cpu().numpy()
    if wav_reconstructed.ndim > 1:
        wav_reconstructed = wav_reconstructed[0]
    save_audio(os.path.join(output_dir, "ddim_roundtrip.wav"), wav_reconstructed)
 
    # ── Métriques ───────────────────────────────────────────
    m_direct = compute_metrics(waveform[:len(wav_direct)], wav_direct[:len(waveform)])
    m_ddim   = compute_metrics(waveform[:len(wav_reconstructed)], wav_reconstructed[:len(waveform)])
 
    print(f"\n{'─'*55}")
    print(f"  Métriques vs original :")
    print(f"    VAE direct (sans diffusion) :")
    for k, v in m_direct.items():
        print(f"      {k:15s} : {v}")
    print(f"    DDIM inversion + reconstruction :")
    for k, v in m_ddim.items():
        print(f"      {k:15s} : {v}")
    print(f"{'─'*55}")
    print(f"""
  Fichiers générés dans {output_dir}/ :
    original.wav              → audio source tronqué (référence)
    vae_direct_roundtrip.wav  → VAE encode→decode direct (sans diffusion)
    ddim_roundtrip.wav        → DDIM inversion + reconstruction complète
 
  vae_direct révèle la capacité du VAE à encoder ta musique.
  ddim_roundtrip révèle si l'inversion est stable end-to-end.
""")

if __name__ == "__main__":
    music_path = "./musique/music2.wav"  # chemin par défaut
    main(music_path, "./encode_decode/reconstructed_audioldm2.wav")