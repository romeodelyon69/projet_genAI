"""
AudioLDM2 — Style Transfer (audio2audio)
-----------------------------------------
Principe : on encode l'audio source dans l'espace latent du VAE,
on ajoute du bruit jusqu'à un niveau `strength`, puis on débruite
en guidant avec un prompt texte décrivant le style cible.
 
C'est l'équivalent audio de img2img dans Stable Diffusion.
 
Paramètre clé — strength (0.0 → 1.0) :
    0.1  → très proche de l'original, style léger
    0.4  → bon compromis contenu/style
    0.7  → style très marqué, contenu altéré
    1.0  → génération complète (ignore l'audio source)
 
Usage:
    python audioldm2_style_transfer.py \
        --input      musique.wav \
        --prompt     "jazz piano, soft, nocturne" \
        --strength   0.5
 
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
 
from diffusers import AudioLDM2Pipeline, DDIMScheduler
 
 
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_ID    = "cvssp/audioldm2-music"
SAMPLE_RATE = 16000
DDIM_STEPS  = 50
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
 
 
# ─────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────
def load_audio(path: str, max_seconds: int = 10) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(wav) > SAMPLE_RATE * max_seconds:
        print(f"      ⚠ Troncature à {max_seconds}s")
        wav = wav[:SAMPLE_RATE * max_seconds]
    wav = wav / (np.abs(wav).max() + 1e-8)
    return wav
 
 
def save_audio(path: str, wav, sr: int = SAMPLE_RATE):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float().numpy()
    if wav.ndim > 1:
        wav = wav.squeeze()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wav = wav / (np.abs(wav).max() + 1e-8)
    sf.write(path, wav, sr)
    print(f"  → {path}")
 
 
# ─────────────────────────────────────────────
# Audio → Mel (paramètres exacts AudioLDM2)
# ─────────────────────────────────────────────
def audio_to_mel(wav: np.ndarray, target_length: int = 1000) -> torch.Tensor:
    """Output shape : [1, 1, T, 64]"""
    wav_t = torch.FloatTensor(wav).unsqueeze(0)
    mel_tf = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, win_length=1024,
        hop_length=160, n_mels=64, f_min=0.0, f_max=8000.0,
        power=1.0, norm="slaney", mel_scale="slaney",
    )
    mel = mel_tf(wav_t)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    t = mel.shape[-1]
    if t < target_length:
        mel = torch.nn.functional.pad(mel, (0, target_length - t))
    else:
        mel = mel[..., :target_length]
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    mel = mel.permute(0, 2, 1).unsqueeze(1)    # [1, 1, T, 64]
    return mel.to(DEVICE)
 
 
# ─────────────────────────────────────────────
# VAE encode / decode
# ─────────────────────────────────────────────
def vae_encode(mel: torch.Tensor, vae) -> torch.Tensor:
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        z = vae.encode(mel.to(dtype)).latent_dist.mode()
        z = z * vae.config.scaling_factor
    return z
 
 
def vae_decode(z: torch.Tensor, vae, pipe) -> np.ndarray:
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        mel = vae.decode(z.to(dtype) / vae.config.scaling_factor).sample
    wav = pipe.mel_spectrogram_to_waveform(mel)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    return wav.squeeze()
 
 
# ─────────────────────────────────────────────
# Ajout de bruit contrôlé (forward diffusion)
# ─────────────────────────────────────────────
def add_noise(z0: torch.Tensor, scheduler, strength: float,
              n_steps: int) -> tuple[torch.Tensor, int]:
    """
    Ajoute du bruit gaussien jusqu'au timestep t = strength * n_steps.
    Retourne (z_noisy, t_start) où t_start est le step de départ
    pour le débruitage.
    """
    scheduler.set_timesteps(n_steps, device=DEVICE)
 
    # Timestep de départ = on ne remonte pas jusqu'au bruit pur
    t_start  = max(1, int(n_steps * strength))
    timestep = scheduler.timesteps[n_steps - t_start]
 
    noise = torch.randn_like(z0)
    z_noisy = scheduler.add_noise(
        z0.float(), noise.float(), timestep.unsqueeze(0)
    ).to(z0.dtype)
 
    print(f"      Strength={strength:.2f} → timestep={timestep.item()}, "
          f"bruit ajouté sur {t_start}/{n_steps} steps")
    return z_noisy, n_steps - t_start
 
 
# ─────────────────────────────────────────────
# Débruitage guidé par prompt (style transfer)
# ─────────────────────────────────────────────
def guided_denoise(z_noisy: torch.Tensor, t_start: int,
                   pipe, prompt: str, guidance_scale: float,
                   n_steps: int) -> torch.Tensor:
    """
    Débruite z_noisy en partant du step t_start,
    guidé par le prompt texte.
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(n_steps, device=DEVICE)
 
    # Encoder le prompt
    do_cfg = guidance_scale > 1.0
    with torch.no_grad():
        prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=DEVICE,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )
 
    dtype = next(pipe.unet.parameters()).dtype
    z = z_noisy.clone()
 
    # On ne débruite qu'à partir de t_start (les steps avant sont "déjà faits")
    active_timesteps = scheduler.timesteps[t_start:]
 
    for t in active_timesteps:
        # Classifier-free guidance : on double le batch (conditioned + unconditioned)
        z_input = torch.cat([z, z]) if do_cfg else z
 
        with torch.no_grad():
            noise_pred = pipe.unet(
                z_input.to(dtype),
                t,
                encoder_hidden_states   = generated_prompt_embeds.to(dtype),
                encoder_hidden_states_1 = prompt_embeds.to(dtype),
                encoder_attention_mask_1= attention_mask,
            ).sample
 
        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
 
        z = scheduler.step(noise_pred, t, z).prev_sample
 
    return z
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(input_path: str, prompt: str, output_dir: str,
         strength: float, guidance_scale: float, n_steps: int):
 
    print(f"\n{'='*55}")
    print(f"  AudioLDM2 — Style Transfer")
    print(f"  Prompt   : \"{prompt}\"")
    print(f"  Strength : {strength}  |  Guidance : {guidance_scale}")
    print(f"  Device   : {DEVICE}")
    print(f"{'='*55}\n")
 
    os.makedirs(output_dir, exist_ok=True)
 
    # ── 1. Modèle ───────────────────────────────────────────
    print("[1/5] Chargement du modèle...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    vae = pipe.vae
 
    # ── 2. Audio source ─────────────────────────────────────
    print(f"\n[2/5] Chargement audio : {input_path}")
    wav_src = load_audio(input_path, max_seconds=30)
    print(f"      Durée : {len(wav_src)/SAMPLE_RATE:.1f}s")
    save_audio(os.path.join(output_dir, "00_original.wav"), wav_src)
 
    # ── 3. Encode dans l'espace latent ──────────────────────
    print(f"\n[3/5] Encodage VAE...")
    mel = audio_to_mel(wav_src)
    z0  = vae_encode(mel, vae)
    print(f"      z0 shape : {z0.shape}")
 
    # Reconstruction directe pour vérifier la qualité du VAE
    wav_recon = vae_decode(z0, vae, pipe)
    save_audio(os.path.join(output_dir, "01_vae_reconstruction.wav"), wav_recon)
 
    # ── 4. Ajout de bruit contrôlé ──────────────────────────
    print(f"\n[4/5] Ajout de bruit (strength={strength})...")
    z_noisy, t_start = add_noise(z0, pipe.scheduler, strength, n_steps)
 
    # ── 5. Débruitage guidé par le prompt ───────────────────
    print(f"\n[5/5] Débruitage guidé : \"{prompt}\"")
    z_styled = guided_denoise(
        z_noisy, t_start, pipe,
        prompt=prompt,
        guidance_scale=guidance_scale,
        n_steps=n_steps,
    )
 
    wav_styled = vae_decode(z_styled, vae, pipe)
    out_name   = f"styled_s{strength}_g{guidance_scale}.wav"
    save_audio(os.path.join(output_dir, out_name), wav_styled)
 
    print(f"\n{'─'*55}")
    print(f"  Fichiers générés dans {output_dir}/ :")
    print(f"    00_original.wav          → audio source")
    print(f"    01_vae_reconstruction    → VAE round-trip (référence qualité)")
    print(f"    {out_name}")
    print(f"{'─'*55}")
    print(f"""
  Si le résultat ne convient pas, ajuster :
    --strength     plus bas  → reste plus proche de l'original
                   plus haut → style plus marqué, moins fidèle
    --guidance     plus haut → prompt plus contraignant (3-7 recommandé)
""")
 
 
if __name__ == "__main__":
    music_path = "./musique/music4.wav"  # chemin par défaut
    style_prompt = "classical orchestra, epic, cinematic"
    output_dir = "./encode_decode"  # répertoire de sortie par défaut
    strength = 0.4  # force du style par défaut
    guidance = 5.0  # guidance scale par défaut
    steps = 50
    main(music_path, style_prompt, output_dir, strength, guidance, steps)