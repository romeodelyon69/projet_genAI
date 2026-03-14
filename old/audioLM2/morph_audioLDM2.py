"""
AudioLDM2 VAE — Morphing entre deux morceaux
---------------------------------------------
Inspiré du code DAC, remplace l'encodeur DAC par le VAE d'AudioLDM2.
 
Structure :
  1. Encode music1 et music2 → zA, zB via VAE AudioLDM2
  2. Trouve la meilleure fenêtre de transition (distance L2 dans le latent)
  3. Interpole chunk par chunk avec SLERP
  4. Reconstruit : pre_transition + morph_segment + post_transition
 
Usage:
    python audioldm2_morph.py --input_a music1.wav --input_b music2.wav
 
Dépendances:
    pip install diffusers transformers accelerate soundfile librosa torch torchaudio
"""
 
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
import torchaudio.transforms as T
from tqdm import tqdm
 
from diffusers import AudioLDM2Pipeline
 
 
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_ID    = "cvssp/audioldm2-music"
SAMPLE_RATE = 16000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
 
# Paramètres Mel exacts d'AudioLDM2
MEL_PARAMS = dict(
    sample_rate = SAMPLE_RATE,
    n_fft       = 1024,
    win_length  = 1024,
    hop_length  = 160,       # 1 frame latent ≈ 160 samples audio
    n_mels      = 64,
    f_min       = 0.0,
    f_max       = 8000.0,
    power       = 1.0,
    norm        = "slaney",
    mel_scale   = "slaney",
)
 
# Facteur de compression VAE sur la dim temporelle : hop(160) × stride_vae(4) = 640
# → 1 pas latent ≈ 640 samples audio à 16kHz ≈ 40ms
LATENT_HOP = 160 * 4   # samples audio par pas latent
 
 
# ─────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────
def load_audio(path: str) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    waveform = waveform / (np.abs(waveform).max() + 1e-8)
    return waveform
 
 
def save_audio(path: str, waveform, sr: int = SAMPLE_RATE):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().float().numpy()
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    waveform = waveform / (np.abs(waveform).max() + 1e-8)
    sf.write(path, waveform, sr)
    print(f"  → Sauvegardé : {path}")
 
 
# ─────────────────────────────────────────────
# Audio → Mel (avec stats locales pour normalisation)
# ─────────────────────────────────────────────
def audio_to_mel(waveform: np.ndarray) -> tuple[torch.Tensor, float, float]:
    """
    Retourne (mel, mean, std) pour pouvoir dénormaliser plus tard si besoin.
    Shape mel : [1, 1, T, 64]
    """
    wav_t = torch.FloatTensor(waveform).unsqueeze(0)
    mel_transform = T.MelSpectrogram(**MEL_PARAMS)
    mel = mel_transform(wav_t)                          # [1, 64, T]
    mel = torch.log(torch.clamp(mel, min=1e-5))         # log-Mel
    mean, std = mel.mean().item(), mel.std().item()
    mel = (mel - mean) / (std + 1e-8)                   # z-score
    mel = mel.permute(0, 2, 1).unsqueeze(1)             # [1, 1, T, 64]
    return mel.to(DEVICE), mean, std
 
 
# ─────────────────────────────────────────────
# VAE encode / decode
# ─────────────────────────────────────────────
def vae_encode(mel: torch.Tensor, vae) -> torch.Tensor:
    """Encode un Mel complet → latent z. Shape : [1, 8, T_lat, 16]"""
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        z = vae.encode(mel.to(dtype)).latent_dist.mode()
        z = z * vae.config.scaling_factor
    return z
 
 
def vae_decode_chunk(z_chunk: torch.Tensor, vae, pipe) -> np.ndarray:
    """Décode un chunk latent → waveform numpy 1D."""
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        mel = vae.decode(z_chunk.to(dtype) / vae.config.scaling_factor).sample
    wav = pipe.mel_spectrogram_to_waveform(mel)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    return wav.squeeze()
 
 
# ─────────────────────────────────────────────
# Trouver la meilleure fenêtre de transition
# ─────────────────────────────────────────────
def find_best_transition(zA: torch.Tensor, zB: torch.Tensor,
                         window: int, step: int = None) -> tuple[int, int]:
    """
    Parcourt zA et zB pour trouver la paire (i, j) minimisant
    la distance L2 entre les fenêtres zA[:,i:i+w] et zB[:,j:j+w].
 
    zA, zB shape : [1, C, T, F]
    On compare sur la dim temporelle T.
    """
    T_A = zA.shape[2]
    T_B = zB.shape[2]
 
    if step is None:
        step = max(1, window // 20)
 
    min_dist = float('inf')
    best = (0, 0)
 
    zA_cpu = zA.float().cpu()
    zB_cpu = zB.float().cpu()
 
    i_range = range(10 * window, T_A - window, step)
    for i in tqdm(i_range, desc="      Recherche fenêtre"):
        chunk_a = zA_cpu[:, :, i : i + window, :]
        for j in range(0, T_B - 10 * window, step):
            chunk_b = zB_cpu[:, :, j : j + window, :]
            dist = torch.norm(chunk_a - chunk_b, p=2).item()
            if dist < min_dist:
                min_dist = dist
                best = (i, j)
 
    print(f"      Meilleure fenêtre : i={best[0]}, j={best[1]}, dist L2={min_dist:.2f}")
    return best
 
 
# ─────────────────────────────────────────────
# SLERP
# ─────────────────────────────────────────────
def slerp(zA: torch.Tensor, zB: torch.Tensor, t: float) -> torch.Tensor:
    za_flat = zA.flatten().float()
    zb_flat = zB.flatten().float()
    dot = torch.dot(za_flat, zb_flat) / (za_flat.norm() * zb_flat.norm() + 1e-8)
    omega = torch.acos(torch.clamp(dot, -1.0, 1.0))
    so = torch.sin(omega)
    if so.abs() < 1e-4:
        return ((1 - t) * zA + t * zB)
    result = (torch.sin((1 - t) * omega) / so) * za_flat + \
             (torch.sin(t * omega) / so) * zb_flat
    return result.reshape(zA.shape).to(zA.dtype)
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(path_a: str, path_b: str, output_dir: str,
         morph_seconds: float, n_steps: int):
 
    print(f"\n{'='*55}")
    print(f"  AudioLDM2 VAE — Morphing musical")
    print(f"  Durée morphing : {morph_seconds}s  |  Steps : {n_steps}")
    print(f"{'='*55}\n")
 
    os.makedirs(output_dir, exist_ok=True)
 
    # ── 1. Modèle ───────────────────────────────────────────
    print("[1/5] Chargement du modèle AudioLDM2...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(DEVICE)
    vae = pipe.vae
    print(f"      VAE scaling factor : {vae.config.scaling_factor:.4f}")
 
    # ── 2. Chargement audio ─────────────────────────────────
    print(f"\n[2/5] Chargement audio...")
    wav_a = load_audio(path_a)
    wav_b = load_audio(path_b)
    print(f"      A : {len(wav_a)/SAMPLE_RATE:.1f}s")
    print(f"      B : {len(wav_b)/SAMPLE_RATE:.1f}s")
 
    # ── 3. Encode ───────────────────────────────────────────
    print(f"\n[3/5] Encodage VAE (music A et B)...")
    mel_a, *_ = audio_to_mel(wav_a)
    mel_b, *_ = audio_to_mel(wav_b)
 
    zA = vae_encode(mel_a, vae)   # [1, 8, T_A, 16]
    zB = vae_encode(mel_b, vae)   # [1, 8, T_B, 16]
 
    print(f"      zA shape : {zA.shape}  mean={zA.mean():.3f}  std={zA.std():.3f}")
    print(f"      zB shape : {zB.shape}  mean={zB.mean():.3f}  std={zB.std():.3f}")
 
    # Taille de la fenêtre de morphing en pas latents
    # 1 pas latent ≈ 640 samples = 40ms à 16kHz
    window_latents = int(morph_seconds * SAMPLE_RATE / LATENT_HOP)
    window_latents = min(window_latents, zA.shape[2] - 1, zB.shape[2] - 1)
    print(f"      Fenêtre morphing : {window_latents} pas latents "
          f"≈ {window_latents * LATENT_HOP / SAMPLE_RATE:.1f}s")
 
    # ── 4. Meilleure fenêtre de transition ──────────────────
    print(f"\n[4/5] Recherche de la meilleure fenêtre de transition...")
    i_start, j_start = find_best_transition(zA, zB, window=window_latents)
 
    # Positions audio correspondantes
    i_audio = i_start * LATENT_HOP
    j_audio = j_start * LATENT_HOP
    window_audio = window_latents * LATENT_HOP
 
    print(f"      Position dans A : {i_audio/SAMPLE_RATE:.1f}s")
    print(f"      Position dans B : {j_audio/SAMPLE_RATE:.1f}s")
 
    # ── 5. Morphing chunk par chunk ─────────────────────────
    print(f"\n[5/5] Morphing SLERP ({n_steps} steps)...")
 
    zA_chunk = zA[:, :, i_start : i_start + window_latents, :]
    zB_chunk = zB[:, :, j_start : j_start + window_latents, :]
 
    # Aligner les tailles
    min_t = min(zA_chunk.shape[2], zB_chunk.shape[2])
    zA_chunk = zA_chunk[:, :, :min_t, :]
    zB_chunk = zB_chunk[:, :, :min_t, :]
 
    chunk_size = max(1, min_t // n_steps)
    morph_chunks = []
 
    for i in tqdm(range(n_steps), desc="      Morphing"):
        t = i / (n_steps - 1) if n_steps > 1 else 0.0
        t_start = i * chunk_size
        t_end   = t_start + chunk_size
 
        za_i = zA_chunk[:, :, t_start:t_end, :]
        zb_i = zB_chunk[:, :, t_start:t_end, :]
 
        z_morph = slerp(za_i, zb_i, t)
        wav_chunk = vae_decode_chunk(z_morph, vae, pipe)
        morph_chunks.append(wav_chunk)
 
    morph_audio = np.concatenate(morph_chunks)
 
    # ── Assemblage final ────────────────────────────────────
    pre_transition  = wav_a[:i_audio]
    post_transition = wav_b[j_audio + window_audio:]
 
    # Crossfade léger aux jonctions (50ms)
    fade_len = int(0.05 * SAMPLE_RATE)
 
    def crossfade(a: np.ndarray, b: np.ndarray, fade: int) -> np.ndarray:
        """Fondu enchaîné entre la fin de a et le début de b."""
        if len(a) < fade or len(b) < fade:
            return np.concatenate([a, b])
        fade_out = np.linspace(1, 0, fade)
        fade_in  = np.linspace(0, 1, fade)
        a_out = a.copy(); a_out[-fade:] *= fade_out
        b_in  = b.copy(); b_in[:fade]  *= fade_in
        overlap = a_out[-fade:] + b_in[:fade]
        return np.concatenate([a_out[:-fade], overlap, b_in[fade:]])
 
    output = crossfade(pre_transition, morph_audio, fade_len)
    output = crossfade(output, post_transition, fade_len)
 
    # ── Sauvegarde ──────────────────────────────────────────
    out_path = os.path.join(output_dir, "morphing_audioldm2.wav")
    save_audio(out_path, output)
 
    print(f"\n{'─'*55}")
    print(f"  Durées :")
    print(f"    Pre-transition  : {len(pre_transition)/SAMPLE_RATE:.1f}s")
    print(f"    Morphing        : {len(morph_audio)/SAMPLE_RATE:.1f}s")
    print(f"    Post-transition : {len(post_transition)/SAMPLE_RATE:.1f}s")
    print(f"    Total           : {len(output)/SAMPLE_RATE:.1f}s")
    print(f"{'─'*55}\n")

if __name__ == "__main__":
    
    music1_path = "./musique/music1.wav"  # chemin par défaut
    music2_path = "./musique/music3.wav"  # chemin par défaut
    output_dir = "./morphing"  # répertoire de sortie par défaut
    morph_seconds = 5.0  # durée de morphing par défaut
    
    n_steps = 20  # nombre de steps dans le morphing par défaut
    main(music1_path, music2_path, output_dir,
         morph_seconds, n_steps)