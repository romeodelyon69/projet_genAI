"""
Morphing musical — HPSS + Phase Vocoder
-----------------------------------------
Stratégie : au lieu d'interpoler dans un espace latent non structuré,
on décompose chaque morceau en composantes physiquement interprétables
(harmonique + percussive) et on interpole chacune séparément.
 
Pipeline :
    Music A → HPSS → Harm_A, Perc_A
    Music B → HPSS → Harm_B, Perc_B
 
    Pour t ∈ [0, 1] :
        Harm_t    = phase_vocoder_interp(Harm_A, Harm_B, t)
        Perc_t    = crossfade(Perc_A, Perc_B, t)
        Audio_t   = Harm_t + Perc_t
 
    Résultat : pre_A + [Audio_0..Audio_N] + post_B
 
Usage:
    python morph_hpss.py --input_a music1.wav --input_b music2.wav
 
Dépendances:
    pip install librosa soundfile numpy torch
"""
 
import argparse
import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
 
 
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SR        = 22050   # librosa natif — bon compromis qualité/vitesse
N_FFT     = 2048
HOP       = 512
N_STEPS   = 30      # frames de transition
 
 
# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────
def load_audio(path: str, sr: int = SR) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr, mono=True)
    wav = wav / (np.abs(wav).max() + 1e-8)
    return wav
 
 
def save_audio(path: str, wav: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wav = wav / (np.abs(wav).max() + 1e-8)
    sf.write(path, wav, sr)
    print(f"  → {path}")
 
 
# ─────────────────────────────────────────────
# HPSS — Harmonic / Percussive Separation
# ─────────────────────────────────────────────
def hpss(wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sépare un signal audio en composante harmonique et percussive.
    Retourne (harmonic, percussive) en domaine temporel.
    """
    D = librosa.stft(wav, n_fft=N_FFT, hop_length=HOP)
    H, P = librosa.decompose.hpss(D, margin=3.0)
    harmonic   = librosa.istft(H, hop_length=HOP, length=len(wav))
    percussive = librosa.istft(P, hop_length=HOP, length=len(wav))
    return harmonic, percussive
 
 
# ─────────────────────────────────────────────
# Trouver la meilleure fenêtre de transition
# ─────────────────────────────────────────────
def find_best_window(wav_a: np.ndarray, wav_b: np.ndarray,
                     window_sec: float = 4.0, sr: int = SR) -> tuple[int, int]:
    """
    Cherche les positions i, j dans wav_a et wav_b qui minimisent
    la distance entre les MFCCs des deux fenêtres.
    MFCCs capturent mieux la similarité timbrale que le signal brut.
    """
    window_samples = int(window_sec * sr)
    step_samples   = int(0.5 * sr)   # step de 500ms
 
    # Calculer les MFCCs une seule fois sur tout le signal
    mfcc_a = librosa.feature.mfcc(y=wav_a, sr=sr, n_mfcc=20, hop_length=HOP)
    mfcc_b = librosa.feature.mfcc(y=wav_b, sr=sr, n_mfcc=20, hop_length=HOP)
 
    window_frames = window_samples // HOP
    step_frames   = step_samples   // HOP
 
    T_A = mfcc_a.shape[1]
    T_B = mfcc_b.shape[1]
 
    min_dist = float('inf')
    best_i_frames, best_j_frames = 0, 0
 
    print("      Recherche fenêtre par MFCC distance...")
    for i in range(0, T_A - window_frames, step_frames):
        chunk_a = mfcc_a[:, i : i + window_frames]
        for j in range(0, T_B - window_frames, step_frames):
            chunk_b = mfcc_b[:, j : j + window_frames]
            dist = np.linalg.norm(chunk_a - chunk_b)
            if dist < min_dist:
                min_dist = dist
                best_i_frames = i
                best_j_frames = j
 
    best_i = best_i_frames * HOP
    best_j = best_j_frames * HOP
    print(f"      Meilleure fenêtre : A@{best_i/sr:.1f}s  B@{best_j/sr:.1f}s  "
          f"dist MFCC={min_dist:.1f}")
    return best_i, best_j
 
 
# ─────────────────────────────────────────────
# Interpolation spectrale avec Phase Vocoder
# ─────────────────────────────────────────────
def phase_vocoder_interp(wav_a: np.ndarray, wav_b: np.ndarray,
                          t: float) -> np.ndarray:
    """
    Interpole deux signaux dans le domaine fréquentiel :
    - Magnitudes interpolées linéairement
    - Phases de A pour t<0.5, de B pour t>0.5 (évite les artefacts de phase)
 
    C'est la version "propre" de l'interpolation spectrale.
    """
    # Aligner les longueurs
    n = min(len(wav_a), len(wav_b))
    wav_a, wav_b = wav_a[:n], wav_b[:n]
 
    DA = librosa.stft(wav_a, n_fft=N_FFT, hop_length=HOP)
    DB = librosa.stft(wav_b, n_fft=N_FFT, hop_length=HOP)
 
    mag_a, phase_a = np.abs(DA), np.angle(DA)
    mag_b, phase_b = np.abs(DB), np.angle(DB)
 
    # Interpolation des magnitudes
    mag_interp = (1 - t) * mag_a + t * mag_b
 
    # Phases : transition douce pour éviter le "phasing" artificiel
    # On utilise un crossfade sur les phases aussi
    if t < 0.5:
        # Côté A dominant — utiliser phases de A avec légère contamination B
        w = t * 2  # 0→1 sur la première moitié
        phase_interp = phase_a + w * _phase_diff(phase_a, phase_b)
    else:
        # Côté B dominant
        w = (t - 0.5) * 2  # 0→1 sur la deuxième moitié
        phase_interp = phase_b - (1 - w) * _phase_diff(phase_a, phase_b)
 
    D_interp = mag_interp * np.exp(1j * phase_interp)
    return librosa.istft(D_interp, hop_length=HOP, length=n)
 
 
def _phase_diff(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """Différence de phase normalisée dans [-π, π]."""
    diff = phase_b - phase_a
    return np.angle(np.exp(1j * diff))
 
 
# ─────────────────────────────────────────────
# Crossfade simple (pour percussive)
# ─────────────────────────────────────────────
def crossfade_signals(wav_a: np.ndarray, wav_b: np.ndarray, t: float) -> np.ndarray:
    n = min(len(wav_a), len(wav_b))
    return (1 - t) * wav_a[:n] + t * wav_b[:n]
 
 
# ─────────────────────────────────────────────
# Fondu enchaîné aux jonctions
# ─────────────────────────────────────────────
def crossfade_join(a: np.ndarray, b: np.ndarray, fade_sec: float = 0.05,
                   sr: int = SR) -> np.ndarray:
    fade = int(fade_sec * sr)
    fade = min(fade, len(a), len(b))
    if fade == 0:
        return np.concatenate([a, b])
    env_out = np.linspace(1, 0, fade)
    env_in  = np.linspace(0, 1, fade)
    overlap = a[-fade:] * env_out + b[:fade] * env_in
    return np.concatenate([a[:-fade], overlap, b[fade:]])
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(path_a: str, path_b: str, output_dir: str,
         morph_sec: float, n_steps: int):
 
    print(f"\n{'='*55}")
    print(f"  Morphing HPSS + Phase Vocoder")
    print(f"  Durée morphing : {morph_sec}s  |  Steps : {n_steps}")
    print(f"{'='*55}\n")
 
    os.makedirs(output_dir, exist_ok=True)
 
    # ── 1. Chargement ───────────────────────────────────────
    print("[1/5] Chargement audio...")
    wav_a = load_audio(path_a)
    wav_b = load_audio(path_b)
    print(f"      A : {len(wav_a)/SR:.1f}s")
    print(f"      B : {len(wav_b)/SR:.1f}s")
 
    # ── 2. HPSS ─────────────────────────────────────────────
    print("\n[2/5] Séparation Harmonique / Percussive (HPSS)...")
    harm_a, perc_a = hpss(wav_a)
    harm_b, perc_b = hpss(wav_b)
    print("      HPSS terminé.")
 
    # ── 3. Meilleure fenêtre de transition ──────────────────
    print("\n[3/5] Recherche de la meilleure fenêtre de transition...")
    window_samples = int(morph_sec * SR)
    i_start, j_start = find_best_window(wav_a, wav_b, window_sec=morph_sec)
 
    harm_a_chunk = harm_a[i_start : i_start + window_samples]
    harm_b_chunk = harm_b[j_start : j_start + window_samples]
    perc_a_chunk = perc_a[i_start : i_start + window_samples]
    perc_b_chunk = perc_b[j_start : j_start + window_samples]
 
    # ── 4. Morphing ─────────────────────────────────────────
    print(f"\n[4/5] Morphing ({n_steps} frames)...")
    chunk_size = window_samples // n_steps
    morph_frames = []
 
    for i in tqdm(range(n_steps), desc="      Morphing"):
        t = i / (n_steps - 1) if n_steps > 1 else 0.0
        s = i * chunk_size
        e = s + chunk_size
 
        h_a = harm_a_chunk[s:e]
        h_b = harm_b_chunk[s:e]
        p_a = perc_a_chunk[s:e]
        p_b = perc_b_chunk[s:e]
 
        # Harmonique : phase vocoder (préserve la cohérence de phase)
        h_t = phase_vocoder_interp(h_a, h_b, t)
 
        # Percussive : crossfade simple (les transitoires sont abrupts par nature)
        p_t = crossfade_signals(p_a, p_b, t)
 
        # Recombiner
        n = min(len(h_t), len(p_t))
        frame = h_t[:n] + p_t[:n]
        morph_frames.append(frame)
 
    morph_audio = np.concatenate(morph_frames)
 
    # ── 5. Assemblage ────────────────────────────────────────
    print("\n[5/5] Assemblage final...")
    pre_transition  = wav_a[:i_start]
    post_transition = wav_b[j_start + window_samples:]
 
    output = crossfade_join(pre_transition, morph_audio, fade_sec=0.1)
    output = crossfade_join(output, post_transition, fade_sec=0.1)
 
    out_path = os.path.join(output_dir, "morphing_hpss.wav")
    save_audio(out_path, output)
 
    # Sauvegarde aussi les composantes pour diagnostic
    save_audio(os.path.join(output_dir, "debug_harm_a.wav"), harm_a[i_start:i_start+window_samples])
    save_audio(os.path.join(output_dir, "debug_harm_b.wav"), harm_b[j_start:j_start+window_samples])
    save_audio(os.path.join(output_dir, "debug_perc_a.wav"), perc_a[i_start:i_start+window_samples])
    save_audio(os.path.join(output_dir, "debug_perc_b.wav"), perc_b[j_start:j_start+window_samples])
 
    print(f"\n{'─'*55}")
    print(f"  Durées :")
    print(f"    Pre-transition  : {len(pre_transition)/SR:.1f}s")
    print(f"    Morphing        : {len(morph_audio)/SR:.1f}s")
    print(f"    Post-transition : {len(post_transition)/SR:.1f}s")
    print(f"    Total           : {len(output)/SR:.1f}s")
    print(f"{'─'*55}")
    print(f"""
  Fichiers de debug générés :
    debug_harm_a/b.wav → composantes harmoniques des fenêtres choisies
    debug_perc_a/b.wav → composantes percussives des fenêtres choisies
 
  Si debug_harm_a sonne bien → HPSS fonctionne correctement
  Si la transition sonne encore mal → ajuster --morph_sec ou --n_steps
""")
 
 
if __name__ == "__main__":
    music1_path = "./musique/music1.wav"  # chemin par défaut
    music2_path = "./musique/music2.wav"  # chemin par défaut
    output_dir = "./morphing"  # répertoire de sortie par défaut
    morph_seconds = 5.0  # durée de morphing par défaut
    
    n_steps = 20  # nombre de steps dans le morphing par défaut
    main(music1_path, music2_path, output_dir,
         morph_seconds, n_steps)