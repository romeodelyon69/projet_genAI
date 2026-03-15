"""
Music2Latent Morphing — Lin-CAE stable version
-----------------------------------------------

Optimisations :
- encodage chunké (10s)
- torch.inference_mode()
- float32 (compatibilité FFT)
- morphing SLERP
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import torch


# ------------------------------------------------
# Path vers Linear Autoencoders
# ------------------------------------------------

_LINEAR_AE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "linear-autoencoders",
    "src"
)

if os.path.exists(_LINEAR_AE_PATH) and _LINEAR_AE_PATH not in sys.path:
    sys.path.insert(0, _LINEAR_AE_PATH)

from linear_cae import Autoencoder as EncoderDecoder


# ------------------------------------------------
# Config
# ------------------------------------------------

SR = 44100
LATENT_SR = 10
CHUNK_SEC = 10


# ------------------------------------------------
# Audio I/O
# ------------------------------------------------

def load_audio(path):

    wav, _ = librosa.load(path, sr=SR, mono=True)

    wav = wav / (np.abs(wav).max() + 1e-8)

    return wav


def save_audio(path, wav):

    wav = wav / (np.abs(wav).max() + 1e-8)

    sf.write(path, wav, SR)

    print("saved:", path)


# ------------------------------------------------
# Encode chunked
# ------------------------------------------------

def encode_chunked(model, wav):

    device = next(model.parameters()).device

    chunk_samples = CHUNK_SEC * SR

    latents = []

    with torch.inference_mode():

        for start in range(0, len(wav), chunk_samples):

            chunk = wav[start:start + chunk_samples]

            if len(chunk) < SR:
                break

            wav_t = torch.from_numpy(chunk).float().unsqueeze(0).to(device)

            z = model.encode(wav_t)

            z = z.detach().cpu().numpy()

            latents.append(z)

            del wav_t, z

    torch.cuda.empty_cache()

    return np.concatenate(latents, axis=2)


# ------------------------------------------------
# Decode
# ------------------------------------------------

def decode(model, latent, full_length=None):

    device = next(model.parameters()).device

    with torch.inference_mode():

        if not isinstance(latent, torch.Tensor):
            latent = torch.from_numpy(latent).float().to(device)

        wav = model.decode(latent, full_length=full_length)

    wav = wav.detach().cpu().numpy()

    if wav.ndim > 1:
        wav = wav.squeeze()

    torch.cuda.empty_cache()

    return wav.astype(np.float32)


# ------------------------------------------------
# SLERP interpolation
# ------------------------------------------------

def slerp(a, b, t):

    a_np = a.flatten().astype(np.float64)
    b_np = b.flatten().astype(np.float64)

    dot = np.dot(a_np, b_np) / (
        np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-8
    )

    omega = np.arccos(np.clip(dot, -1, 1))

    so = np.sin(omega)

    if abs(so) < 1e-5:
        return (1 - t) * a + t * b

    res = (
        np.sin((1 - t) * omega) / so * a_np +
        np.sin(t * omega) / so * b_np
    )

    return res.reshape(a.shape)


# ------------------------------------------------
# Crossfade
# ------------------------------------------------

def crossfade(a, b, fade_sec=0.05):

    fade = int(fade_sec * SR)

    fade = min(fade, len(a), len(b))

    if fade == 0:
        return np.concatenate([a, b])

    env_out = np.linspace(1, 0, fade)
    env_in = np.linspace(0, 1, fade)

    overlap = a[-fade:] * env_out + b[:fade] * env_in

    return np.concatenate([a[:-fade], overlap, b[fade:]])


# ------------------------------------------------
# MFCC window search
# ------------------------------------------------

def find_best_window(a, b, window_sec):

    hop = 512

    win = int(window_sec * SR)

    step = int(1 * SR)

    win_f = win // hop
    step_f = step // hop

    mfcc_a = librosa.feature.mfcc(y=a, sr=SR, n_mfcc=20, hop_length=hop)
    mfcc_b = librosa.feature.mfcc(y=b, sr=SR, n_mfcc=20, hop_length=hop)

    best = (0, 0)
    best_dist = 1e12

    for i in tqdm(range(5*win_f, mfcc_a.shape[1] - win_f, step_f)):

        ca = mfcc_a[:, i:i + win_f]

        for j in range(0, mfcc_b.shape[1] - 5 * win_f, step_f):

            cb = mfcc_b[:, j:j + win_f]

            d = np.linalg.norm(ca - cb)

            if d < best_dist:

                best_dist = d

                best = (i * hop, j * hop)

    return best


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main(path_a, path_b, output_dir, morph_sec=5, n_steps=20):

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    model = EncoderDecoder.from_pretrained(
        "lin-cae",
        max_chunk_size=SR * 10,
        overlap_percentage=0.1,
        max_batch_size=1
    ).eval().to(device)

    # ------------------------------------------------
    # Audio
    # ------------------------------------------------

    wav_a = load_audio(path_a)
    wav_b = load_audio(path_b)

    print("A:", len(wav_a) / SR, "sec")
    print("B:", len(wav_b) / SR, "sec")

    # ------------------------------------------------
    # Encode
    # ------------------------------------------------

    print("encoding...")

    zA = encode_chunked(model, wav_a)
    zB = encode_chunked(model, wav_b)

    print("zA:", zA.shape)
    print("zB:", zB.shape)

    latent_hop = len(wav_a) // zA.shape[2]

    # ------------------------------------------------
    # Find transition
    # ------------------------------------------------

    i_audio, j_audio = find_best_window(wav_a, wav_b, morph_sec)

    time_a = i_audio / SR
    time_b = j_audio / SR
    print(f"Best transition found at {time_a:.2f}s (A) and {time_b:.2f}s (B)")

    win = int(morph_sec * SR)

    i_lat = int(i_audio / latent_hop)
    j_lat = int(j_audio / latent_hop)

    win_lat = int(win / latent_hop)

    # ------------------------------------------------
    # Morphing
    # ------------------------------------------------

    zA_chunk = zA[:, :, i_lat:i_lat + win_lat]
    zB_chunk = zB[:, :, j_lat:j_lat + win_lat]

    chunk = win_lat // n_steps

    morph_latents = []

    for i in tqdm(range(n_steps)):

        t = 0.5 - 0.5 * np.cos(np.pi * i / (n_steps - 1))

        s = i * chunk
        e = s + chunk

        if i == n_steps - 1:
            e = win_lat

        za = zA_chunk[:, :, s:e]
        zb = zB_chunk[:, :, s:e]

        zm = slerp(za, zb, t)

        morph_latents.append(zm)

    morph_latents = np.concatenate(morph_latents, axis=2)

    morph_audio = decode(
        model,
        morph_latents,
        full_length=int(morph_latents.shape[2] * latent_hop)
    )

    # ------------------------------------------------
    # Assemble
    # ------------------------------------------------

    pre = wav_a[:i_audio]
    post = wav_b[j_audio + win:]

    out = crossfade(pre, morph_audio)
    out = crossfade(out, post)

    save_audio(os.path.join(output_dir, "morph.wav"), out)


# ------------------------------------------------

if __name__ == "__main__":

    main(
        "./musique/music1.wav",
        "./musique/music4.wav",
        "./morph_output",
        morph_sec=10,
        n_steps=20
    )