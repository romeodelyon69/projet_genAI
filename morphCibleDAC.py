import torch
import torchaudio
import dac
import dac.utils

device = "cuda" if torch.cuda.is_available() else "cpu"
SR = 44100
CHUNK_SEC = 4        # Durée du chunk de transition en secondes (~2-4s)
CROSSFADE_SEC = 0.2  # Fondu crossfade en secondes
CROSSFADE = int(SR * CROSSFADE_SEC)

# =========================
# Charger modèle DAC
# =========================
model_path = dac.utils.download("44khz")
model = dac.DAC.load(model_path).to(device)
model.eval()

# =========================
# Préparation audio
# =========================
def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # stéréo → mono
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)
    waveform = waveform / waveform.abs().max()
    return waveform

audio1 = load_audio("music1.wav")
audio2 = load_audio("music2.wav")

# =========================
# Encoder toute la musique
# =========================
with torch.no_grad():
    z1, *_ = model.encode(audio1.unsqueeze(0).to(device))
    z2, *_ = model.encode(audio2.unsqueeze(0).to(device))

print("Latent shapes:", z1.shape, z2.shape)

# =========================
# Trouver meilleure fenêtre de transition
# =========================
def find_best_transition_point(z1, z2, window_size_steps=2000):
    """
    Parcourt z1 et z2 pour trouver la meilleure correspondance L2
    window_size_steps: taille de la fenêtre en pas latents
    """
    T1, T2 = z1.shape[2], z2.shape[2]
    min_dist = float('inf')
    best_idx = (0, 0)
    step = max(1, window_size_steps // 10)

    for i in range(0, T1 - window_size_steps, step):
        feat1 = z1[:, :, i : i + window_size_steps]
        for j in range(0, T2 - window_size_steps, step):
            feat2 = z2[:, :, j : j + window_size_steps]
            dist = torch.norm(feat1 - feat2, p=2)
            if dist < min_dist:
                min_dist = dist
                best_idx = (i, j)
    return best_idx

# Choisir chunk latent
window_size_latents = min(2000, z1.shape[2], z2.shape[2])
start1, start2 = find_best_transition_point(z1, z2, window_size_steps=window_size_latents)
print("Meilleure transition:", start1, start2)

# extraire les chunks latents
z1_chunk = z1[:, :, start1 : start1 + window_size_latents]
z2_chunk = z2[:, :, start2 : start2 + window_size_latents]

# s'assurer que z1_chunk et z2_chunk ont même taille
min_len = min(z1_chunk.shape[2], z2_chunk.shape[2])
z1_chunk = z1_chunk[:, :, :min_len]
z2_chunk = z2_chunk[:, :, :min_len]

# =========================
# SLERP interpolation
# =========================
def slerp(zA, zB, t):
    z1_flat = zA.flatten()
    z2_flat = zB.flatten()
    dot = torch.dot(z1_flat, z2_flat) / (torch.norm(z1_flat) * torch.norm(z2_flat))
    omega = torch.acos(torch.clamp(dot, -1, 1))
    so = torch.sin(omega)
    if so == 0:
        return (1 - t) * zA + t * zB
    return torch.sin((1 - t) * omega) / so * zA + torch.sin(t * omega) / so * zB

# =========================
# Morphing chunk par chunk avec crossfade
# =========================
steps = 20  # nombre d’étapes dans la fenêtre de transition
morph_chunks = []

with torch.no_grad():
    prev_chunk_audio = None
    for i in range(steps):
        t = i / (steps - 1)
        z = slerp(z1_chunk, z2_chunk, t)
        recon = model.decode(z).squeeze(0).cpu()
        # crossfade avec chunk précédent
        if prev_chunk_audio is not None:
            fade_in = torch.linspace(0, 1, CROSSFADE)
            fade_out = 1 - fade_in
            recon[:, :CROSSFADE] = prev_chunk_audio[:, -CROSSFADE:] * fade_out + recon[:, :CROSSFADE] * fade_in
            morph_chunks.append(recon[:, CROSSFADE:])
        else:
            morph_chunks.append(recon)
        prev_chunk_audio = recon

morph_segment = torch.cat(morph_chunks, dim=1)

# =========================
# Construire audio final
# =========================
pre_transition = audio1[:, : start1].cpu()
post_transition = audio2[:, start2 + window_size_latents :].cpu()

output_audio = torch.cat([pre_transition, morph_segment, post_transition], dim=1)

# =========================
# Sauvegarde
# =========================
torchaudio.save("morphing_best_window_crossfade.wav", output_audio, SR)
print("Morphing terminé → morphing_best_window_crossfade.wav")