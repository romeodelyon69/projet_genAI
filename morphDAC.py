import torch
import torchaudio
import dac
import dac.utils

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Charger modèle
# =========================
model_path = dac.utils.download("44khz")
model = dac.DAC.load(model_path).to(device)
model.eval()

SR = 44100
CHUNK_SEC = 8
CHUNK = SR * CHUNK_SEC

def find_best_transition_point(z1, z2, window_size_steps=50):
    """
    Parcourt z1 et z2 pour trouver les indices où la distance est minimale.
    window_size_steps: taille de la fenêtre de comparaison (ex: ~1 seconde)
    """
    # z shape: [1, 1024, T]
    T = z1.shape[2]
    min_dist = float('inf')
    best_idx = (0, 0)

    # Pour gagner du temps, on peut moyenner sur la dimension temporelle 
    # ou comparer par blocs. Ici on fait une corrélation glissante simple.
    for i in range(0, T - window_size_steps, 10): # Step de 10 pour la vitesse
        feat1 = z1[:, :, i : i + window_size_steps]
        
        for j in range(0, T - window_size_steps, 10):
            feat2 = z2[:, :, j : j + window_size_steps]
            
            # Distance L2 entre les deux patterns latents
            dist = torch.norm(feat1 - feat2, p=2)
            
            if dist < min_dist:
                min_dist = dist
                best_idx = (i, j)
                
    return best_idx # Retourne (start_A, start_B)


# =========================
# SLERP interpolation optimisée pour DAC
# =========================
def slerp(z1, z2, t):
    # z shape is [1, 1024, T]
    # On normalise sur la dimension des features (1024)
    z1_norm = z1 / torch.norm(z1, dim=1, keepdim=True)
    z2_norm = z2 / torch.norm(z2, dim=1, keepdim=True)
    
    # Calcul du produit scalaire par pas temporel
    dot = (z1_norm * z2_norm).sum(dim=1, keepdim=True) # [1, 1, T]
    dot = torch.clamp(dot, -1.0, 1.0)
    
    omega = torch.acos(dot)
    so = torch.sin(omega)
    
    # Gestion du cas où les vecteurs sont identiques (so=0)
    # On utilise une condition masque pour éviter les divisions par zéro
    res = torch.where(
        so < 1e-6,
        (1 - t) * z1 + t * z2,
        (torch.sin((1 - t) * omega) / so) * z1 + (torch.sin(t * omega) / so) * z2
    )
    return res


# =========================
# Charger audio
# =========================
def load_audio(path):

    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)

    waveform = waveform / waveform.abs().max()

    return waveform


audio1 = load_audio("music1.wav")
audio2 = load_audio("music2.wav")

# longueur commune
min_len = min(audio1.shape[1], audio2.shape[1])
audio1 = audio1[:, :min_len]
audio2 = audio2[:, :min_len]

# =========================
# découpe en chunks
# =========================
chunks = min_len // CHUNK

output_audio = []

with torch.no_grad():

    for i in range(chunks):

        start = i * CHUNK
        end = start + CHUNK

        chunk1 = audio1[:, start:end].unsqueeze(0).to(device)
        chunk2 = audio2[:, start:end].unsqueeze(0).to(device)

        # encode
        z1, *_ = model.encode(chunk1)
        z2, *_ = model.encode(chunk2)

        # progression morph
        t = i / (chunks - 1)

        z = slerp(z1, z2, t)

        # decode
        recon = model.decode(z)

        output_audio.append(recon.squeeze(0).cpu())

# concat final
output = torch.cat(output_audio, dim=1)

torchaudio.save("morphing_slerp.wav", output, SR)

print("Morphing terminé → morphing_slerp.wav")