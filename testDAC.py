import torch
import torchaudio
import dac
import dac.utils

device = "cuda" if torch.cuda.is_available() else "cpu"

# télécharger modèle
model_path = dac.utils.download("44khz")

# charger modèle
model = dac.DAC.load(model_path).to(device)

# charger audio
waveform, sr = torchaudio.load("Vois-sur-ton-chemin.wav")

# stéréo -> mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# resample
if sr != 44100:
    waveform = torchaudio.functional.resample(waveform, sr, 44100)

# normalisation
waveform = waveform / waveform.abs().max()

waveform = waveform.unsqueeze(0).to(device)

with torch.no_grad():

    # encode
    z, codes, latents, *_ = model.encode(waveform)

    print("latent shape:", z.shape)

    # decode
    reconstruction = model.decode(z)

torchaudio.save(
    "reconstruction.wav",
    reconstruction.squeeze(0).cpu(),
    44100
)

print("Reconstruction terminée")