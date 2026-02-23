import torch
import torchaudio
import soundfile as sf
import numpy as np
from diffusers import AudioLDMPipeline
from pydub import AudioSegment
import tempfile
import os

# -----------------------------
# CONFIG
# -----------------------------
MP3_INPUT = "input.mp3"
MP3_OUTPUT = "edited.mp3"
PROMPT = "make it slower, darker, ambient, remove drums"
TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD MP3 → WAV (temp)
# -----------------------------
audio = AudioSegment.from_mp3(MP3_INPUT)
audio = audio.set_channels(1).set_frame_rate(TARGET_SR)

samples = np.array(audio.get_array_of_samples()).astype(np.float32)
samples /= np.iinfo(audio.array_type).max  # normalize to [-1, 1]

waveform = torch.tensor(samples).unsqueeze(0)

# -----------------------------
# LOAD MODEL
# -----------------------------
pipe = AudioLDMPipeline.from_pretrained(
    "cvssp/audioldm2-music", torch_dtype=torch.float16
).to(DEVICE)

pipe.enable_xformers_memory_efficient_attention()

# -----------------------------
# AUDIO → LATENT (INVERSION)
# -----------------------------
with torch.no_grad():
    latents = pipe.vae.encode(waveform.to(DEVICE)).latent_dist.sample()

# -----------------------------
# PROMPT-GUIDED EDITING
# -----------------------------
edited_audio = pipe(
    prompt=PROMPT,
    latents=latents,
    num_inference_steps=100,
    guidance_scale=4.5,
    audio_length_in_s=waveform.shape[-1] / TARGET_SR,
).audios[0]

# -----------------------------
# SAVE TEMP WAV
# -----------------------------
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp_wav = tmp.name
    sf.write(tmp_wav, edited_audio, TARGET_SR)

# -----------------------------
# WAV → MP3
# -----------------------------
edited_segment = AudioSegment.from_wav(tmp_wav)
edited_segment.export(MP3_OUTPUT, format="mp3", bitrate="192k")

os.remove(tmp_wav)

print("✅ MP3 music editing done:", MP3_OUTPUT)
