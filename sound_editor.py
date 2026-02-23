import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from diffusers import AudioLDMPipeline
from pydub import AudioSegment

# -----------------------------
# CONFIG
# -----------------------------
MP3_INPUT = "input.mp3"
MP3_OUTPUT = "edited.mp3"
PROMPT = "make it slower, darker, ambient and clearer"
TARGET_SR = 16000

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# -----------------------------
# LOAD MP3 (duration reference)
# -----------------------------
audio = AudioSegment.from_mp3(MP3_INPUT)
audio = audio.set_channels(1).set_frame_rate(TARGET_SR)

samples = np.array(audio.get_array_of_samples()).astype(np.float32)
samples /= np.iinfo(audio.array_type).max
audio_length_in_s = len(samples) / TARGET_SR

# -----------------------------
# LOAD MODEL
# -----------------------------
pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-m-full", torch_dtype=DTYPE).to(
    DEVICE
)

if DEVICE == "cuda":
    try:
        import xformers  # noqa: F401

        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

# -----------------------------
# PROMPT-GUIDED GENERATION
# -----------------------------
edited_audio = pipe(
    prompt=PROMPT,
    num_inference_steps=100,
    guidance_scale=4.5,
    audio_length_in_s=audio_length_in_s,
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
