import numpy as np
import torch
import soundfile as sf
import librosa

from stylus import StylusConfig, StylusPipeline


def load_audio(path: str, target_sr: int, duration: float = 5.0) -> np.ndarray:
    audio, sr = librosa.load(path, sr=target_sr, mono=True, duration=duration)
    target_len = int(target_sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    return audio.astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sr: int):
    audio_norm = audio / (np.abs(audio).max() + 1e-8) * 0.9
    sf.write(path, audio_norm, sr)
    print(f"Saved: {path}")


def main():
    # ── Paramètres ────────────────────────────────────────────────────────────
    style    = "chime.wav"
    content  = "accordion.wav"
    output   = "stylized_output.wav"
    save_dir = "./stylus_outputs"
    duration = 5.0

    alpha          = 0.5    # style guidance : 0=content pur, 1=style pur
    steps          = 50
    up_blocks      = [1, 2, 3]   # couches self-attention ciblées
    model_id       = "runwayml/stable-diffusion-v1-5"
    device         = "cuda" if torch.cuda.is_available() else "cpu"
    fp32           = False

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = StylusConfig(
        alpha=alpha,
        num_inference_steps=steps,
        target_up_block_indices=up_blocks,
        model_id=model_id,
        device=device,
        dtype=torch.float32 if fp32 or device == "cpu" else torch.float16,
    )

    # ── Chargement audio ──────────────────────────────────────────────────────
    print(f"Loading style  : {style}")
    print(f"Loading content: {content}")
    style_audio   = load_audio(style,   cfg.sample_rate, duration)
    content_audio = load_audio(content, cfg.sample_rate, duration)

    print(f"\nConfig:")
    print(f"  α (style guidance)   = {cfg.alpha}")
    print(f"  DDIM steps           = {cfg.num_inference_steps}")
    print(f"  Target up_blocks     = {cfg.target_up_block_indices}")
    print(f"  Device               = {cfg.device} / {cfg.dtype}")
    print()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = StylusPipeline(cfg)
    pipeline.load_model()

    output_audio = pipeline.transfer(
        style_audio,
        content_audio,
        save_dir=save_dir,
    )

    save_audio(output, output_audio, cfg.sample_rate)


if __name__ == "__main__":
    main()