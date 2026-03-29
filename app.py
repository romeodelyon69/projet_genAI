#!/usr/bin/env python3
"""
Flask web app for comparing 3 audio style-transfer models.
Run with:  python app.py
Then open: http://localhost:5000
"""

import os
import sys
import uuid
import traceback
import threading

import numpy as np
import librosa
import soundfile as sf
import torch

from flask import Flask, render_template, jsonify, request, send_from_directory

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "stylus"))
sys.path.insert(0, os.path.join(ROOT, "musicLDM"))
sys.path.insert(0, os.path.join(ROOT, "stylus-audio_LDM2"))

DATASET_ROOT = os.path.join(ROOT, "musicTI_dataset", "audios")
CONTENT_DIR = os.path.join(DATASET_ROOT, "content")
STYLE_DIR = os.path.join(DATASET_ROOT, "timbre")
OUTPUT_DIR = os.path.join(ROOT, "static", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

# ── Hyperparameters (same as evaluation_comparison.py) ───────────────────────
STYLUS_ALPHA = 0.9
STYLUS_GAMMA = 0.1
STYLUS_STEPS = 50
STYLUS_DURATION = 5.0

AUDIOLDM2_ALPHA = 0.9
AUDIOLDM2_GAMMA = 0.1
AUDIOLDM2_STEPS = 50

MUSICLDM_STRENGTH = 0.7
MUSICLDM_GUIDANCE_SCALE = 10
MUSICLDM_STEPS = 50

# ── VRAM management ──────────────────────────────────────────────────────────
_models_lock = threading.Lock()


def clear_vram():
    """Release all GPU memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Audio helpers ────────────────────────────────────────────────────────────
def save_wav(path, audio, sr):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.9
    sf.write(path, audio.astype(np.float32), sr)


def list_audio_files_in(directory):
    exts = {".wav", ".mp3", ".flac", ".ogg"}
    return sorted(f for f in os.listdir(directory)
                  if os.path.splitext(f)[1].lower() in exts)


# ── Model loading (fresh each time, freed after use) ────────────────────────
def load_stylus():
    from stylus import StylusConfig, StylusPipeline
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = StylusConfig(
        alpha=STYLUS_ALPHA, gamma=STYLUS_GAMMA,
        num_inference_steps=STYLUS_STEPS, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    pipe = StylusPipeline(cfg)
    pipe.load_model()
    return pipe


def load_audioldm2():
    from stylus_audioldm2_v4 import StylusAudioLDM2Config, StylusAudioLDM2Pipeline
    cfg = StylusAudioLDM2Config(
        alpha=AUDIOLDM2_ALPHA, gamma=AUDIOLDM2_GAMMA,
        num_inference_steps=AUDIOLDM2_STEPS,
        skip_roundtrip_check=True,
    )
    pipe = StylusAudioLDM2Pipeline(cfg)
    pipe.load_model()
    return pipe


def load_musicldm():
    import musicldm_style_transfer as mldm
    from diffusers import MusicLDMPipeline, DDIMScheduler
    pipe = MusicLDMPipeline.from_pretrained(
        mldm.MODEL_ID, torch_dtype=torch.float32,
    ).to(mldm.DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe, mldm


# ── Model runners (accept params dict) ───────────────────────────────────────
def run_stylus(content_path, style_path, out_dir, params):
    alpha = params.get("alpha", STYLUS_ALPHA)
    gamma = params.get("gamma", STYLUS_GAMMA)
    steps = params.get("steps", STYLUS_STEPS)
    duration = params.get("duration", STYLUS_DURATION)

    with _models_lock:
        pipe = load_stylus()
        pipe.cfg.alpha = alpha
        pipe.cfg.gamma = gamma
        pipe.cfg.num_inference_steps = steps
        sr = pipe.cfg.sample_rate

        def _load(p):
            wav, _ = librosa.load(p, sr=sr, mono=True, duration=duration)
            target = int(sr * duration)
            if len(wav) < target:
                wav = np.pad(wav, (0, target - len(wav)))
            return wav.astype(np.float32)

        audio_out = pipe.transfer(_load(style_path), _load(content_path), save_dir=out_dir)
        out_path = os.path.join(out_dir, "output.wav")
        save_wav(out_path, audio_out, sr)

        del pipe
        clear_vram()
    return out_path


def run_audioldm2(content_path, style_path, out_dir, params):
    alpha = params.get("alpha", AUDIOLDM2_ALPHA)
    gamma = params.get("gamma", AUDIOLDM2_GAMMA)
    steps = params.get("steps", AUDIOLDM2_STEPS)

    with _models_lock:
        pipe = load_audioldm2()
        pipe.cfg.alpha = alpha
        pipe.cfg.gamma = gamma
        pipe.cfg.num_inference_steps = steps
        pipe.transfer(style_path=style_path, content_path=content_path, output_dir=out_dir)
        out_path = os.path.join(out_dir, "output.wav")
        for f in os.listdir(out_dir):
            if f.startswith("stylized_") and f.endswith(".wav"):
                out_path = os.path.join(out_dir, f)
                break

        del pipe
        clear_vram()
    return out_path


def run_musicldm(content_path, style_path, out_dir, params):
    strength = params.get("strength", MUSICLDM_STRENGTH)
    guidance_scale = params.get("guidance_scale", MUSICLDM_GUIDANCE_SCALE)
    steps = params.get("steps", MUSICLDM_STEPS)

    with _models_lock:
        pipe, mldm = load_musicldm()
        sr = mldm.SAMPLE_RATE
        do_cfg = guidance_scale > 1.0

        def _pad(wav):
            if len(wav) < mldm.CHUNK_SAMPLES:
                rep = int(np.ceil(mldm.CHUNK_SAMPLES / len(wav)))
                return np.tile(wav, rep)[:mldm.CHUNK_SAMPLES]
            return wav[:mldm.CHUNK_SAMPLES]

        wav_content = _pad(mldm.load_audio(content_path))
        wav_style = _pad(mldm.load_audio(style_path))

        p_embeds, p_mask = mldm.encode_audio_as_prompt(wav_style, pipe, do_cfg)
        mel = mldm.audio_to_mel(wav_content)
        z0 = mldm.vae_encode(mel, pipe.vae)
        z_n, t_start = mldm.add_noise(z0, pipe.scheduler, strength, steps)
        z_out = mldm.guided_denoise(
            z_n, t_start, pipe, guidance_scale, steps, p_embeds, p_mask
        )
        wav_out = mldm.vae_decode(z_out, pipe.vae, pipe)

        out_path = os.path.join(out_dir, "output.wav")
        save_wav(out_path, wav_out, sr)

        del pipe
        clear_vram()
    return out_path


MODEL_RUNNERS = {
    "Stylus": run_stylus,
    "StylusAudioLDM2": run_audioldm2,
    "MusicLDM": run_musicldm,
}


# ── Flask routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/categories")
def api_categories():
    """Return content and style categories with their audio files."""
    content_cats = {}
    for cat in sorted(os.listdir(CONTENT_DIR)):
        cat_path = os.path.join(CONTENT_DIR, cat)
        if os.path.isdir(cat_path):
            content_cats[cat] = list_audio_files_in(cat_path)

    style_cats = {}
    for cat in sorted(os.listdir(STYLE_DIR)):
        cat_path = os.path.join(STYLE_DIR, cat)
        if os.path.isdir(cat_path):
            style_cats[cat] = list_audio_files_in(cat_path)

    return jsonify({"content": content_cats, "style": style_cats})


@app.route("/api/audio/<path:filepath>")
def api_audio(filepath):
    """Serve an audio file from the dataset."""
    full = os.path.normpath(os.path.join(DATASET_ROOT, filepath))
    if not full.startswith(os.path.normpath(DATASET_ROOT)):
        return "Forbidden", 403
    directory = os.path.dirname(full)
    filename = os.path.basename(full)
    return send_from_directory(directory, filename)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Run all 3 models on the selected content+style pair."""
    data = request.json
    content_cat = data.get("content_category", "")
    content_file = data.get("content_file", "")
    style_cat = data.get("style_category", "")
    style_file = data.get("style_file", "")

    content_path = os.path.normpath(
        os.path.join(CONTENT_DIR, content_cat, content_file))
    style_path = os.path.normpath(
        os.path.join(STYLE_DIR, style_cat, style_file))

    # Security: validate paths stay within dataset
    if not content_path.startswith(os.path.normpath(CONTENT_DIR)):
        return jsonify({"error": "Invalid content path"}), 400
    if not style_path.startswith(os.path.normpath(STYLE_DIR)):
        return jsonify({"error": "Invalid style path"}), 400
    if not os.path.isfile(content_path):
        return jsonify({"error": f"Content file not found: {content_file}"}), 404
    if not os.path.isfile(style_path):
        return jsonify({"error": f"Style file not found: {style_file}"}), 404

    run_id = uuid.uuid4().hex[:8]
    results = {}

    for model_name, runner in MODEL_RUNNERS.items():
        out_dir = os.path.join(OUTPUT_DIR, run_id, model_name)
        os.makedirs(out_dir, exist_ok=True)
        try:
            out_path = runner(content_path, style_path, out_dir)
            rel = os.path.relpath(out_path, os.path.join(ROOT, "static"))
            results[model_name] = {
                "status": "ok",
                "url": f"/static/{rel.replace(os.sep, '/')}",
            }
        except Exception:
            tb = traceback.format_exc()
            print(f"[ERROR] {model_name}:\n{tb}")
            results[model_name] = {"status": "error", "error": tb}

    return jsonify({"run_id": run_id, "results": results})


@app.route("/api/generate/<model_name>", methods=["POST"])
def api_generate_single(model_name):
    """Run a single model (for progressive loading)."""
    if model_name not in MODEL_RUNNERS:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    data = request.json
    content_path = os.path.normpath(
        os.path.join(CONTENT_DIR, data["content_category"], data["content_file"]))
    style_path = os.path.normpath(
        os.path.join(STYLE_DIR, data["style_category"], data["style_file"]))

    if not content_path.startswith(os.path.normpath(CONTENT_DIR)):
        return jsonify({"error": "Invalid content path"}), 400
    if not style_path.startswith(os.path.normpath(STYLE_DIR)):
        return jsonify({"error": "Invalid style path"}), 400
    if not os.path.isfile(content_path):
        return jsonify({"error": "Content file not found"}), 404
    if not os.path.isfile(style_path):
        return jsonify({"error": "Style file not found"}), 404

    run_id = data.get("run_id", uuid.uuid4().hex[:8])
    out_dir = os.path.join(OUTPUT_DIR, run_id, model_name)
    os.makedirs(out_dir, exist_ok=True)

    params = data.get("params", {}).get(model_name, {})

    try:
        out_path = MODEL_RUNNERS[model_name](content_path, style_path, out_dir, params)
        rel = os.path.relpath(out_path, os.path.join(ROOT, "static"))
        return jsonify({
            "status": "ok",
            "model": model_name,
            "url": f"/static/{rel.replace(os.sep, '/')}",
        })
    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR] {model_name}:\n{tb}")
        return jsonify({"status": "error", "model": model_name, "error": tb}), 500


if __name__ == "__main__":
    import signal

    def cleanup(*_):
        print("\nArret du serveur...")
        clear_vram()
        os._exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"\n  Dataset:  {DATASET_ROOT}")
    print(f"  Outputs:  {OUTPUT_DIR}")
    print(f"  Open:     http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
