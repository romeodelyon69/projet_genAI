# Music Style Transfer — Multimodal GenAI Project

Training-free music style transfer: given a **content** audio and a **style** audio, we seek to produce an output that preserves the melodic/rhythmic structure of the content while adopting the timbre of the style.

Three models are implemented and compared.

---

## Repository structure

```
projet_genAI/
├── musicTI_dataset/              # Shared dataset (content/ + timbre/)
├── evaluation_comparison.py      # Cross-model evaluation script
│
├── stylus/                       # Model 1 — Stylus
├── stylus_outputs/               # Stylus outputs
├── stylus-audio_LDM2/            # Model 2 — StylusAudioLDM2
├── stylus-audio_LDM2_outputs/    # StylusAudioLDM2 outputs
├── musicLDM/                     # Model 3 — MusicLDM
├── musicLDM_outputs/             # MusicLDM outputs
│
└── demo/                         # Flask web demo
    ├── app.py                    # Demo entry point
    ├── templates/                # HTML templates
```

---

## The 3 models

### 1. Stylus — `stylus/`
Re-implementation of [Stylus (arxiv:2411.15913)](https://arxiv.org/abs/2411.15913).  
Repurposes **Stable Diffusion 1.5** for audio style transfer via attention injection on mel-spectrograms.

- **Mechanism**: DDIM inversion of style (captures K, V) + inversion of content (captures Q) + AdaIN on latents + DDIM reverse with injection Q←γ·Q_content + (1−γ)·Q_current and out ← out_content + α·(out_style − out_content)
- **Key hyperparameters**: `alpha` (style strength), `gamma` (content preservation)
- **TTS / tuning**: `tts_grid_search.py` — 8α × 8γ = 64-run grid 
- **Outputs**: `stylus_outputs/`
- **Scoring**: `score_combined.py`, `scores_clap.py`, `plot_scores.py`

### 2. StylusAudioLDM2 — `stylus-audio_LDM2/`
Adaptation of Stylus to **AudioLDM2-music** (`cvssp/audioldm2-music`).  
Same attention-injection principle, but operating in the AudioLDM2 latent space (64-bin mel, hop=160, fmax=8000 Hz).

- **Mechanism**: same as Stylus but conditioning uses CLAP audio/text (T5 + GPT-2) instead of CLIP text
- **Versions**: `stylus_audioldm2_v4.py` (stable), `stylus_audioldm2_v5.py` (AdaIN fix + asymmetric attention)
- **TTS / tuning**: `tts_grid_search.py` — 8α × 8γ = 64-run grid, results in `stylus-audio_LDM2_outputs/grid_search_output/`
- **Outputs**: `stylus-audio_LDM2_outputs/`

### 3. MusicLDM — `musicLDM/`
Style transfer via **MusicLDM** (`ucsd-reach/musicldm`) leveraging the model's stochasticity for a Best-of-N strategy.

- **Mechanism**: encode content to latent, add_noise(z, strength), guided_denoise conditioned on CLAP(style)
- **Key hyperparameters**: `strength` (0=stay close to content, 1=full restyle), `guidance_scale`
- **TTS / tuning**: `tts_best_of_n.py` — generates N=64 stochastic samples, estimates E[best-of-N] by Monte Carlo for N∈[1,2,4,8,16,32,64]
- **Outputs**: `musicLDM_outputs/bon_musicldm/`

---

## Cross-model evaluation — `evaluation_comparison.py`

Compares all 3 models on (content, style) pairs drawn from `musicTI_dataset/`.

**Metrics computed per pair:**
| Metric | Description |
|---|---|
| `clap_style` | cos(e_out, e_style) |
| `clap_content` | cos(e_out, e_content) |
| `clap_directional` | cos(e_out−e_content, e_style−e_content) — main metric |
| `combined` | λ·mel_style_score + (1−λ)·mfcc_content_score |

---

## Dataset — `musicTI_dataset/`

```
musicTI_dataset/
├── content/     # Categories: hiphop, violin, piano, adventure, color, ...
└── timbre/      # Categories: accordion, bird, chime, clarinet, erhu, ...
└── other/       # Other content musics : TheFatRat Unity
```

Each category contains ~15 WAV files (16 kHz, mono).

---

## Running a model

```bash
# Stylus
python stylus/run_stylus.py

# StylusAudioLDM2 (single transfer)
python stylus-audio_LDM2/stylus_audioldm2_v4.py

# StylusAudioLDM2 (grid search TTS)
python stylus-audio_LDM2/tts_grid_search.py

# MusicLDM (single transfer)
python musicLDM/musicldm_style_transfer.py

# MusicLDM (Best-of-N)
python musicLDM/tts_best_of_n.py

# Cross-model evaluation
python evaluation_comparison.py
```

---

## Interactive demo — `app.py`

A Flask web interface lets you run and compare the 3 models side-by-side directly from your browser.

### Launch

```bash
python demo/app.py
```

Then open **http://localhost:5001** in your browser.
