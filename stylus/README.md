# Stylus — Training-Free Music Style Transfer

Réimplémentation PyTorch du papier :
**"Stylus: Repurposing Stable Diffusion for Training-Free Music Style Transfer on Mel-Spectrograms"**
(arxiv:2411.15913)

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate
pip install librosa soundfile pillow
```

## Usage rapide

```python
from stylus import StylusConfig, StylusPipeline
import librosa, soundfile as sf

# Charger les audios
style_audio,   sr = librosa.load("style.wav",   sr=22050, mono=True, duration=5.0)
content_audio, _  = librosa.load("content.wav", sr=22050, mono=True, duration=5.0)

# Pipeline
cfg = StylusConfig(
    alpha=0.6,                    # query preservation (0=pur style, 1=pur content)
    num_inference_steps=50,
    attention_temperature=0.5,    # sharpen attention maps
    device="cuda",
)
pipeline = StylusPipeline(cfg).load_model()
output = pipeline.transfer(style_audio, content_audio)

sf.write("output.wav", output, sr)
```

## CLI

```bash
python run_stylus.py \
    --style   accordion_sample.wav \
    --content piano_melody.wav \
    --output  piano_in_accordion_style.wav \
    --alpha   0.6 \
    --steps   50
```

## Tests

```bash
python test_stylus.py
```

## Architecture

```
Audio (style)                 Audio (content)
    │                               │
    ▼                               ▼
Mel-spectrogram               Mel-spectrogram
    │                               │
    ▼                               ▼
Image RGB [-1,1]              Image RGB [-1,1]
    │                               │
    ▼                               ▼
VAE encode → z_s              VAE encode → z_c
    │                               │
    ▼                               ▼
DDIM Inversion                DDIM Inversion
(capture K,V self-attn)       (mode off)
    │                               │
    │              AdaIN ◄──────────┤
    │           z_output_T          │
    │                               │
    └──── inject K,V ──►  DDIM Reverse z_T → z_0
                               │
                               ▼
                         VAE decode → image
                               │
                               ▼
                         Image → Mel → Audio (Griffin-Lim)
```

## Hyperparamètres clés

| Paramètre | Valeur papier | Description |
|-----------|--------------|-------------|
| `alpha`   | 0.6          | Préservation de la query : 1=garde structure content, 0=full style |
| `attention_temperature` | 0.5 | Sharpening des attention maps |
| `target_up_blocks` | [0,1] | Couches U-Net ciblées (decoder layers 7-12) |
| `num_inference_steps` | 50 | Steps DDIM |

## Notes d'implémentation

- **Couches cibles** : `up_blocks[1]` et `up_blocks[2]` du U-Net SD 1.5 (résolutions 16x16 et 32x32)
  — correspond aux "decoder layers 7–12" mentionnées dans le papier.
- **AdaIN initial** : modulation channel-wise des stats du latent bruit content → style.
- **Query preservation** : `Q_blend = α·Q_content + (1-α)·Q_output` empêche la dérive structurelle.
- **Temperature scaling** : divise le score QKᵀ par `temperature` avant softmax → attention plus concentrée.
- **Vocodeur** : le pipeline utilise Griffin-Lim par défaut.
  Pour meilleure qualité audio, remplacer `mel_to_audio()` par HiFi-GAN.
