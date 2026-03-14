"""
Stylus: Repurposing Stable Diffusion for Training-Free Music Style Transfer
Implémentation fidèle du papier arxiv:2411.15913

Deux éléments critiques du papier correctement implémentés :

1. CFG-inspired attention interpolation (pas un simple remplacement K/V) :
      out_content = Attention(Q, K_content, V_content)   ← forward normal
      out_style   = Attention(Q, K_style,   V_style)     ← K/V du style
      out_final   = out_content + α*(out_style - out_content)

2. Phase-preserving reconstruction :
      - On garde la phase STFT originale du content
      - On combine phase_content + magnitude_stylisée via ISTFT
      - Pas de Griffin-Lim, pas de vocodeur
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StylusConfig:
    # α : style guidance scale (0 = content pur, 1 = style pur)
    # Le papier appelle ça le "style guidance scale" inspiré de CFG
    alpha: float = 0.5

    # Couches U-Net SD1.5 ciblées (self-attention uniquement)
    # up_blocks[1] = résolution 8×8, up_blocks[2] = 16×16, up_blocks[3] = 32×32
    target_up_block_indices: list = field(default_factory=lambda: [1, 2, 3])

    # DDIM
    num_inference_steps: int = 50

    # STFT (pour mel + phase preservation)
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 512
    fmin: float = 0.0
    fmax: float = 8000.0
    target_length: int = 512   # largeur image SD (512px)

    # Modèle
    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# AttentionStore — stocke K, V du style ET du content
# ─────────────────────────────────────────────────────────────────────────────

class AttentionStore:
    """
    Stocke K et V pour le style ET le content séparément.
    En mode inject : calcule les deux sorties attention et les interpole (CFG).

    Modes :
      capture_style   → stocke K_style,   V_style
      capture_content → stocke K_content, V_content
      inject          → CFG interpolation
      off             → forward normal
    """

    def __init__(self):
        self.mode: str = "off"
        self.current_t: int = 0
        self.alpha: float = 0.5
        self._sk: dict = {}  # style keys
        self._sv: dict = {}  # style values
        self._ck: dict = {}  # content keys
        self._cv: dict = {}  # content values

    def set_timestep(self, t):
        self.current_t = int(t)

    def _key(self, name):
        return (name, self.current_t)

    def store_style(self, name, k, v):
        key = self._key(name)
        self._sk[key] = k.detach().clone()
        self._sv[key] = v.detach().clone()

    def store_content(self, name, k, v):
        key = self._key(name)
        self._ck[key] = k.detach().clone()
        self._cv[key] = v.detach().clone()

    def get_style(self, name):
        key = self._key(name)
        return self._sk.get(key), self._sv.get(key)

    def get_content(self, name):
        key = self._key(name)
        return self._ck.get(key), self._cv.get(key)

    def clear(self):
        self._sk.clear(); self._sv.clear()
        self._ck.clear(); self._cv.clear()


# ─────────────────────────────────────────────────────────────────────────────
# StylusAttnProcessor — CFG-inspired interpolation
# ─────────────────────────────────────────────────────────────────────────────

class StylusAttnProcessor:
    """
    Implémente l'équation du papier :
      out_content = Attention(Q, K_content, V_content)
      out_style   = Attention(Q, K_style,   V_style)
      out         = out_content + α * (out_style - out_content)

    On ne modifie QUE la self-attention (encoder_hidden_states is None).
    """

    def __init__(self, store: AttentionStore, layer_name: str):
        self.store = store
        self.layer_name = layer_name

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        is_self = (encoder_hidden_states is None)
        kv_src  = hidden_states if is_self else encoder_hidden_states

        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(kv_src))
        v = attn.head_to_batch_dim(attn.to_v(kv_src))

        store = self.store

        if is_self:
            if store.mode == "capture_style":
                store.store_style(self.layer_name, k, v)

            elif store.mode == "capture_content":
                store.store_content(self.layer_name, k, v)

            elif store.mode == "inject":
                sk, sv = store.get_style(self.layer_name)
                ck, cv = store.get_content(self.layer_name)

                if sk is not None and ck is not None:
                    # CFG-inspired interpolation (équation papier)
                    out_content = self._attn(q, ck, cv, attn, attention_mask)
                    out_style   = self._attn(q, sk, sv, attn, attention_mask)
                    out = out_content + store.alpha * (out_style - out_content)
                    out = attn.to_out[0](out)
                    out = attn.to_out[1](out)
                    return out

        # Forward normal (capture ou off)
        out = self._attn(q, k, v, attn, attention_mask)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

    @staticmethod
    def _attn(q, k, v, attn_module, mask=None):
        """Calcul d'attention standard."""
        w = torch.bmm(q, k.transpose(-2, -1)) * attn_module.scale
        if mask is not None:
            w = w + mask
        w = w.softmax(dim=-1).to(q.dtype)
        out = torch.bmm(w, v)
        return attn_module.batch_to_head_dim(out)


# ─────────────────────────────────────────────────────────────────────────────
# Audio ↔ Mel ↔ Phase — avec conservation de phase (section 3.3 du papier)
# ─────────────────────────────────────────────────────────────────────────────

class AudioProcessor:
    """
    Gère toutes les conversions audio ↔ mel ↔ image.

    Phase-preserving reconstruction (papier section 3.3) :
      - On stocke la phase STFT complexe du content audio
      - Après stylisation, on reconstruit : magnitude_stylisée × exp(j * phase_content)
      - ISTFT → waveform propre sans Griffin-Lim
    """

    def __init__(self, cfg: StylusConfig):
        self.cfg = cfg
        try:
            import librosa
            self.librosa = librosa
        except ImportError:
            raise ImportError("pip install librosa")

    def audio_to_mel_and_phase(self, audio: np.ndarray):
        """
        Retourne (mel_db, stft_complex) où stft_complex contient la phase originale.
        """
        cfg = self.cfg
        # STFT complet (complexe) — on garde la phase
        stft = self.librosa.stft(
            audio, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            window='hann', center=True,
        )  # shape: (n_fft//2+1, T), complexe

        magnitude = np.abs(stft)

        # Mel filterbank
        mel_fb = self.librosa.filters.mel(
            sr=cfg.sample_rate, n_fft=cfg.n_fft,
            n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
        )  # (n_mels, n_fft//2+1)

        mel_power = mel_fb @ (magnitude ** 2)
        mel_db = self.librosa.power_to_db(mel_power, ref=np.max)

        return mel_db, stft  # stft contient amplitude ET phase

    def mel_to_image(self, mel_db: np.ndarray) -> torch.Tensor:
        """Mel dB → image RGB [-1, 1] shape (1, 3, H, W) pour SD."""
        cfg = self.cfg
        mel_min, mel_max = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

        t = torch.from_numpy(mel_norm).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(cfg.n_mels, cfg.target_length),
                          mode='bilinear', align_corners=False)
        t = t.repeat(1, 3, 1, 1)
        return t * 2.0 - 1.0  # → [-1, 1]

    def image_to_mel_norm(self, image: torch.Tensor) -> np.ndarray:
        """Image RGB [-1, 1] → mel normalisé [0, 1] (avant dé-normalisation).
        Moyenne des 3 canaux pour réduire le bruit VAE, clamp strict."""
        img = image[0].float().cpu().numpy()  # (3, H, W)
        img = img.mean(axis=0)                # moyenne RGB → (H, W)
        return ((img + 1.0) / 2.0).clip(0.0, 1.0)  # → [0, 1]

    def phase_preserving_reconstruct(
        self, mel_db_stylized: np.ndarray, content_stft: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruction phase-preserving (section 3.3 papier).

        Approche : spectral envelope transfer.
        On ne reconstruit PAS une magnitude STFT depuis le mel stylisé
        (toujours instable en hautes fréquences).

        À la place :
          1. Calculer l'enveloppe spectrale du mel stylisé (par bande temporelle)
          2. Calculer l'enveloppe spectrale du content STFT original
          3. Ratio = enveloppe_style / enveloppe_content → filtre spectral
          4. Appliquer le ratio sur la magnitude STFT du content
          5. Combiner avec la phase du content → ISTFT

        Le résultat : timbre/couleur du style, structure temporelle/phase du content.
        Aucune inversion de filterbank, aucun bruit artificiel.
        """
        cfg = self.cfg
        librosa = self.librosa

        # ── 1. Magnitude STFT du content ──────────────────────────────────────
        mag_content = np.abs(content_stft)  # (n_fft//2+1, T)
        T_stft = content_stft.shape[1]

        # ── 2. Mel filterbank ─────────────────────────────────────────────────
        mel_fb = librosa.filters.mel(
            sr=cfg.sample_rate, n_fft=cfg.n_fft,
            n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
        )  # (n_mels, n_fft//2+1)

        # ── 3. Enveloppe mel du content (depuis STFT) ─────────────────────────
        # Appliquer la filterbank mel sur la magnitude STFT du content
        mel_content_power = mel_fb @ (mag_content ** 2)          # (n_mels, T)
        mel_content_amp   = np.sqrt(np.maximum(mel_content_power, 1e-10))

        # ── 4. Amplitude mel stylisée ─────────────────────────────────────────
        mel_style_amp = librosa.db_to_amplitude(mel_db_stylized)  # (n_mels, T_mel)

        # Aligner temporellement
        T_mel = mel_style_amp.shape[1]
        if T_mel != T_stft:
            import torch, torch.nn.functional as F_torch
            t = torch.from_numpy(mel_style_amp).float().unsqueeze(0).unsqueeze(0)
            t = F_torch.interpolate(t, size=(cfg.n_mels, T_stft),
                                    mode="bilinear", align_corners=False)
            mel_style_amp = t.squeeze().numpy()

        # ── 5. Ratio spectral : style / content dans l'espace mel ─────────────
        # Éviter division par zéro avec un plancher
        ratio_mel = mel_style_amp / (mel_content_amp + 1e-10)  # (n_mels, T)

        # Lisser le ratio temporellement pour éviter les artéfacts de modulation
        from scipy.ndimage import uniform_filter1d
        ratio_mel = uniform_filter1d(ratio_mel, size=5, axis=1)

        # ── 6. Projeter le ratio mel → espace STFT ────────────────────────────
        # Utiliser la transposée normalisée de la filterbank mel
        # (chaque bin STFT reçoit la moyenne pondérée des ratios mel qui le couvrent)
        mel_fb_norm = mel_fb / (mel_fb.sum(axis=0, keepdims=True) + 1e-10)
        ratio_stft = mel_fb_norm.T @ ratio_mel  # (n_fft//2+1, T)

        # Écrêter le ratio pour éviter les amplifications excessives
        ratio_stft = np.clip(ratio_stft, 0.0, 10.0)

        # ── 7. Magnitude stylisée = magnitude content × ratio ─────────────────
        mag_stylized = mag_content * ratio_stft

        # ── 8. Phase du content + ISTFT ───────────────────────────────────────
        stft_stylized = mag_stylized * np.exp(1j * np.angle(content_stft))

        audio = librosa.istft(
            stft_stylized, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            window="hann", center=True,
        )
        return audio.astype(np.float32)

    # ── Visualisation ────────────────────────────────────────────────────────

    def save_mel_image(self, mel_db: np.ndarray, path: str, title: str = ""):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cfg = self.cfg
        dur = mel_db.shape[1] * cfg.hop_length / cfg.sample_rate
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(mel_db, origin='lower', aspect='auto', cmap='magma',
                       extent=[0, dur, cfg.fmin, cfg.fmax], interpolation='nearest')
        ax.set_xlabel("Temps (s)", fontsize=11)
        ax.set_ylabel("Fréquence (Hz)", fontsize=11)
        fig.colorbar(im, ax=ax, format="%+2.0f dB").set_label("dB", fontsize=10)
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    def save_comparison(self, mel_s, mel_c, mel_out, path: str):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cfg = self.cfg
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        mels = [mel_s, mel_c, mel_out]
        titles = ["Style", "Content", "Stylized output"]
        vmin = min(m.min() for m in mels)
        vmax = max(m.max() for m in mels)
        for ax, mel, title in zip(axes, mels, titles):
            dur = mel.shape[1] * cfg.hop_length / cfg.sample_rate
            im = ax.imshow(mel, origin='lower', aspect='auto', cmap='magma',
                           extent=[0, dur, cfg.fmin, cfg.fmax],
                           vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel("Temps (s)"); ax.set_ylabel("Fréquence (Hz)")
        fig.colorbar(im, ax=axes, format="%+2.0f dB", shrink=0.8, label="dB")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

class StylusPipeline:

    def __init__(self, cfg: Optional[StylusConfig] = None):
        self.cfg   = cfg or StylusConfig()
        self.proc  = AudioProcessor(self.cfg)
        self.store = AttentionStore()
        self.store.alpha = self.cfg.alpha
        self._pipe = None

    def load_model(self):
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        print(f"Loading {self.cfg.model_id} ...")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.model_id, torch_dtype=self.cfg.dtype,
            safety_checker=None, requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.cfg.device)
        pipe.unet.eval()
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        self._install_processors()
        print("Model ready.")
        return self

    def _install_processors(self):
        installed = 0
        for name, module in self._pipe.unet.named_modules():
            if not any(f"up_blocks.{i}." in name for i in self.cfg.target_up_block_indices):
                continue
            if not name.endswith("attn1"):
                continue
            module.set_processor(StylusAttnProcessor(self.store, name))
            installed += 1
        print(f"Installed StylusAttnProcessor on {installed} layers.")
        if installed == 0:
            raise RuntimeError("Aucune couche trouvée.")

    @torch.no_grad()
    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.cfg.device, dtype=self.cfg.dtype)
        z = self._pipe.vae.encode(image).latent_dist.mean
        return z * self._pipe.vae.config.scaling_factor

    @torch.no_grad()
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self._pipe.vae.config.scaling_factor
        return self._pipe.vae.decode(z).sample.float().cpu()

    @torch.no_grad()
    def _null_emb(self):
        return self._pipe.encode_prompt(
            "", self.cfg.device, 1, False,
        )[0]

    @torch.no_grad()
    def _ddim_inversion(self, z0, emb, label=""):
        sched = self._pipe.scheduler
        sched.set_timesteps(self.cfg.num_inference_steps)
        timesteps = sched.timesteps.flip(0)
        alphas = sched.alphas_cumprod.to(z0.device)
        zt = z0.clone()
        for i, t in enumerate(timesteps):
            self.store.set_timestep(int(t))
            eps = self._pipe.unet(zt, t, encoder_hidden_states=emb).sample
            stride = sched.config.num_train_timesteps // self.cfg.num_inference_steps
            t_next = min(int(t) + stride, sched.config.num_train_timesteps - 1)
            a_t = alphas[int(t)]; a_n = alphas[t_next]
            x0 = (zt - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp(min=1e-8)
            zt = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps
            if (i + 1) % 10 == 0:
                print(f"  [{label}] inversion {i+1}/{len(timesteps)}")
        return zt

    @torch.no_grad()
    def _ddim_reverse(self, zT, emb, label=""):
        sched = self._pipe.scheduler
        sched.set_timesteps(self.cfg.num_inference_steps)
        zt = zT.clone()
        for i, t in enumerate(sched.timesteps):
            self.store.set_timestep(int(t))
            eps = self._pipe.unet(zt, t, encoder_hidden_states=emb).sample
            zt = sched.step(eps, t, zt).prev_sample
            if (i + 1) % 10 == 0:
                print(f"  [{label}] denoising {i+1}/{len(sched.timesteps)}")
        return zt

    def transfer(
        self,
        style_audio: np.ndarray,
        content_audio: np.ndarray,
        save_dir: Optional[str] = None,
    ) -> np.ndarray:
        if self._pipe is None:
            self.load_model()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        emb = self._null_emb()

        # ── 1. Audio → mel + phase ────────────────────────────────────────────
        print("\n[1/5] Audio → mel + phase...")
        mel_s, stft_s = self.proc.audio_to_mel_and_phase(style_audio)
        mel_c, stft_c = self.proc.audio_to_mel_and_phase(content_audio)

        img_s = self.proc.mel_to_image(mel_s).to(self.cfg.device, dtype=self.cfg.dtype)
        img_c = self.proc.mel_to_image(mel_c).to(self.cfg.device, dtype=self.cfg.dtype)

        if save_dir:
            self.proc.save_mel_image(mel_s, os.path.join(save_dir, "mel_style.png"),   "Style")
            self.proc.save_mel_image(mel_c, os.path.join(save_dir, "mel_content.png"), "Content")

        # ── 2. VAE encode ─────────────────────────────────────────────────────
        print("[2/5] VAE encode...")
        z_s = self._encode(img_s)
        z_c = self._encode(img_c)

        # ── 3. DDIM inversion style → capture K_style, V_style ────────────────
        print("[3/5] DDIM inversion — style (capture K, V)...")
        self.store.mode = "capture_style"
        self._ddim_inversion(z_s, emb, label="style")

        # ── 4. DDIM inversion content → capture K_content, V_content ──────────
        print("[4/5] DDIM inversion — content (capture K, V)...")
        self.store.mode = "capture_content"
        zT_c = self._ddim_inversion(z_c, emb, label="content")

        # ── 5. DDIM reverse avec CFG-inspired interpolation ───────────────────
        print("[5/5] DDIM reverse — CFG-inspired style injection (α={})...".format(
            self.cfg.alpha))
        self.store.mode = "inject"
        z0_out = self._ddim_reverse(zT_c, emb, label="stylized")
        self.store.mode = "off"

        # ── Décodage VAE → mel ────────────────────────────────────────────────
        print("\nDecoding (VAE)...")
        img_out = self._decode(z0_out)

        # image_to_mel_norm retourne [0, 1] (moyenne des 3 canaux RGB)
        mel_norm = self.proc.image_to_mel_norm(img_out)  # (H, W) ∈ [0, 1]

        # Resize vers les dimensions du mel content original
        import torch as _torch, torch.nn.functional as _F
        t = _torch.from_numpy(mel_norm).float().unsqueeze(0).unsqueeze(0)
        t = _F.interpolate(t, size=(mel_c.shape[0], mel_c.shape[1]),
                           mode='bilinear', align_corners=False)
        mel_norm_resized = t.squeeze().numpy()  # (n_mels, T_content)

        # Re-normaliser avec les stats du content pour préserver la dynamique.
        # Le VAE SD1.5 n'a pas été entraîné sur des mels — sa sortie normalisée
        # est relative. On la mappe dans la même plage dB que le content.
        c_min, c_max = mel_c.min(), mel_c.max()
        mel_out_resized = mel_norm_resized * (c_max - c_min) + c_min

        # ── Phase-preserving reconstruction ───────────────────────────────────
        print("Phase-preserving reconstruction (ISTFT + phase content)...")
        audio_out = self.proc.phase_preserving_reconstruct(mel_out_resized, stft_c)

        if save_dir:
            self.proc.save_mel_image(mel_out_resized,
                                     os.path.join(save_dir, "mel_stylized.png"),
                                     "Stylized output")
            self.proc.save_comparison(mel_s, mel_c, mel_out_resized,
                                      os.path.join(save_dir, "mel_comparison.png"))

        print("Done!")
        return audio_out