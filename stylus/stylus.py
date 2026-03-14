"""
Stylus: Repurposing Stable Diffusion for Training-Free Music Style Transfer
Implémentation fidèle du papier arxiv:2411.15913

Pipeline exact du papier :
  1. DDIM inversion style   → capture K_style[t],   V_style[t]   à chaque t
  2. DDIM inversion content → capture Q_content[t]              à chaque t
  3. AdaIN(z_T_content, z_T_style) → z_T_init
  4. DDIM reverse depuis z_T_init :
       Q_bar = γ*Q_content[t] + (1-γ)*Q_current[t]          ← query preservation
       out   = out_content + α*(out_style - out_content)      ← style guidance scale
  5. Phase-preserving reconstruction : phase STFT content + magnitude stylisée
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
    # γ : query preservation — contrôle comment la query "regarde"
    #     0 = Q du reverse courant (libre), 1 = Q du content inversé (structure pure)
    gamma: float = 0.8

    # α : style guidance scale — contrôle combien la sortie attention penche vers le style
    #     interpolation CFG sur la sortie : out = out_content + α*(out_style - out_content)
    #     0 = content pur, 1 = style pur
    alpha: float = 0.5

    # Couches U-Net SD1.5 ciblées — layers 7-12 du papier = up_blocks[1,2,3]
    # up_blocks[1] = 8×8, up_blocks[2] = 16×16, up_blocks[3] = 32×32
    target_up_block_indices: list = field(default_factory=lambda: [1, 2, 3])

    # DDIM
    num_inference_steps: int = 50

    # STFT
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 512
    fmin: float = 0.0
    fmax: float = 8000.0
    target_length: int = 512

    # Modèle
    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# AttentionStore — stocke Q_content, K_style, V_style par timestep
# ─────────────────────────────────────────────────────────────────────────────


class AttentionStore:
    """
    Modes :
      capture_style   → stocke K[t], V[t] du style
      capture_content → stocke Q[t] du content
      inject          → query preservation + injection K/V style
      off             → forward normal
    """

    def __init__(self):
        self.mode: str = "off"
        self.current_t: int = 0
        self.gamma: float = 0.8
        self.alpha: float = 0.5
        self._qs: dict = {}  # Q content  : (layer, t) → tensor
        self._ks: dict = {}  # K style    : (layer, t) → tensor
        self._vs: dict = {}  # V style    : (layer, t) → tensor

    def set_timestep(self, t):
        self.current_t = int(t)

    def _key(self, name):
        return (name, self.current_t)

    def store_style_kv(self, name, k, v):
        key = self._key(name)
        self._ks[key] = k.detach().clone()
        self._vs[key] = v.detach().clone()

    def store_content_q(self, name, q):
        key = self._key(name)
        self._qs[key] = q.detach().clone()

    def get_style_kv(self, name):
        key = self._key(name)
        return self._ks.get(key), self._vs.get(key)

    def get_content_q(self, name):
        key = self._key(name)
        return self._qs.get(key)

    def clear(self):
        self._qs.clear()
        self._ks.clear()
        self._vs.clear()


# ─────────────────────────────────────────────────────────────────────────────
# StylusAttnProcessor — query preservation + injection K/V
# ─────────────────────────────────────────────────────────────────────────────


class StylusAttnProcessor:
    """
    Implémente l'équation exacte du papier :

      Capture style   : stocke K_style[t], V_style[t]
      Capture content : stocke Q_content[t]
      Inject          :
        Q_bar = γ * Q_content[t] + (1-γ) * Q_current[t]
        out   = Attn(Q_bar, K_style[t], V_style[t])

    Seule la self-attention est modifiée (encoder_hidden_states is None).
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

        is_self = encoder_hidden_states is None
        kv_src = hidden_states if is_self else encoder_hidden_states

        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(kv_src))
        v = attn.head_to_batch_dim(attn.to_v(kv_src))

        store = self.store

        if is_self:
            if store.mode == "capture_style":
                store.store_style_kv(self.layer_name, k, v)

            elif store.mode == "capture_content":
                store.store_content_q(self.layer_name, q)

            elif store.mode == "inject":
                ks, vs = store.get_style_kv(self.layer_name)
                qc = store.get_content_q(self.layer_name)

                if ks is not None and qc is not None:
                    # Query preservation : Q_bar = γ*Q_content + (1-γ)*Q_current
                    q_bar = store.gamma * qc + (1 - store.gamma) * q

                    # Style guidance scale α (CFG-inspired sur les sorties) :
                    #   out_content = Attn(Q_bar, K_content, V_content)  ← structure
                    #   out_style   = Attn(Q_bar, K_style,  V_style)     ← texture
                    #   out = out_content + α * (out_style - out_content)
                    out_content = self._attn(q_bar, k, v, attn, attention_mask)
                    out_style = self._attn(q_bar, ks, vs, attn, attention_mask)
                    out = out_content + store.alpha * (out_style - out_content)
                    out = attn.to_out[0](out)
                    out = attn.to_out[1](out)
                    return out

        # Forward normal
        out = self._attn(q, k, v, attn, attention_mask)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

    @staticmethod
    def _attn(q, k, v, attn_module, mask=None):
        w = torch.bmm(q, k.transpose(-2, -1)) * attn_module.scale
        if mask is not None:
            w = w + mask
        w = w.softmax(dim=-1).to(q.dtype)
        out = torch.bmm(w, v)
        return attn_module.batch_to_head_dim(out)


# ─────────────────────────────────────────────────────────────────────────────
# AdaIN sur les latents
# ─────────────────────────────────────────────────────────────────────────────


def adain_latent(z_content: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
    """
    AdaIN(z_content, z_style) = σ(z_style) * (z_content - μ(z_content)) / σ(z_content) + μ(z_style)
    Calcul par canal sur les dimensions spatiales (H, W) du latent.
    """
    eps = 1e-5
    # Moments sur H×W
    mu_c = z_content.mean(dim=[2, 3], keepdim=True)
    sig_c = z_content.std(dim=[2, 3], keepdim=True) + eps
    mu_s = z_style.mean(dim=[2, 3], keepdim=True)
    sig_s = z_style.std(dim=[2, 3], keepdim=True) + eps
    return sig_s * (z_content - mu_c) / sig_c + mu_s


# ─────────────────────────────────────────────────────────────────────────────
# AudioProcessor
# ─────────────────────────────────────────────────────────────────────────────


class AudioProcessor:
    def __init__(self, cfg: StylusConfig):
        self.cfg = cfg
        try:
            import librosa

            self.librosa = librosa
        except ImportError:
            raise ImportError("pip install librosa")

    def audio_to_mel_and_phase(self, audio: np.ndarray):
        cfg = self.cfg
        stft = self.librosa.stft(
            audio,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window="hann",
            center=True,
        )
        magnitude = np.abs(stft)
        mel_fb = self.librosa.filters.mel(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        )
        mel_db = self.librosa.power_to_db(mel_fb @ (magnitude**2), ref=np.max)
        return mel_db, stft

    def mel_to_image(self, mel_db: np.ndarray) -> torch.Tensor:
        cfg = self.cfg
        mel_min, mel_max = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
        t = torch.from_numpy(mel_norm).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(
            t,
            size=(cfg.n_mels, cfg.target_length),
            mode="bilinear",
            align_corners=False,
        )
        t = t.repeat(1, 3, 1, 1)
        return t * 2.0 - 1.0

    def image_to_mel_norm(self, image: torch.Tensor) -> np.ndarray:
        """Image → mel normalisé [0,1], moyenne des 3 canaux."""
        img = image[0].float().cpu().numpy().mean(axis=0)
        return ((img + 1.0) / 2.0).clip(0.0, 1.0)

    # def phase_preserving_reconstruct(
    #     self, mel_db_stylized: np.ndarray, content_stft: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Spectral envelope transfer + phase du content.
    #     On n'inverse pas la filterbank (instable) : on calcule le ratio
    #     d'enveloppe spectrale style/content et on l'applique sur la
    #     magnitude STFT du content.
    #     """
    #     cfg = self.cfg
    #     librosa = self.librosa

    #     mag_content = np.abs(content_stft)
    #     T_stft = content_stft.shape[1]

    #     mel_fb = librosa.filters.mel(
    #         sr=cfg.sample_rate,
    #         n_fft=cfg.n_fft,
    #         n_mels=cfg.n_mels,
    #         fmin=cfg.fmin,
    #         fmax=cfg.fmax,
    #     )

    #     # Enveloppe mel du content
    #     mel_content_amp = np.sqrt(np.maximum(mel_fb @ (mag_content**2), 1e-10))

    #     # Amplitude mel stylisée → aligner temporellement
    #     mel_style_amp = librosa.db_to_amplitude(mel_db_stylized)
    #     T_mel = mel_style_amp.shape[1]
    #     if T_mel != T_stft:
    #         import torch as _t, torch.nn.functional as _F

    #         tt = _t.from_numpy(mel_style_amp).float().unsqueeze(0).unsqueeze(0)
    #         tt = _F.interpolate(
    #             tt, size=(cfg.n_mels, T_stft), mode="bilinear", align_corners=False
    #         )
    #         mel_style_amp = tt.squeeze().numpy()

    #     # Ratio spectral
    #     ratio_mel = mel_style_amp / (mel_content_amp + 1e-10)

    #     from scipy.ndimage import uniform_filter1d

    #     ratio_mel = uniform_filter1d(ratio_mel, size=5, axis=1)

    #     # Projeter ratio mel → STFT
    #     mel_fb_norm = mel_fb / (mel_fb.sum(axis=0, keepdims=True) + 1e-10)
    #     ratio_stft = np.clip(mel_fb_norm.T @ ratio_mel, 0.0, 10.0)

    #     # Magnitude stylisée × phase content
    #     stft_stylized = (mag_content * ratio_stft) * np.exp(1j * np.angle(content_stft))

    #     audio = librosa.istft(
    #         stft_stylized,
    #         n_fft=cfg.n_fft,
    #         hop_length=cfg.hop_length,
    #         window="hann",
    #         center=True,
    #     )
    #     return audio.astype(np.float32)

    def phase_preserving_reconstruct(
        self, mel_db_stylized: np.ndarray, content_stft: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruction directe depuis le mel stylisé.
 
        Ancienne approche (spectral envelope transfer) :
          mag_stylized = mag_content * ratio  ← ancré sur le content
          → le style ne peut jamais dominer, même avec alpha=1 / gamma=0
 
        Approche correcte :
          1. Mel stylisé → magnitude STFT directement (transposée pondérée)
          2. Phase du content comme initialisation GL (ancre le timing)
          3. 8 itérations GL pour la cohérence STFT
 
        La magnitude vient entièrement du mel stylisé.
        La phase du content ancre le rythme sans imposer sa magnitude.
        """
        cfg = self.cfg
        librosa = self.librosa
 
        T_stft = content_stft.shape[1]
 
        # 1. Mel dB → amplitude mel
        mel_amp = librosa.db_to_amplitude(mel_db_stylized)
 
        # Aligner temporellement
        T_mel = mel_amp.shape[1]
        if T_mel != T_stft:
            import torch as _t, torch.nn.functional as _F
            tt = _t.from_numpy(mel_amp).float().unsqueeze(0).unsqueeze(0)
            tt = _F.interpolate(tt, size=(cfg.n_mels, T_stft),
                                mode="bilinear", align_corners=False)
            mel_amp = tt.squeeze().numpy()
 
        # 2. Amplitude mel → magnitude STFT via transposée pondérée
        mel_fb = librosa.filters.mel(
            sr=cfg.sample_rate, n_fft=cfg.n_fft,
            n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
        )
        mel_fb_norm = mel_fb / (mel_fb.sum(axis=0, keepdims=True) + 1e-10)
        mag_stft = np.maximum(mel_fb_norm.T @ mel_amp, 0.0)
 
        # Supprimer artefacts HF au-delà de fmax
        freqs = np.linspace(0, cfg.sample_rate / 2, mag_stft.shape[0])
        rolloff = cfg.fmax if cfg.fmax > 0 else cfg.sample_rate / 2
        mask = np.where(freqs <= rolloff, 1.0,
                        np.maximum(0.0, 1.0 - (freqs - rolloff) / 500.0))
        mag_stft = mag_stft * mask[:, np.newaxis]
 
        # 3. Griffin-Lim initialisé avec la phase du content
        phase_init = np.angle(content_stft)
        stft_cur = mag_stft * np.exp(1j * phase_init)
 
        target_len = librosa.istft(
            content_stft, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            window='hann', center=True,
        ).shape[0]
 
        for _ in range(8):
            audio_tmp = librosa.istft(
                stft_cur, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                window='hann', center=True, length=target_len,
            )
            stft_cur = librosa.stft(
                audio_tmp, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                window='hann', center=True,
            )
            stft_cur = mag_stft * np.exp(1j * np.angle(stft_cur))
 
        audio = librosa.istft(
            stft_cur, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            window='hann', center=True, length=target_len,
        )
        return audio.astype(np.float32)

    def save_mel_image(self, mel_db, path, title=""):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cfg = self.cfg
        dur = mel_db.shape[1] * cfg.hop_length / cfg.sample_rate
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            mel_db,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[0, dur, cfg.fmin, cfg.fmax],
        )
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Fréquence (Hz)")
        fig.colorbar(im, ax=ax, format="%+2.0f dB").set_label("dB")
        if title:
            ax.set_title(title, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    def save_comparison(self, mel_s, mel_c, mel_out, path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        mels = [mel_s, mel_c, mel_out]
        titles = ["Style", "Content", "Stylized output"]
        vmin = min(m.min() for m in mels)
        vmax = max(m.max() for m in mels)
        cfg = self.cfg
        for ax, mel, title in zip(axes, mels, titles):
            dur = mel.shape[1] * cfg.hop_length / cfg.sample_rate
            im = ax.imshow(
                mel,
                origin="lower",
                aspect="auto",
                cmap="magma",
                extent=[0, dur, cfg.fmin, cfg.fmax],
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("Fréquence (Hz)")
        fig.colorbar(im, ax=axes, format="%+2.0f dB", shrink=0.8, label="dB")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class StylusPipeline:
    def __init__(self, cfg: Optional[StylusConfig] = None):
        self.cfg = cfg or StylusConfig()
        self.proc = AudioProcessor(self.cfg)
        self.store = AttentionStore()
        self.store.gamma = self.cfg.gamma
        self.store.alpha = self.cfg.alpha
        self._pipe = None

    def load_model(self):
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        print(f"Loading {self.cfg.model_id} ...")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.model_id,
            torch_dtype=self.cfg.dtype,
            safety_checker=None,
            requires_safety_checker=False,
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
            if not any(
                f"up_blocks.{i}." in name for i in self.cfg.target_up_block_indices
            ):
                continue
            if not name.endswith("attn1"):
                continue
            module.set_processor(StylusAttnProcessor(self.store, name))
            installed += 1
        print(f"StylusAttnProcessor installé sur {installed} couches.")
        if installed == 0:
            raise RuntimeError(
                "Aucune couche trouvée — vérifier target_up_block_indices."
            )

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
        return self._pipe.encode_prompt("", self.cfg.device, 1, False)[0]

    @torch.no_grad()
    def _ddim_inversion(self, z0, emb, label=""):
        """
        Inversion DDIM t=0→T.
        Capture K_style,V_style (mode capture_style)
             ou Q_content       (mode capture_content)
        à chaque timestep.
        Retourne zT.
        """
        sched = self._pipe.scheduler
        sched.set_timesteps(self.cfg.num_inference_steps)
        timesteps = sched.timesteps.flip(0)  # 0 → T
        alphas = sched.alphas_cumprod.to(z0.device)
        zt = z0.clone()
        stride = sched.config.num_train_timesteps // self.cfg.num_inference_steps
        for i, t in enumerate(timesteps):
            self.store.set_timestep(int(t))
            eps = self._pipe.unet(zt, t, encoder_hidden_states=emb).sample
            t_next = min(int(t) + stride, sched.config.num_train_timesteps - 1)
            a_t = alphas[int(t)]
            a_n = alphas[t_next]
            x0 = (zt - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp(min=1e-8)
            zt = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps
            if (i + 1) % 10 == 0:
                print(f"  [{label}] inversion {i + 1}/{len(timesteps)}")
        return zt

    @torch.no_grad()
    def _ddim_reverse(self, zT, emb, label=""):
        """
        Reverse DDIM t=T→0 en mode inject.
        Utilise Q_content[t] + K_style[t] + V_style[t] à chaque step.
        """
        sched = self._pipe.scheduler
        sched.set_timesteps(self.cfg.num_inference_steps)
        zt = zT.clone()
        for i, t in enumerate(sched.timesteps):
            self.store.set_timestep(int(t))
            eps = self._pipe.unet(zt, t, encoder_hidden_states=emb).sample
            zt = sched.step(eps, t, zt).prev_sample
            if (i + 1) % 10 == 0:
                print(f"  [{label}] denoising {i + 1}/{len(sched.timesteps)}")
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
            self.proc.save_mel_image(
                mel_s, os.path.join(save_dir, "mel_style.png"), "Style"
            )
            self.proc.save_mel_image(
                mel_c, os.path.join(save_dir, "mel_content.png"), "Content"
            )

        # ── 2. VAE encode ─────────────────────────────────────────────────────
        print("[2/5] VAE encode...")
        z0_s = self._encode(img_s)
        z0_c = self._encode(img_c)

        # ── 3. DDIM inversion style → capture K_style[t], V_style[t] ──────────
        print("[3/5] DDIM inversion style (capture K, V)...")
        self.store.mode = "capture_style"
        zT_s = self._ddim_inversion(z0_s, emb, label="style")

        # ── 4. DDIM inversion content → capture Q_content[t] ──────────────────
        print("[4/5] DDIM inversion content (capture Q)...")
        self.store.mode = "capture_content"
        zT_c = self._ddim_inversion(z0_c, emb, label="content")

        # ── 5. AdaIN(zT_content, zT_style) → initialisation du latent ─────────
        print("[5/5] AdaIN + DDIM reverse (inject)...")
        zT_init = adain_latent(zT_c, zT_s)

        # ── 6. DDIM reverse avec query preservation + injection K/V ───────────
        self.store.mode = "inject"
        z0_out = self._ddim_reverse(zT_init, emb, label="stylized")
        self.store.mode = "off"

        # ── Décodage VAE → mel ────────────────────────────────────────────────
        print("\nDecoding (VAE)...")
        img_out = self._decode(z0_out)
        mel_norm = self.proc.image_to_mel_norm(img_out)

        # Resize + re-normaliser avec les stats du content
        import torch as _t, torch.nn.functional as _F

        tt = _t.from_numpy(mel_norm).float().unsqueeze(0).unsqueeze(0)
        tt = _F.interpolate(
            tt,
            size=(mel_c.shape[0], mel_c.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        mel_norm_r = tt.squeeze().numpy()
        c_min, c_max = mel_c.min(), mel_c.max()
        mel_out = mel_norm_r * (c_max - c_min) + c_min

        # ── Phase-preserving reconstruction ───────────────────────────────────
        print("Phase-preserving reconstruction...")
        audio_out = self.proc.phase_preserving_reconstruct(mel_out, stft_c)

        if save_dir:
            self.proc.save_mel_image(
                mel_out, os.path.join(save_dir, "mel_stylized.png"), "Stylized output"
            )
            self.proc.save_comparison(
                mel_s, mel_c, mel_out, os.path.join(save_dir, "mel_comparison.png")
            )

        print("Done!")
        return audio_out
