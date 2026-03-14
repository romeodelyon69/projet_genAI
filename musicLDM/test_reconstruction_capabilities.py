"""
Stylus-AudioLDM2 : Music Style Transfer via AudioLDM2
========================================================
Adaptation fidèle de "Stylus: Repurposing Stable Diffusion for Training-Free
Music Style Transfer" (arxiv:2411.15913) pour AudioLDM2.

Remplace Stable Diffusion (image) par AudioLDM2 (audio natif) :
  - VAE AudioLDM2 encode directement les mel-spectrogrammes
  - UNet AudioLDM2 (AudioLDM2UNet2DConditionModel) conditionné CLAP+T5+GPT2
  - Self-attention hookée (attn1) — identique à Stylus SD
  - Pas de conversion mel→image→latent : pipeline 100% audio natif
  - Reconstruction via vocoder HiFi-GAN interne

Pipeline exact (même logique que Stylus) :
  1. DDIM inversion style   → capture K_style[t],   V_style[t]   à chaque t
  2. DDIM inversion content → capture Q_content[t]              à chaque t
  3. AdaIN(zT_content, zT_style) → zT_init
  4. DDIM reverse depuis zT_init :
       Q_bar = γ*Q_content[t] + (1-γ)*Q_current[t]   (query preservation)
       out   = out_content + α*(out_style - out_content) (style guidance)
  5. VAE decode → mel → HiFi-GAN → audio

Usage :
    python stylus_audioldm2.py

Requirements :
    pip install diffusers transformers accelerate librosa soundfile torch matplotlib
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from typing import Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StylusAudioLDM2Config:
    # Stylus params
    gamma: float = 0.5   # query preservation : 0=libre, 1=structure pure  (0.5 = plus de liberté)
    alpha: float = 0.8   # style guidance     : 0=content pur, 1=style pur   (0.8 = style dominant)

    # Couches AudioLDM2 UNet ciblées (self-attention uniquement)
    # AudioLDM2 : up_blocks[1,2,3] — même convention que Stylus SD
    target_up_block_indices: list = field(default_factory=lambda: [0, 1, 2, 3])  # tous les up_blocks

    # DDIM
    num_inference_steps: int = 50

    # Audio / Mel — paramètres AudioLDM2 natifs
    sample_rate: int     = 16_000
    clap_sr: int         = 48_000   # CLAP attend 48kHz
    n_fft: int           = 1024
    hop_length: int      = 160
    n_mels: int          = 64       # AudioLDM2 : 64 mel bins
    target_length: int   = -1       # -1 = calculé automatiquement depuis la durée audio
    duration: float      = -1.0     # -1 = charge tout le fichier

    # Modèle
    model_id: str = "cvssp/audioldm2-music"
    device: str   = "cuda" if torch.cuda.is_available() else "cpu"

    # Fichiers audio
    style_audio_path:   str = "musicTI_dataset/audios/timbre/chime/chime1.wav"
    content_audio_path: str = "musicTI_dataset/audios/content/violin/violin1.wav"
    output_dir:         str = "stylus_audioldm2_output"


# ─────────────────────────────────────────────────────────────────────────────
# AttentionStore — identique à Stylus, stocke Q/K/V par (layer, timestep)
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
        self.mode: str    = "off"
        self.current_t: int = 0
        self.gamma: float = 0.8
        self.alpha: float = 0.5
        self._qs: dict    = {}
        self._ks: dict    = {}
        self._vs: dict    = {}

    def set_timestep(self, t):
        self.current_t = int(t)

    def _key(self, name):
        return (name, self.current_t)

    def store_style_kv(self, name, k, v):
        key = self._key(name)
        self._ks[key] = k.detach().clone()
        self._vs[key] = v.detach().clone()

    def store_content_q(self, name, q):
        self._qs[self._key(name)] = q.detach().clone()

    def get_style_kv(self, name):
        key = self._key(name)
        return self._ks.get(key), self._vs.get(key)

    def get_content_q(self, name):
        return self._qs.get(self._key(name))

    def clear(self):
        self._qs.clear()
        self._ks.clear()
        self._vs.clear()


# ─────────────────────────────────────────────────────────────────────────────
# StylusAttnProcessor — query preservation + injection K/V
# Identique à Stylus mais adapté à l'API attn d'AudioLDM2
# ─────────────────────────────────────────────────────────────────────────────

class StylusAttnProcessor:
    """
    Hookée sur les self-attentions (attn1) du UNet AudioLDM2.

    Capture style   : stocke K_style[t], V_style[t]
    Capture content : stocke Q_content[t]
    Inject          :
        Q_bar = γ * Q_content[t] + (1-γ) * Q_current[t]
        out_content = Attn(Q_bar, K_content, V_content)
        out_style   = Attn(Q_bar, K_style[t], V_style[t])
        out = out_content + α * (out_style - out_content)
    """
    def __init__(self, store: AttentionStore, layer_name: str):
        self.store      = store
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
        kv_src  = hidden_states if is_self else encoder_hidden_states

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
                qc     = store.get_content_q(self.layer_name)

                if ks is not None and qc is not None:
                    # Query preservation
                    q_bar = store.gamma * qc + (1 - store.gamma) * q

                    # Style guidance scale (CFG-like sur les sorties attention)
                    out_content = self._attn(q_bar, k,  v,  attn, attention_mask)
                    out_style   = self._attn(q_bar, ks, vs, attn, attention_mask)
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
# AdaIN sur les latents (identique à Stylus)
# ─────────────────────────────────────────────────────────────────────────────

def adain_latent(z_content: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
    """AdaIN spatial sur les dimensions (H, W) du latent."""
    eps = 1e-5
    mu_c  = z_content.mean(dim=[2, 3], keepdim=True)
    sig_c = z_content.std(dim=[2, 3],  keepdim=True) + eps
    mu_s  = z_style.mean(dim=[2, 3],   keepdim=True)
    sig_s = z_style.std(dim=[2, 3],    keepdim=True) + eps
    return sig_s * (z_content - mu_c) / sig_c + mu_s


# ─────────────────────────────────────────────────────────────────────────────
# AudioProcessor — mel ↔ audio (paramètres AudioLDM2 natifs)
# ─────────────────────────────────────────────────────────────────────────────

class AudioProcessor:
    def __init__(self, cfg: StylusAudioLDM2Config):
        self.cfg = cfg

    def load_audio(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        # duration=-1 → charge tout le fichier
        dur = None if self.cfg.duration <= 0 else self.cfg.duration
        wav, _ = librosa.load(path, sr=self.cfg.sample_rate, mono=True, duration=dur)
        wav = wav / (np.abs(wav).max() + 1e-8)
        return wav.astype(np.float32)

    def audio_to_mel(self, wav: np.ndarray) -> np.ndarray:
        """
        Waveform → log-mel (n_mels, T) avec les paramètres exacts AudioLDM2.
        """
        cfg = self.cfg
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=0.0,
            fmax=8000.0,
            power=1.0,
            norm="slaney",
        )
        return np.log(mel + 1e-5)  # (n_mels, T)

    def audio_to_stft(self, wav: np.ndarray) -> np.ndarray:
        """Waveform → STFT complexe (F, T) — pour la phase du content."""
        cfg = self.cfg
        return librosa.stft(
            wav,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.n_fft,
            window="hann",
            center=True,
        )

    def mel_to_vae_input(self, mel: np.ndarray, target_length: int = -1) -> torch.Tensor:
        """
        Mel (n_mels, T) → tenseur VAE (1, 1, T, n_mels).
        AudioLDM2 VAE attend [B, 1, T, F] (temps en dim 2, fréquences en dim 3).
        target_length=-1 → utilise la longueur réelle du mel (pad à multiple de 8 pour le VAE).
        """
        T = mel.shape[1]

        if target_length > 0:
            # Longueur fixe imposée
            if T < target_length:
                mel = np.pad(mel, ((0, 0), (0, target_length - T)))
            else:
                mel = mel[:, :target_length]
        else:
            # Longueur automatique : pad au multiple de 8 le plus proche (requis par le VAE)
            pad_to = ((T + 7) // 8) * 8
            if T < pad_to:
                mel = np.pad(mel, ((0, 0), (0, pad_to - T)))

        mel_t = torch.FloatTensor(mel)  # (n_mels, T_pad)
        # Normalisation locale — meilleure reconstruction VAE observée empiriquement
        mel_t = (mel_t - mel_t.mean()) / (mel_t.std() + 1e-8)
        # Transpose : (T, n_mels) puis → (1, 1, T, n_mels)
        return mel_t.T.unsqueeze(0).unsqueeze(0)  # (1, 1, T_pad, n_mels)

    def save_audio(self, wav: np.ndarray, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        wav = wav / (np.abs(wav).max() + 1e-8) * 0.9
        sf.write(path, wav.astype(np.float32), self.cfg.sample_rate)
        print(f"  → {path}")

    def save_mel_plot(self, mel: np.ndarray, path: str, title: str = ""):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cfg = self.cfg
        dur = mel.shape[1] * cfg.hop_length / cfg.sample_rate
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
                       extent=[0, dur, 0, 8000])
        ax.set_xlabel("Temps (s)"); ax.set_ylabel("Fréquence (Hz)")
        fig.colorbar(im, ax=ax).set_label("log amplitude")
        if title: ax.set_title(title, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

class StylusAudioLDM2Pipeline:
    def __init__(self, cfg: Optional[StylusAudioLDM2Config] = None):
        self.cfg   = cfg or StylusAudioLDM2Config()
        self.proc  = AudioProcessor(self.cfg)
        self.store = AttentionStore()
        self.store.gamma = self.cfg.gamma
        self.store.alpha = self.cfg.alpha
        self._pipe = None

    # ── Chargement modèle ────────────────────────────────────────────────────

    def load_model(self):
        from diffusers import AudioLDM2Pipeline, DDIMScheduler
        print(f"[INFO] Chargement {self.cfg.model_id} ...")
        pipe = AudioLDM2Pipeline.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch.float32,  # float32 pour stabilité DDIM inversion
        ).to(self.cfg.device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        self._install_processors()
        print("  Modèle prêt ✓")
        return self

    def _install_processors(self):
        """
        Installe StylusAttnProcessor sur les self-attentions du UNet AudioLDM2.

        AudioLDM2 nomme ses attentions différemment selon la couche :
          - BasicTransformerBlock.attn1 = self-attention
          - BasicTransformerBlock.attn2 = cross-attention (vers CLAP/T5)
        On cible attn1 dans tous les up_blocks (down + mid inclus pour plus d'effet).
        """
        # D'abord lister toutes les couches d'attention disponibles
        all_attn = [(name, module) for name, module in self._pipe.unet.named_modules()
                    if hasattr(module, 'set_processor')]
        print(f"  Couches avec set_processor : {len(all_attn)}")

        # Identifier les self-attentions (attn1) vs cross-attentions (attn2)
        self_attns  = [(n, m) for n, m in all_attn if n.endswith("attn1")]
        cross_attns = [(n, m) for n, m in all_attn if n.endswith("attn2")]
        print(f"  Self-attentions (attn1)  : {len(self_attns)}")
        print(f"  Cross-attentions (attn2) : {len(cross_attns)}")

        if not self_attns:
            # Fallback : nommage alternatif dans AudioLDM2
            # Certaines versions utilisent "processor" directement sur Attention
            print("  [WARN] Pas de 'attn1' — listing complet :")
            for n, _ in all_attn[:20]:
                print(f"    {n}")
            self_attns = all_attn  # hook tout

        # Filtrer sur les up_blocks ciblés
        target_attns = [
            (n, m) for n, m in self_attns
            if any(f"up_blocks.{i}." in n for i in self.cfg.target_up_block_indices)
        ]

        if not target_attns:
            print(f"  [WARN] Aucune attn1 dans up_blocks {self.cfg.target_up_block_indices}")
            print(f"  → Fallback : hook sur TOUTES les self-attentions du UNet")
            target_attns = self_attns

        installed = 0
        for name, module in target_attns:
            module.set_processor(StylusAttnProcessor(self.store, name))
            installed += 1

        print(f"  Hookées ({installed} couches) :")
        for name, _ in target_attns:
            print(f"    {name}")

    # ── Encoding du conditioning AudioLDM2 ───────────────────────────────────

    def _get_audio_conditioning(self, wav: np.ndarray, label: str = ""):
        """
        Produit les embeddings de conditioning AudioLDM2 depuis un audio.

        Flux : CLAP(audio) → projection → GPT-2 → (generated_embeds, t5_embeds)

        Utilisé pour :
          - Conditionner l'inversion style avec l'audio du style
          - Conditionner l'inversion content avec l'audio du content
          - Conditionner le reverse avec l'audio du style (guide la génération)

        Avoir le bon conditioning pour chaque audio est critique : sinon le UNet
        génère selon un conditioning générique qui tire vers une distribution moyenne,
        et le style transfer par attention est écrasé par ce biais.
        """
        pipe  = self._pipe
        dev   = self.cfg.device

        clap_model        = pipe.text_encoder
        feature_extractor = pipe.feature_extractor
        projection_model  = pipe.projection_model
        language_model    = pipe.language_model
        t5_model          = pipe.text_encoder_2
        tokenizer_2       = pipe.tokenizer_2

        with torch.no_grad():
            # ── CLAP audio embed ──
            clap_sr  = self.cfg.clap_sr
            wav_48k  = librosa.resample(wav.astype(np.float32),
                                         orig_sr=self.cfg.sample_rate, target_sr=clap_sr)
            fe_inputs = feature_extractor(
                wav_48k, sampling_rate=clap_sr, return_tensors="pt"
            )
            fe_inputs = {k: v.to(dev) for k, v in fe_inputs.items()}
            clap_dtype = next(clap_model.parameters()).dtype
            fe_inputs = {k: v.to(clap_dtype) if v.is_floating_point() else v
                         for k, v in fe_inputs.items()}

            clap_embed = clap_model.get_audio_features(**fe_inputs)  # (1, D_clap)
            clap_embed = clap_embed.unsqueeze(1)                      # (1, 1, D_clap)

            # ── T5 : description générique (même pour tous) ──
            toks2    = tokenizer_2("music", padding=True, return_tensors="pt").to(dev)
            t5_embed = t5_model(**toks2).last_hidden_state  # (1, seq, D_t5)

            # ── Projection CLAP + T5 → espace partagé → GPT-2 ──
            proj_dtype  = next(projection_model.parameters()).dtype
            proj_out    = projection_model(
                hidden_states=clap_embed.to(proj_dtype),
                hidden_states_1=t5_embed[:, :1, :].to(proj_dtype),
            )
            proj_hidden = proj_out.hidden_states

            lm_out = language_model(
                inputs_embeds=proj_hidden,
                output_hidden_states=True,
                return_dict=True,
            )
            generated_embeds = lm_out.last_hidden_state  # (1, seq, D_lm)

        unet_dtype       = next(self._pipe.unet.parameters()).dtype
        generated_embeds = generated_embeds.to(dtype=unet_dtype)
        t5_embed         = t5_embed.to(dtype=unet_dtype)

        if label:
            print(f"  CLAP conditioning [{label}] : "
                  f"gen={generated_embeds.shape}, t5={t5_embed.shape}")
        return generated_embeds, t5_embed

    # ── VAE encode / decode ───────────────────────────────────────────────────

    @torch.no_grad()
    def _vae_encode(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """mel (1,1,T,F) → latent z mis à l'échelle."""
        vae   = self._pipe.vae
        dtype = next(vae.parameters()).dtype
        mel   = mel_tensor.to(self.cfg.device, dtype=dtype)
        z     = vae.encode(mel).latent_dist.mode()
        return z * vae.config.scaling_factor

    def _mel_stats(self, mel: np.ndarray):
        """Retourne (mean, std) du mel pour pouvoir dénormaliser après décodage VAE."""
        return float(mel.mean()), float(mel.std() + 1e-8)

    @torch.no_grad()
    def _vae_decode_to_mel(self, z: torch.Tensor) -> np.ndarray:
        """
        Latent z → mel log-amplitude (F, T) dans l'espace librosa dénormalisé.
        Utilisé pour la reconstruction phase-preserving à la Stylus.
        """
        vae   = self._pipe.vae
        dtype = next(vae.parameters()).dtype

        with torch.no_grad():
            mel_out = vae.decode(z.to(dtype) / vae.config.scaling_factor).sample
        # mel_out : (1, 1, T, F)
        mel_np = mel_out.squeeze().cpu().float().numpy()  # (T, F) ou (F, T)

        # Forcer orientation (F, T)
        if mel_np.shape[0] > mel_np.shape[1]:
            mel_np = mel_np.T   # (F, T)

        # Dénormalisation exacte avec les stats sauvegardées pendant l'encodage
        mel_mean = getattr(self, '_mel_mean', -6.0)
        mel_std  = getattr(self, '_mel_std',  2.0)
        return mel_np * mel_std + mel_mean   # log-mel dénormalisé (F, T)

    def _vae_decode_to_audio(self, z: torch.Tensor,
                              content_stft: np.ndarray = None) -> np.ndarray:
        """
        Latent z → audio via méthode Stylus :
          1. VAE decode → mel log-amplitude dénormalisé
          2. Mel → amplitude STFT (pseudo-inverse filterbank)
          3. Griffin-Lim initialisé avec la phase du STFT content (8 iter)

        Si content_stft=None → Griffin-Lim sans ancrage de phase (reconstruction seule).
        """
        cfg    = self.cfg
        mel_np = self._vae_decode_to_mel(z)   # log-mel (F, T)

        # Log-mel → amplitude mel
        mel_amp = np.exp(np.clip(mel_np, -12, 2))   # (F, T)

        if content_stft is None:
            # Reconstruction simple sans phase content
            wav = librosa.feature.inverse.mel_to_audio(
                mel_amp, sr=cfg.sample_rate,
                n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                win_length=cfg.n_fft, fmin=0.0, fmax=8000.0,
                n_iter=32, norm="slaney",
            )
        else:
            # ── Méthode Stylus : phase du content + 8 iter Griffin-Lim ──

            # 1. Amplitude mel → magnitude STFT via transposée pondérée
            mel_fb = librosa.filters.mel(
                sr=cfg.sample_rate, n_fft=cfg.n_fft,
                n_mels=cfg.n_mels, fmin=0.0, fmax=8000.0, norm="slaney",
            )
            mel_fb_norm = mel_fb / (mel_fb.sum(axis=0, keepdims=True) + 1e-10)
            mag_stft = np.maximum(mel_fb_norm.T @ mel_amp, 0.0)  # (F_stft, T)

            # Aligner temporellement avec le STFT content
            T_stft = content_stft.shape[1]
            if mag_stft.shape[1] != T_stft:
                mag_t = torch.from_numpy(mag_stft).float().unsqueeze(0).unsqueeze(0)
                mag_t = F.interpolate(mag_t, size=(mag_stft.shape[0], T_stft),
                                      mode="bilinear", align_corners=False)
                mag_stft = mag_t.squeeze().numpy()

            # Masque rolloff HF
            freqs  = np.linspace(0, cfg.sample_rate / 2, mag_stft.shape[0])
            mask   = np.where(freqs <= 8000.0, 1.0,
                              np.maximum(0.0, 1.0 - (freqs - 8000.0) / 500.0))
            mag_stft = mag_stft * mask[:, np.newaxis]

            # 2. Initialiser avec la phase du content
            stft_cur = mag_stft * np.exp(1j * np.angle(content_stft))

            target_len = librosa.istft(
                content_stft, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                window="hann", center=True,
            ).shape[0]

            # 3. 8 itérations Griffin-Lim ancrées sur la magnitude stylisée
            for _ in range(8):
                audio_tmp = librosa.istft(
                    stft_cur, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                    window="hann", center=True, length=target_len,
                )
                stft_cur = librosa.stft(
                    audio_tmp, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                    window="hann", center=True,
                )
                stft_cur = mag_stft * np.exp(1j * np.angle(stft_cur))

            wav = librosa.istft(
                stft_cur, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                window="hann", center=True, length=target_len,
            )

        if np.abs(wav).max() > 0:
            wav = wav / np.abs(wav).max() * 0.9
        return wav.astype(np.float32)

    # ── DDIM Inversion (t=0→T) ───────────────────────────────────────────────

    @torch.no_grad()
    def _ddim_inversion(
        self,
        z0: torch.Tensor,
        gen_embeds: torch.Tensor,
        t5_embeds: torch.Tensor,
        label: str = "",
    ) -> torch.Tensor:
        """
        Inversion DDIM déterministe z0 → zT.
        Le mode AttentionStore (capture_style ou capture_content) est déjà
        positionné avant l'appel.
        """
        sched     = self._pipe.scheduler
        sched.set_timesteps(self.cfg.num_inference_steps)
        timesteps = sched.timesteps.flip(0)  # 0 → T (sens inversion)
        alphas    = sched.alphas_cumprod.to(z0.device)
        stride    = sched.config.num_train_timesteps // self.cfg.num_inference_steps

        zt    = z0.clone().to(dtype=torch.float32)
        dtype = next(self._pipe.unet.parameters()).dtype

        for i, t in enumerate(timesteps):
            self.store.set_timestep(int(t))

            with torch.no_grad():
                noise_pred = self._pipe.unet(
                    zt.to(dtype),
                    t,
                    encoder_hidden_states=gen_embeds,
                    encoder_hidden_states_1=t5_embeds,
                    return_dict=False,
                )[0].float()

            t_next  = min(int(t) + stride, sched.config.num_train_timesteps - 1)
            a_t     = alphas[int(t)]
            a_n     = alphas[t_next]
            x0_pred = (zt - (1 - a_t).sqrt() * noise_pred) / a_t.sqrt().clamp(min=1e-8)
            zt      = a_n.sqrt() * x0_pred + (1 - a_n).sqrt() * noise_pred

            if (i + 1) % 10 == 0:
                print(f"    [{label}] inversion {i+1}/{len(timesteps)}")

        return zt

    # ── DDIM Reverse (t=T→0) avec injection ──────────────────────────────────

    @torch.no_grad()
    def _ddim_reverse(
        self,
        zT: torch.Tensor,
        gen_embeds: torch.Tensor,
        t5_embeds: torch.Tensor,
        label: str = "",
    ) -> torch.Tensor:
        """
        Reverse DDIM zT → z0 en mode inject.
        Utilise Q_content[t] + K_style[t] + V_style[t] à chaque step.
        """
        sched = self._pipe.scheduler
        sched.set_timesteps(self.cfg.num_inference_steps)
        zt    = zT.clone()
        dtype = next(self._pipe.unet.parameters()).dtype

        for i, t in enumerate(sched.timesteps):
            self.store.set_timestep(int(t))

            with torch.no_grad():
                noise_pred = self._pipe.unet(
                    zt.to(dtype),
                    t,
                    encoder_hidden_states=gen_embeds,
                    encoder_hidden_states_1=t5_embeds,
                    return_dict=False,
                )[0]

            zt = sched.step(noise_pred, t, zt).prev_sample

            if (i + 1) % 10 == 0:
                print(f"    [{label}] denoising {i+1}/{len(sched.timesteps)}")

        return zt

    # ── Transfer principal ────────────────────────────────────────────────────

    def transfer(
        self,
        style_path: str,
        content_path: str,
        output_dir: Optional[str] = None,
    ) -> np.ndarray:

        if self._pipe is None:
            self.load_model()

        out_dir = output_dir or self.cfg.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # ── 1. Conditioning audio-specific (calculé après chargement des wavs) ──
        # On charge d'abord les wavs, puis on calcule les conditionnings
        print("\n[1/6] Chargement audio + conversion mel...")
        wav_style   = self.proc.load_audio(style_path)
        wav_content = self.proc.load_audio(content_path)

        mel_style   = self.proc.audio_to_mel(wav_style)
        mel_content = self.proc.audio_to_mel(wav_content)
        stft_content = self.proc.audio_to_stft(wav_content)  # phase pour reconstruction

        # Aligner style et content sur la même longueur temporelle (min des deux)
        T_style   = mel_style.shape[1]
        T_content = mel_content.shape[1]
        T_common  = min(T_style, T_content)
        T_padded  = ((T_common + 7) // 8) * 8
        print(f"  Durée commune : {T_common} frames "
              f"({T_common * self.cfg.hop_length / self.cfg.sample_rate:.2f}s) "
              f"→ padded à {T_padded} frames")

        mel_vae_style   = self.proc.mel_to_vae_input(mel_style,   target_length=T_padded)
        mel_vae_content = self.proc.mel_to_vae_input(mel_content, target_length=T_padded)

        print(f"  Mel style   : {mel_style.shape}  → VAE input {mel_vae_style.shape}")
        print(f"  Mel content : {mel_content.shape} → VAE input {mel_vae_content.shape}")

        self.proc.save_mel_plot(mel_style,   os.path.join(out_dir, "mel_style.png"),   "Style (chime)")
        self.proc.save_mel_plot(mel_content, os.path.join(out_dir, "mel_content.png"), "Content (violin)")
        self.proc.save_audio(wav_style,   os.path.join(out_dir, "00_input_style.wav"))
        self.proc.save_audio(wav_content, os.path.join(out_dir, "00_input_content.wav"))

        # Conditioning CLAP audio spécifique à chaque source
        print("\n[1b/6] Calcul des conditionnings CLAP audio...")
        gen_style,   t5_style   = self._get_audio_conditioning(wav_style,   label="style")
        gen_content, t5_content = self._get_audio_conditioning(wav_content, label="content")

        # [2/6] déjà fait dans [1/6]

        # ── 3. VAE encode ─────────────────────────────────────────────────────
        print("\n[3/6] VAE encode...")
        z0_style   = self._vae_encode(mel_vae_style)
        z0_content = self._vae_encode(mel_vae_content)
        print(f"  z0_style   : {z0_style.shape}")
        print(f"  z0_content : {z0_content.shape}")

        # Sauvegarder les stats mel content pour dénorm exacte dans decode
        self._mel_mean, self._mel_std = self._mel_stats(mel_content)
        print(f"  Stats mel content : mean={self._mel_mean:.3f}, std={self._mel_std:.3f}")

        # Test reconstruction VAE (référence qualité)
        print("  Test reconstruction VAE content (phase-preserving)...")
        wav_recon = self._vae_decode_to_audio(z0_content, content_stft=stft_content)
        self.proc.save_audio(wav_recon, os.path.join(out_dir, "01_vae_reconstruction.wav"))

        # ── 4. DDIM inversion style → capture K_style[t], V_style[t] ─────────
        # Utilise le conditioning CLAP du style pour guider l'inversion
        print("\n[4/6] DDIM inversion style (capture K, V)...")
        self.store.mode = "capture_style"
        zT_style = self._ddim_inversion(z0_style, gen_style, t5_style, label="style")

        # ── 5. DDIM inversion content → capture Q_content[t] ─────────────────
        # Utilise le conditioning CLAP du content pour guider son inversion
        print("\n[5/6] DDIM inversion content (capture Q)...")
        self.store.mode = "capture_content"
        zT_content = self._ddim_inversion(z0_content, gen_content, t5_content, label="content")

        # Diagnostic : vérifier que des Q/K/V ont bien été capturés
        n_ks = len(self.store._ks)
        n_qs = len(self.store._qs)
        print(f"  Diagnostic capture — K/V style: {n_ks} entrées, Q content: {n_qs} entrées")
        if n_ks == 0 or n_qs == 0:
            print("  [ERREUR] Aucune attention capturée ! Les hooks ne fonctionnent pas.")
            print("  Vérifier que set_processor() est supporté par cette version de diffusers.")

        # ── 6. AdaIN + DDIM reverse (inject) ──────────────────────────────────
        print("\n[6/6] AdaIN(zT_content, zT_style) + DDIM reverse (inject)...")
        zT_init = adain_latent(zT_content, zT_style)
        print(f"  γ={self.cfg.gamma} (query preservation)  "
              f"α={self.cfg.alpha} (style guidance)")

        # Le reverse est guidé par le conditioning du STYLE
        # → le UNet pousse vers le style pendant que les Q/K/V injectés préservent le content
        self.store.mode = "inject"
        z0_out = self._ddim_reverse(zT_init, gen_style, t5_style, label="stylized")
        self.store.mode = "off"

        # ── Décodage audio (méthode Stylus : phase content + 8 iter GL) ─────
        print("\n[+] Décodage phase-preserving (Stylus) → audio...")
        wav_out = self._vae_decode_to_audio(z0_out, content_stft=stft_content)
        print(f"  Waveform shape : {wav_out.shape}  "
              f"({len(wav_out)/self.cfg.sample_rate:.2f}s)")

        out_name = f"stylized_g{self.cfg.gamma}_a{self.cfg.alpha}.wav"
        self.proc.save_audio(wav_out, os.path.join(out_dir, out_name))

        # Mel de sortie pour visualisation
        # Mel stylized depuis le latent directement (avant Griffin-Lim)
        mel_out = self._vae_decode_to_mel(z0_out)
        self.proc.save_mel_plot(mel_out, os.path.join(out_dir, "mel_stylized.png"),
                                f"Stylized (γ={self.cfg.gamma}, α={self.cfg.alpha})")

        print(f"\n{'─'*55}")
        print(f"  Fichiers dans {out_dir}/ :")
        print(f"    00_input_style.wav        ← style  (chime)")
        print(f"    00_input_content.wav      ← content (violin)")
        print(f"    01_vae_reconstruction.wav ← reconstruction VAE (qualité ref)")
        print(f"    {out_name}  ← résultat")
        print(f"    mel_*.png                 ← spectrogrammes")
        print(f"{'─'*55}")
        print(f"""
  Ajustements :
    gamma : 0.0 = structure libre   / 1.0 = structure content préservée
    alpha : 0.0 = content pur       / 1.0 = style pur
""")
        return wav_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = StylusAudioLDM2Config(
        gamma=0.1,   # 0=structure libre / 1=structure content préservée
        alpha=0.8,   # 0=content pur      / 1=style pur
        num_inference_steps=50,
        duration=-1.0,   # -1 = charge tout le fichier audio (5s, 10s, etc.)
        style_audio_path="musicTI_dataset/audios/timbre/chime/chime1.wav",
        content_audio_path="musicTI_dataset/audios/content/violin/violin1.wav",
        output_dir="stylus_audioldm2_output",
    )

    print("=" * 55)
    print("  Stylus-AudioLDM2 — Music Style Transfer")
    print(f"  Style   : {cfg.style_audio_path}")
    print(f"  Content : {cfg.content_audio_path}")
    print(f"  γ={cfg.gamma}  α={cfg.alpha}  steps={cfg.num_inference_steps}")
    print(f"  Device  : {cfg.device}")
    print("=" * 55)

    pipeline = StylusAudioLDM2Pipeline(cfg)
    pipeline.load_model()
    pipeline.transfer(
        style_path=cfg.style_audio_path,
        content_path=cfg.content_audio_path,
        output_dir=cfg.output_dir,
    )


if __name__ == "__main__":
    main()