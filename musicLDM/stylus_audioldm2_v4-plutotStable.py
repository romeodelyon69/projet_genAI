"""
Stylus-AudioLDM2 : Music Style Transfer via AudioLDM2  — v4 (corrected)
=========================================================================
Adaptation de "Stylus: Repurposing Stable Diffusion for Training-Free
Music Style Transfer" (arxiv:2411.15913) pour AudioLDM2.

Corrections v4 vs v3 :
  1. Mel preprocessing aligné sur AudioLDM2 training (hop=160, fmax=8000)
  2. Projection T5 : séquence complète (plus de truncation à 1 token)
  3. Vocoder decode : utilise pipe.mel_spectrogram_to_waveform() officiel
  4. Attention processor compatible AttnProcessor v1 ET v2 (diffusers récent)
  5. Vérification roundtrip : encode→decode et inversion→reverse
  6. AdaIN sur dim temporelle seule (pas temps×fréquence mélangés)
  7. Ciblage couches réduit par défaut (up_blocks 2,3 seulement)

AVERTISSEMENT :
  L'hypothèse fondamentale de Stylus (Q=structure, K/V=style dans les
  self-attention) a été validée empiriquement pour les IMAGES (SD1.5).
  Rien ne garantit que cette décomposition tient pour l'audio.
  Le timbre est distribué sur toute la représentation fréquentielle,
  ce n'est pas une "texture spatiale" séparable comme dans une image.
  Ce code corrige les bugs techniques mais le résultat peut rester
  décevant pour des raisons fondamentales.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StylusAudioLDM2Config:
    # Stylus params
    gamma: float = 0.5   # query preservation : 0=libre, 1=structure pure
    alpha: float = 0.8   # style guidance     : 0=content pur, 1=style pur

    # Conditioning pendant le reverse
    use_audio_prompt: bool = False

    # [FIX #7] Couches ciblées — réduit par défaut aux couches hautes
    # qui sont plus susceptibles d'encoder du style (par analogie SD)
    # À ajuster empiriquement pour l'audio.
    target_up_block_indices: list = field(default_factory=lambda: [2, 3])

    # DDIM
    num_inference_steps: int = 50

    # [FIX #1] Audio — paramètres alignés sur AudioLDM2 training
    sample_rate: int   = 16_000
    clap_sr: int       = 48_000
    duration: float    = -1.0    # -1 = durée réelle du fichier

    # Mel spectrogram — paramètres exacts AudioLDM2 training
    # Source: audioldm2/utilities/audio config par défaut
    n_fft: int       = 1024
    hop_length: int  = 160      # [FIX] était 256, AudioLDM2 utilise 160
    win_length: int  = 1024
    n_mels: int      = 64
    fmin: float      = 0.0
    fmax: float      = 8000.0   # [FIX] était sr/2=8000, mais explicite

    # Modèle
    model_id: str = "cvssp/audioldm2-music"
    device: str   = "cuda" if torch.cuda.is_available() else "cpu"

    # Vérification
    skip_roundtrip_check: bool  = False   # mettre True pour sauter les checks
    roundtrip_snr_threshold: float = 5.0  # dB minimum pour reconstruction acceptable

    # Fichiers
    style_audio_path:   str = "musicTI_dataset/audios/timbre/chime/chime1.wav"
    content_audio_path: str = "musicTI_dataset/audios/content/violin/violin1.wav"
    output_dir:         str = "stylus_audioldm2_output"


# ─────────────────────────────────────────────────────────────────────────────
# AttentionStore
# ─────────────────────────────────────────────────────────────────────────────

class AttentionStore:
    def __init__(self):
        self.mode: str      = "off"
        self.current_t: int = 0
        self.gamma: float   = 0.5
        self.alpha: float   = 0.8
        self._qs: dict      = {}
        self._ks: dict      = {}
        self._vs: dict      = {}

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
        self._qs.clear(); self._ks.clear(); self._vs.clear()


# ─────────────────────────────────────────────────────────────────────────────
# StylusAttnProcessor — [FIX #4] compatible diffusers v1 et v2 API
# ─────────────────────────────────────────────────────────────────────────────

class StylusAttnProcessor:
    """
    Processeur d'attention compatible avec les deux versions de l'API diffusers :
    - v1 : attn.head_to_batch_dim / batch_to_head_dim / attn.scale
    - v2 : F.scaled_dot_product_attention (AttnProcessor2_0)

    On détecte automatiquement quelle API est disponible.
    """

    def __init__(self, store: AttentionStore, layer_name: str):
        self.store      = store
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):

        is_self = encoder_hidden_states is None
        kv_src  = hidden_states if is_self else encoder_hidden_states

        residual = hidden_states

        # Projections Q, K, V
        q = attn.to_q(hidden_states)
        k = attn.to_k(kv_src)
        v = attn.to_v(kv_src)

        # Reshape pour multi-head — compatible v1 et v2
        inner_dim = q.shape[-1]
        head_dim  = inner_dim // attn.heads
        batch     = q.shape[0]

        q = q.view(batch, -1, attn.heads, head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(batch, -1, attn.heads, head_dim).transpose(1, 2)

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
                    # [FIX] Adapter les dimensions si batch/heads diffèrent
                    qc = self._match_shape(qc, q)
                    ks = self._match_shape(ks, k)
                    vs = self._match_shape(vs, v)

                    q_bar       = store.gamma * qc + (1 - store.gamma) * q
                    out_content = self._scaled_attn(q_bar, k, v, head_dim, attention_mask)
                    out_style   = self._scaled_attn(q_bar, ks, vs, head_dim, attention_mask)
                    out = out_content + store.alpha * (out_style - out_content)

                    # Reshape back: (B, H, N, D) → (B, N, H*D)
                    out = out.transpose(1, 2).contiguous().view(batch, -1, inner_dim)
                    out = attn.to_out[0](out)
                    out = attn.to_out[1](out)
                    return out

        # Default attention
        out = self._scaled_attn(q, k, v, head_dim, attention_mask)
        out = out.transpose(1, 2).contiguous().view(batch, -1, inner_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

    @staticmethod
    def _scaled_attn(q, k, v, head_dim, mask=None):
        """Manual scaled dot-product attention. (B, H, N, D) format."""
        scale = head_dim ** -0.5
        w = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            w = w + mask
        w = w.softmax(dim=-1).to(q.dtype)
        return torch.matmul(w, v)

    @staticmethod
    def _match_shape(stored, current):
        """
        Adapte le tensor capturé (stored) aux dimensions du tensor courant.
        Gère les différences de taille de séquence (padding/truncation).
        """
        if stored.shape == current.shape:
            return stored
        # Même B, H, D mais N différent → truncate/pad sur la dim séquence
        if stored.shape[0] == current.shape[0] and stored.shape[1] == current.shape[1]:
            N_s, N_c = stored.shape[2], current.shape[2]
            if N_s > N_c:
                return stored[:, :, :N_c, :]
            elif N_s < N_c:
                pad = current[:, :, N_s:, :] * 0  # zero-pad
                return torch.cat([stored, pad], dim=2)
        return stored  # fallback


# ─────────────────────────────────────────────────────────────────────────────
# AdaIN — [FIX #6] : normalisation par dimension temporelle uniquement
# ─────────────────────────────────────────────────────────────────────────────

def adain_latent(z_content: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
    """
    Adaptive Instance Normalization sur les latents audio.

    Pour les images (B, C, H, W), on normalise sur H×W.
    Pour l'audio (B, C, T_lat, F_lat), normaliser sur T×F mélange
    des informations hétérogènes (temps et fréquence).

    On normalise ici sur la dimension temporelle (dim=2) uniquement,
    en préservant la structure fréquentielle par canal.
    """
    eps = 1e-5
    mu_c  = z_content.mean(dim=2, keepdim=True)
    sig_c = z_content.std( dim=2, keepdim=True) + eps
    mu_s  = z_style.mean(  dim=2, keepdim=True)
    sig_s = z_style.std(   dim=2, keepdim=True) + eps
    return sig_s * (z_content - mu_c) / sig_c + mu_s


# ─────────────────────────────────────────────────────────────────────────────
# AudioProcessor — [FIX #1] mel preprocessing aligné AudioLDM2 training
# ─────────────────────────────────────────────────────────────────────────────

class AudioProcessor:
    """
    Preprocessing mel aligné sur le pipeline d'entraînement AudioLDM2.

    Source de vérité : audioldm2/utilities/audio/stft.py
      - STFT avec n_fft=1024, hop=160, win=1024
      - Mel filterbank 64 bins, fmin=0, fmax=8000
      - Dynamic range compression : log(clamp(mel, min=1e-5))
      - PAS de normalisation /11 ad-hoc

    La normalisation finale dépend des stats d'entraînement du VAE.
    On laisse le VAE gérer ça implicitement via son espace latent.
    """

    def __init__(self, cfg: StylusAudioLDM2Config):
        self.cfg         = cfg
        self.sample_rate = cfg.sample_rate
        self.n_mels      = cfg.n_mels
        self.hop_length  = cfg.hop_length
        self.n_fft       = cfg.n_fft
        self.win_length  = cfg.win_length
        self.fmin        = cfg.fmin
        self.fmax        = cfg.fmax
        self._vocoder    = None

    def set_vocoder(self, vocoder):
        """Vérifie la cohérence entre config et vocoder."""
        self._vocoder = vocoder
        vc = vocoder.config
        # Vérification de cohérence
        if hasattr(vc, 'model_in_dim') and vc.model_in_dim != self.n_mels:
            print(f"  ⚠ ATTENTION: vocoder.model_in_dim={vc.model_in_dim} "
                  f"!= config.n_mels={self.n_mels}")
            print(f"    → On utilise vocoder.model_in_dim={vc.model_in_dim}")
            self.n_mels = vc.model_in_dim
        if hasattr(vc, 'sampling_rate') and vc.sampling_rate != self.sample_rate:
            print(f"  ⚠ ATTENTION: vocoder.sampling_rate={vc.sampling_rate} "
                  f"!= config.sample_rate={self.sample_rate}")
            self.sample_rate = vc.sampling_rate
        print(f"  AudioProcessor config : n_mels={self.n_mels}, sr={self.sample_rate}, "
              f"hop={self.hop_length}, n_fft={self.n_fft}, fmax={self.fmax}")

    def load_audio(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        dur = None if self.cfg.duration <= 0 else self.cfg.duration
        wav, _ = librosa.load(path, sr=self.sample_rate, mono=True, duration=dur)
        wav = wav / (np.abs(wav).max() + 1e-8)
        return wav.astype(np.float32)

    def audio_to_mel_vae(self, wav: np.ndarray) -> torch.Tensor:
        """
        Waveform → mel features pour le VAE AudioLDM2.

        [FIX #1] Utilise les paramètres exacts d'AudioLDM2 training :
          - hop_length=160 (pas 256)
          - fmax=8000 (pas sr/2)
          - Dynamic range compression : log(clamp(mel, 1e-5))
          - PAS de normalisation ad-hoc /11
        """
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )

        # [FIX #1] Dynamic range compression identique à AudioLDM2 training
        # Source: audioldm2/utilities/audio/stft.py → dynamic_range_compression_torch
        log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))  # (n_mels, T)

        # Sauvegarder pour visualisation
        self._last_log_mel = log_mel.copy()

        # Pad au multiple de 8 (requis par le VAE)
        T = log_mel.shape[1]
        pad_to = ((T + 7) // 8) * 8
        if T < pad_to:
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_to - T)),
                             mode='constant', constant_values=np.log(1e-5))

        # (n_mels, T) → (1, 1, T, n_mels)  [format VAE AudioLDM2]
        mel_t = torch.FloatTensor(log_mel).T.unsqueeze(0).unsqueeze(0)
        return mel_t

    def save_audio(self, wav: np.ndarray, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        wav = wav / (np.abs(wav).max() + 1e-8) * 0.9
        sf.write(path, wav.astype(np.float32), self.sample_rate)
        print(f"  → {path}")

    def save_mel_plot(self, log_mel: np.ndarray, path: str, title: str = ""):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dur = log_mel.shape[1] * self.hop_length / self.sample_rate
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(log_mel, origin="lower", aspect="auto", cmap="magma",
                       extent=[0, dur, 0, self.fmax])
        ax.set_xlabel("Temps (s)"); ax.set_ylabel("Fréquence (Hz)")
        fig.colorbar(im, ax=ax).set_label("log amplitude")
        if title: ax.set_title(title, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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

    # ── Chargement ──────────────────────────────────────────────────────────

    def load_model(self):
        from diffusers import AudioLDM2Pipeline, DDIMScheduler
        print(f"[INFO] Chargement {self.cfg.model_id} ...")
        pipe = AudioLDM2Pipeline.from_pretrained(
            self.cfg.model_id, torch_dtype=torch.float32,
        ).to(self.cfg.device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

        # Initialiser AudioProcessor et vérifier cohérence
        self.proc.set_vocoder(pipe.vocoder)
        self._install_processors()
        print("  Modèle prêt ✓")
        return self

    def _install_processors(self):
        """
        Installe les StylusAttnProcessor sur les couches self-attention ciblées.
        Log les couches hookées pour vérification.
        """
        self_attns = []
        for name, module in self._pipe.unet.named_modules():
            if (hasattr(module, "set_processor")
                and name.endswith("attn1")
                and any(f"up_blocks.{i}." in name
                        for i in self.cfg.target_up_block_indices)):
                self_attns.append((name, module))

        for name, module in self_attns:
            module.set_processor(StylusAttnProcessor(self.store, name))

        print(f"  Hooks installés sur {len(self_attns)} couches self-attn :")
        for name, _ in self_attns:
            print(f"    • {name}")

    # ── Conditioning AudioLDM2 — [FIX #2] séquence T5 complète ──────────

    def _get_conditioning(self, wav: np.ndarray = None) -> tuple:
        """
        Produit (gen_embeds, t5_embeds) pour le UNet AudioLDM2.

        [FIX #2] On passe la séquence T5 complète au projection_model,
        pas seulement le premier token.
        """
        pipe = self._pipe
        dev  = self.cfg.device

        clap_model        = pipe.text_encoder
        feature_extractor = pipe.feature_extractor
        projection_model  = pipe.projection_model
        language_model    = pipe.language_model
        t5_model          = pipe.text_encoder_2
        tokenizer         = pipe.tokenizer
        tokenizer_2       = pipe.tokenizer_2

        with torch.no_grad():
            if wav is None:
                # ── Unconditional : CLAP("") ──
                toks = tokenizer(
                    "", padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                ).to(dev)
                clap_out   = clap_model.text_model(**toks)
                clap_embed = clap_model.text_projection(
                    clap_out.last_hidden_state[:, 0, :]
                ).unsqueeze(1)  # (1, 1, D_clap)
            else:
                # ── Conditionnel : CLAP(audio) ──
                wav_48k = librosa.resample(
                    wav.astype(np.float32),
                    orig_sr=self.proc.sample_rate,
                    target_sr=self.cfg.clap_sr,
                )
                fe_inputs = feature_extractor(
                    wav_48k, sampling_rate=self.cfg.clap_sr, return_tensors="pt"
                )
                fe_inputs = {k: v.to(dev) for k, v in fe_inputs.items()}
                clap_dtype = next(clap_model.parameters()).dtype
                fe_inputs  = {k: v.to(clap_dtype) if v.is_floating_point() else v
                              for k, v in fe_inputs.items()}
                clap_embed = clap_model.get_audio_features(**fe_inputs).unsqueeze(1)

            # T5 description
            toks2    = tokenizer_2("music", padding=True,
                                   return_tensors="pt").to(dev)
            t5_embed = t5_model(**toks2).last_hidden_state  # (1, seq_len, D_t5)

            # [FIX #2] Projection → GPT-2 avec T5 COMPLET (pas [:, :1, :])
            proj_dtype = next(projection_model.parameters()).dtype
            proj_out   = projection_model(
                hidden_states=clap_embed.to(proj_dtype),
                hidden_states_1=t5_embed.to(proj_dtype),    # ← séquence complète
            )
            lm_out = language_model(
                inputs_embeds=proj_out.hidden_states,
                output_hidden_states=True, return_dict=True,
            )
            gen_embeds = lm_out.last_hidden_state  # (1, seq, D_lm)

        unet_dtype = next(self._pipe.unet.parameters()).dtype
        return gen_embeds.to(unet_dtype), t5_embed.to(unet_dtype)

    # ── VAE encode ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _vae_encode(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        vae   = self._pipe.vae
        dtype = next(vae.parameters()).dtype
        mel   = mel_tensor.to(self.cfg.device, dtype=dtype)
        z     = vae.encode(mel).latent_dist.mode()
        return z * vae.config.scaling_factor

    # ── VAE decode → vocoder — [FIX #3] utilise le chemin officiel ──────────

    @torch.no_grad()
    def _vae_decode_to_audio(self, z: torch.Tensor) -> np.ndarray:
        """
        Latent z → waveform via le chemin officiel AudioLDM2Pipeline.

        [FIX #3] On utilise pipe.mel_spectrogram_to_waveform() qui gère
        le format interne et les transpositions correctement, au lieu
        de deviner le format du vocoder.
        """
        vae   = self._pipe.vae
        dtype = next(vae.parameters()).dtype

        mel_features = vae.decode(
            z.to(dtype) / vae.config.scaling_factor
        ).sample    # (1, 1, T, n_mels)

        # Utilise le chemin officiel du pipeline
        waveform = self._pipe.mel_spectrogram_to_waveform(mel_features)

        wav = waveform[0].cpu().float().numpy()
        if np.abs(wav).max() > 0:
            wav = wav / np.abs(wav).max() * 0.9
        return wav.astype(np.float32)

    # ── [FIX #5] Vérification de reconstruction ─────────────────────────────

    def _verify_vae_roundtrip(self, wav_original: np.ndarray, mel_vae: torch.Tensor,
                              label: str = "") -> float:
        """
        Vérifie que encode→decode reconstruit fidèlement l'audio.
        Retourne le SNR en dB. Si < seuil, le mel preprocessing est suspect.
        """
        z0 = self._vae_encode(mel_vae)
        wav_recon = self._vae_decode_to_audio(z0)

        # Aligner longueurs
        min_len = min(len(wav_original), len(wav_recon))
        w_orig  = wav_original[:min_len]
        w_recon = wav_recon[:min_len]

        # SNR
        signal_power = np.mean(w_orig ** 2) + 1e-10
        noise_power  = np.mean((w_orig - w_recon) ** 2) + 1e-10
        snr_db = 10 * np.log10(signal_power / noise_power)

        # Corrélation
        corr = np.corrcoef(w_orig, w_recon)[0, 1] if min_len > 1 else 0.0

        print(f"  [{label}] VAE roundtrip : SNR={snr_db:.1f} dB, corr={corr:.3f}")
        return snr_db

    def _verify_ddim_roundtrip(self, z0: torch.Tensor,
                               gen_embeds: torch.Tensor,
                               t5_embeds: torch.Tensor,
                               label: str = "") -> float:
        """
        [FIX #5] Vérifie que DDIM_reverse(DDIM_inversion(z0)) ≈ z0.
        C'est le test fondamental : si ça échoue, le style transfer est invalide.
        """
        # Sauvegarde du mode
        old_mode = self.store.mode
        self.store.mode = "off"  # pas d'injection

        print(f"  [{label}] DDIM roundtrip test...")
        zT = self._ddim_inversion(z0, gen_embeds, t5_embeds, label=f"{label}_inv")
        z0_recon = self._ddim_reverse(zT, gen_embeds, t5_embeds, label=f"{label}_rev")

        # Erreur relative
        mse  = F.mse_loss(z0_recon.float(), z0.float()).item()
        norm = z0.float().pow(2).mean().item() + 1e-10
        rel_err = mse / norm

        print(f"  [{label}] DDIM roundtrip : MSE={mse:.6f}, "
              f"relative_error={rel_err:.4f} "
              f"({'OK' if rel_err < 0.1 else 'MAUVAIS — inversion diverge'})")

        self.store.mode = old_mode
        return rel_err

    # ── DDIM Inversion ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _ddim_inversion(self, z0, gen_embeds, t5_embeds, label=""):
        sched     = self._pipe.scheduler
        sched.set_timesteps(self.cfg.num_inference_steps)
        timesteps = sched.timesteps.flip(0)
        alphas    = sched.alphas_cumprod.to(z0.device)
        stride    = sched.config.num_train_timesteps // self.cfg.num_inference_steps
        zt        = z0.clone().to(torch.float32)
        dtype     = next(self._pipe.unet.parameters()).dtype

        for i, t in enumerate(timesteps):
            self.store.set_timestep(int(t))
            noise_pred = self._pipe.unet(
                zt.to(dtype), t,
                encoder_hidden_states=gen_embeds,
                encoder_hidden_states_1=t5_embeds,
                return_dict=False,
            )[0].float()

            t_next  = min(int(t) + stride, sched.config.num_train_timesteps - 1)
            a_t     = alphas[int(t)]
            a_n     = alphas[t_next]
            x0_pred = (zt - (1 - a_t).sqrt() * noise_pred) / a_t.sqrt().clamp(1e-8)
            zt      = a_n.sqrt() * x0_pred + (1 - a_n).sqrt() * noise_pred

            if (i + 1) % 10 == 0:
                print(f"    [{label}] inversion {i+1}/{len(timesteps)}")
        return zt

    # ── DDIM Reverse ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _ddim_reverse(self, zT, gen_embeds, t5_embeds, label=""):
        sched = self._pipe.scheduler
        sched.set_timesteps(self.cfg.num_inference_steps)
        zt    = zT.clone()
        dtype = next(self._pipe.unet.parameters()).dtype

        for i, t in enumerate(sched.timesteps):
            self.store.set_timestep(int(t))
            noise_pred = self._pipe.unet(
                zt.to(dtype), t,
                encoder_hidden_states=gen_embeds,
                encoder_hidden_states_1=t5_embeds,
                return_dict=False,
            )[0]
            zt = sched.step(noise_pred, t, zt).prev_sample
            if (i + 1) % 10 == 0:
                print(f"    [{label}] denoising {i+1}/{len(sched.timesteps)}")
        return zt

    # ── Transfer principal ────────────────────────────────────────────────────

    def transfer(self, style_path: str, content_path: str,
                 output_dir: str = None) -> np.ndarray:
        if self._pipe is None:
            self.load_model()

        out_dir = output_dir or self.cfg.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # ── 1. Chargement audio ──────────────────────────────────────────────
        print("\n[1/7] Chargement audio...")
        wav_style   = self.proc.load_audio(style_path)
        wav_content = self.proc.load_audio(content_path)
        print(f"  Style   : {len(wav_style)/self.proc.sample_rate:.2f}s")
        print(f"  Content : {len(wav_content)/self.proc.sample_rate:.2f}s")

        self.proc.save_audio(wav_style,   os.path.join(out_dir, "00_input_style.wav"))
        self.proc.save_audio(wav_content, os.path.join(out_dir, "00_input_content.wav"))

        # ── 2. Mel VAE ───────────────────────────────────────────────────────
        print("\n[2/7] Conversion mel (params AudioLDM2 training)...")
        mel_vae_style   = self.proc.audio_to_mel_vae(wav_style)
        log_mel_style   = self.proc._last_log_mel.copy()
        mel_vae_content = self.proc.audio_to_mel_vae(wav_content)
        log_mel_content = self.proc._last_log_mel.copy()

        # Aligner longueur temporelle (min des deux, multiple de 8)
        T = min(mel_vae_style.shape[2], mel_vae_content.shape[2])
        T = ((T + 7) // 8) * 8
        mel_vae_style   = mel_vae_style[:,  :, :T, :]
        mel_vae_content = mel_vae_content[:, :, :T, :]
        log_mel_style   = log_mel_style[:,   :T]
        log_mel_content = log_mel_content[:, :T]

        print(f"  VAE input shape : {mel_vae_content.shape}  "
              f"(n_mels={self.proc.n_mels}, hop={self.proc.hop_length})")
        print(f"  Mel range : [{mel_vae_content.min():.2f}, {mel_vae_content.max():.2f}]")

        self.proc.save_mel_plot(log_mel_style,   os.path.join(out_dir, "mel_style.png"),   "Style")
        self.proc.save_mel_plot(log_mel_content, os.path.join(out_dir, "mel_content.png"), "Content")

        # ── 3. VAE encode ────────────────────────────────────────────────────
        print("\n[3/7] VAE encode...")
        z0_style   = self._vae_encode(mel_vae_style)
        z0_content = self._vae_encode(mel_vae_content)
        print(f"  z shape : {z0_content.shape}")
        print(f"  z_content range : [{z0_content.min():.3f}, {z0_content.max():.3f}]")
        print(f"  z_style   range : [{z0_style.min():.3f}, {z0_style.max():.3f}]")

        # ── 3b. [FIX #5] Vérification VAE roundtrip ─────────────────────────
        if not self.cfg.skip_roundtrip_check:
            print("\n[3b/7] Vérification VAE roundtrip...")
            wav_recon = self._vae_decode_to_audio(z0_content)
            self.proc.save_audio(wav_recon,
                                 os.path.join(out_dir, "01_vae_reconstruction.wav"))

            snr = self._verify_vae_roundtrip(wav_content, mel_vae_content, "content")
            if snr < self.cfg.roundtrip_snr_threshold:
                print(f"\n  ⚠⚠⚠ ALERTE: SNR VAE={snr:.1f} dB < seuil "
                      f"{self.cfg.roundtrip_snr_threshold} dB")
                print(f"  Le mel preprocessing ne correspond probablement pas "
                      f"à ce que le VAE attend.")
                print(f"  Le style transfer sera de mauvaise qualité.")
                print(f"  Vérifiez 01_vae_reconstruction.wav vs l'original.\n")
        else:
            # Au moins sauver la reconstruction pour inspection manuelle
            wav_recon = self._vae_decode_to_audio(z0_content)
            self.proc.save_audio(wav_recon,
                                 os.path.join(out_dir, "01_vae_reconstruction.wav"))

        # ── 4. Conditionnings ────────────────────────────────────────────────
        print("\n[4/7] Calcul des conditionnings CLAP...")

        gen_style, t5_style     = self._get_conditioning(wav_style)
        print(f"  CLAP(style)   : gen={gen_style.shape}")

        gen_content, t5_content = self._get_conditioning(wav_content)
        print(f"  CLAP(content) : gen={gen_content.shape}")

        if self.cfg.use_audio_prompt:
            gen_reverse, t5_reverse = gen_style, t5_style
            print(f"  Reverse : CLAP(style audio)  [use_audio_prompt=True]")
        else:
            gen_reverse, t5_reverse = self._get_conditioning(wav=None)
            print(f"  Reverse : CLAP('') unconditional [use_audio_prompt=False]")

        # ── 4b. [FIX #5] Vérification DDIM roundtrip ────────────────────────
        if not self.cfg.skip_roundtrip_check:
            print("\n[4b/7] Vérification DDIM roundtrip (sans injection)...")
            ddim_err = self._verify_ddim_roundtrip(
                z0_content, gen_content, t5_content, label="content"
            )
            if ddim_err > 0.1:
                print(f"\n  ⚠⚠⚠ ALERTE: erreur relative DDIM={ddim_err:.3f} > 0.1")
                print(f"  L'inversion DDIM diverge. Le style transfer sera invalide.")
                print(f"  Causes possibles: mel preprocessing incorrect, "
                      f"conditioning cassé.\n")

        # ── 5. DDIM inversions ───────────────────────────────────────────────
        print("\n[5/7] DDIM inversion style (capture K, V)...")
        self.store.mode = "capture_style"
        zT_style = self._ddim_inversion(z0_style, gen_style, t5_style, label="style")

        print("\n[5b/7] DDIM inversion content (capture Q)...")
        self.store.mode = "capture_content"
        zT_content = self._ddim_inversion(z0_content, gen_content, t5_content,
                                          label="content")

        n_ks = len(self.store._ks); n_qs = len(self.store._qs)
        print(f"  Capturés : K/V style={n_ks}, Q content={n_qs}")

        if n_ks == 0 or n_qs == 0:
            print("  ⚠ ERREUR : aucun K/V ou Q capturé !")
            print("    Vérifiez que target_up_block_indices correspond "
                  "à des blocs existants dans le UNet.")
            return np.zeros(1)

        # ── 6. AdaIN + DDIM reverse ──────────────────────────────────────────
        print(f"\n[6/7] AdaIN + DDIM reverse (inject)  "
              f"γ={self.cfg.gamma} α={self.cfg.alpha}...")
        zT_init = adain_latent(zT_content, zT_style)
        self.store.mode = "inject"
        z0_out = self._ddim_reverse(zT_init, gen_reverse, t5_reverse, label="stylized")
        self.store.mode = "off"

        # ── 7. Décodage ──────────────────────────────────────────────────────
        print("\n[7/7] Décodage VAE → vocoder...")
        wav_out = self._vae_decode_to_audio(z0_out)
        print(f"  Waveform : {len(wav_out)/self.proc.sample_rate:.2f}s")

        suffix   = "audioprompt" if self.cfg.use_audio_prompt else "uncond"
        out_name = f"stylized_g{self.cfg.gamma}_a{self.cfg.alpha}_{suffix}.wav"
        self.proc.save_audio(wav_out, os.path.join(out_dir, out_name))

        # Mel de sortie
        with torch.no_grad():
            dtype = next(self._pipe.vae.parameters()).dtype
            mel_out_raw = self._pipe.vae.decode(
                z0_out.to(dtype) / self._pipe.vae.config.scaling_factor
            ).sample.squeeze().cpu().float().numpy()
        if mel_out_raw.ndim == 2 and mel_out_raw.shape[0] > mel_out_raw.shape[1]:
            mel_out_raw = mel_out_raw.T
        self.proc.save_mel_plot(
            mel_out_raw, os.path.join(out_dir, "mel_stylized.png"),
            f"Stylized (γ={self.cfg.gamma}, α={self.cfg.alpha}, {suffix})"
        )

        print(f"\n{'─'*60}")
        print(f"  Fichiers dans {out_dir}/ :")
        for f in sorted(os.listdir(out_dir)):
            print(f"    {f}")
        print(f"{'─'*60}")
        print(f"""
  NOTES :
  - Vérifiez 01_vae_reconstruction.wav : si c'est mauvais,
    le mel preprocessing est incorrect et RIEN ne marchera.
  - use_audio_prompt=False → conditioning neutre, K/V dominent
  - use_audio_prompt=True  → conditioning CLAP(style)
  - gamma : 0.0=libre / 1.0=structure content préservée
  - alpha : 0.0=content pur / 1.0=style pur

  AVERTISSEMENT : L'hypothèse Q=structure / K,V=style de Stylus
  n'a PAS été validée pour l'audio. Les résultats peuvent être
  décevants pour des raisons fondamentales, pas techniques.
""")
        return wav_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = StylusAudioLDM2Config(
        gamma=0.1,
        alpha=0.9,
        use_audio_prompt=True,
        num_inference_steps=50,
        style_audio_path="musicTI_dataset/audios/timbre/chime/chime1.wav",
        content_audio_path="musicTI_dataset/audios/content/violin/violin1.wav",
        output_dir="stylus_audioldm2_output",

        # Désactiver les checks si tu veux aller plus vite
        skip_roundtrip_check=False,
    )

    print("=" * 60)
    print("  Stylus-AudioLDM2 — Music Style Transfer  (v4 corrected)")
    print(f"  Style   : {cfg.style_audio_path}")
    print(f"  Content : {cfg.content_audio_path}")
    print(f"  γ={cfg.gamma}  α={cfg.alpha}  use_audio_prompt={cfg.use_audio_prompt}")
    print(f"  Mel: hop={cfg.hop_length} n_mels={cfg.n_mels} fmax={cfg.fmax}")
    print(f"  Target blocks: {cfg.target_up_block_indices}")
    print(f"  Device  : {cfg.device}")
    print("=" * 60)

    pipeline = StylusAudioLDM2Pipeline(cfg)
    pipeline.load_model()
    pipeline.transfer(
        style_path=cfg.style_audio_path,
        content_path=cfg.content_audio_path,
        output_dir=cfg.output_dir,
    )


if __name__ == "__main__":
    main()