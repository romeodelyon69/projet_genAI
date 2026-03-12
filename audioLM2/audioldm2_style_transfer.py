"""
AudioLDM2 — Style Transfer (audio2audio)
-----------------------------------------
Principe : on encode l'audio source dans l'espace latent du VAE,
on ajoute du bruit jusqu'à un niveau `strength`, puis on débruite
en guidant avec un prompt texte décrivant le style cible.

C'est l'équivalent audio de img2img dans Stable Diffusion.

Paramètre clé — strength (0.0 → 1.0) :
    0.1  → très proche de l'original, style léger
    0.4  → bon compromis contenu/style
    0.7  → style très marqué, contenu altéré
    1.0  → génération complète (ignore l'audio source)

Usage:
    python audioldm2_style_transfer.py \
        --input      musique.wav \
        --prompt     "jazz piano, soft, nocturne" \
        --strength   0.5

Dépendances:
    pip install diffusers transformers accelerate soundfile librosa torch torchaudio
"""

import argparse
import os
import torch
import numpy as np
import soundfile as sf
import librosa
import torchaudio
import torchaudio.transforms as T

from diffusers import AudioLDM2Pipeline, DDIMScheduler


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_ID    = "cvssp/audioldm2-music"
SAMPLE_RATE = 16000
DDIM_STEPS  = 50
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────
CHUNK_SECONDS  = 10       # durée de chaque chunk traité par le VAE
CHUNK_SAMPLES  = SAMPLE_RATE * CHUNK_SECONDS
CHUNK_OVERLAP  = int(SAMPLE_RATE * 0.1)   # 100ms de overlap entre chunks


def load_audio(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    wav = wav / (np.abs(wav).max() + 1e-8)
    return wav


def split_chunks(wav: np.ndarray) -> list[np.ndarray]:
    """Découpe le signal en chunks de CHUNK_SECONDS avec overlap."""
    chunks = []
    start = 0
    while start < len(wav):
        end = min(start + CHUNK_SAMPLES, len(wav))
        chunk = wav[start:end]
        # Padding si le dernier chunk est trop court
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunks.append(chunk)
        start += CHUNK_SAMPLES - CHUNK_OVERLAP
    return chunks


def merge_chunks(chunks: list[np.ndarray], original_length: int) -> np.ndarray:
    """
    Recolle les chunks en appliquant un crossfade sur les zones d'overlap.
    """
    if len(chunks) == 1:
        return chunks[0][:original_length]

    result = chunks[0].copy()
    for i, chunk in enumerate(chunks[1:], 1):
        overlap = CHUNK_OVERLAP
        # Fenêtres de fondu
        fade_out = np.linspace(1, 0, overlap)
        fade_in  = np.linspace(0, 1, overlap)
        # Zone de recouvrement
        result[-overlap:] = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
        result = np.concatenate([result, chunk[overlap:]])

    return result[:original_length]


def save_audio(path: str, wav, sr: int = SAMPLE_RATE):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float().numpy()
    if wav.ndim > 1:
        wav = wav.squeeze()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wav = wav / (np.abs(wav).max() + 1e-8)
    sf.write(path, wav, sr)
    print(f"  → {path}")


# ─────────────────────────────────────────────
# Audio → Mel (paramètres exacts AudioLDM2)
# ─────────────────────────────────────────────
MEL_TRANSFORM = None  # initialisé une fois au premier appel

def get_mel_transform():
    global MEL_TRANSFORM
    if MEL_TRANSFORM is None:
        MEL_TRANSFORM = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, win_length=1024,
            hop_length=160, n_mels=64, f_min=0.0, f_max=8000.0,
            power=1.0, norm="slaney", mel_scale="slaney",
        )
    return MEL_TRANSFORM


def audio_to_mel(wav: np.ndarray, target_length: int = 1000) -> torch.Tensor:
    """
    Convertit un chunk audio (10s max) en Mel.
    Output shape : [1, 1, T, 64] avec T=target_length.
    """
    wav_t = torch.FloatTensor(wav).unsqueeze(0)
    mel = get_mel_transform()(wav_t)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    t = mel.shape[-1]
    if t < target_length:
        mel = torch.nn.functional.pad(mel, (0, target_length - t))
    else:
        mel = mel[..., :target_length]
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    mel = mel.permute(0, 2, 1).unsqueeze(1)    # [1, 1, T, 64]
    return mel.to(DEVICE)


def process_chunks(wav: np.ndarray, fn) -> np.ndarray:
    """
    Applique fn(chunk) → wav_chunk sur chaque chunk de l audio,
    puis recolle les morceaux avec crossfade.
    fn doit accepter un np.ndarray de CHUNK_SAMPLES samples
    et retourner un np.ndarray audio.
    """
    chunks_in  = split_chunks(wav)
    chunks_out = []
    print(f"      Traitement de {len(chunks_in)} chunk(s) de {CHUNK_SECONDS}s...")
    for i, chunk in enumerate(chunks_in):
        print(f"      Chunk {i+1}/{len(chunks_in)}", end="", flush=True)
        result = fn(chunk)
        chunks_out.append(result)
        print(f" → {len(result)/SAMPLE_RATE:.1f}s")
    return merge_chunks(chunks_out, len(wav))


# ─────────────────────────────────────────────
# VAE encode / decode
# ─────────────────────────────────────────────
def vae_encode(mel: torch.Tensor, vae) -> torch.Tensor:
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        z = vae.encode(mel.to(dtype)).latent_dist.mode()
        z = z * vae.config.scaling_factor
    return z


def vae_decode(z: torch.Tensor, vae, pipe) -> np.ndarray:
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        mel = vae.decode(z.to(dtype) / vae.config.scaling_factor).sample
    wav = pipe.mel_spectrogram_to_waveform(mel)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    return wav.squeeze()


# ─────────────────────────────────────────────
# Encoding CLAP d'un chunk audio
# ─────────────────────────────────────────────
CLAP_SR = 48000   # CLAP d'AudioLDM2 entraîné à 48kHz


def clap_encode_audio_projected(wav: np.ndarray, pipe) -> torch.Tensor:
    """
    Encode un chunk audio via CLAP puis le projette dans l'espace GPT2
    via la ProjectionModel d'AudioLDM2.

    Chaîne correcte :
        wav → CLAP audio encoder → E1_audio (512d)
            → projection_model  → P1_audio (même espace que P1_texte)

    C'est P1_audio qu'on peut mixer avec P1_texte (issu du prompt),
    puis passer à GPT2 pour générer des generated_prompt_embeds cohérents.
    """
    dtype = next(pipe.text_encoder.parameters()).dtype

    # 1. Resample 16kHz → 48kHz pour CLAP
    wav_t   = torch.FloatTensor(wav).unsqueeze(0)
    wav_48k = torchaudio.functional.resample(wav_t, SAMPLE_RATE, CLAP_SR)
    wav_np  = wav_48k.squeeze(0).numpy()

    inputs = pipe.feature_extractor(
        wav_np,
        sampling_rate  = CLAP_SR,
        return_tensors = "pt",
    )

    with torch.no_grad():
        # CLAP audio features → [1, 512]
        audio_feat = pipe.text_encoder.get_audio_features(
            **{k: v.to(DEVICE) for k, v in inputs.items()}
        ).to(dtype)

        # Projection dans l'espace commun CLAP/T5 → même dim que P1 texte
        # projection_model attend (clap_embeds, t5_embeds)
        # On passe None pour T5 et on récupère seulement la projection CLAP
        projected = pipe.projection_model(
            hidden_states   = audio_feat.unsqueeze(1),   # [1, 1, 512]
            hidden_states_1 = None,
        ).hidden_states                                   # [1, 1, proj_dim]

    return projected   # [1, 1, proj_dim]  — même espace que P1 texte


def encode_prompt_text(prompt: str, pipe, guidance: bool) -> tuple:
    """Encode un prompt texte → (prompt_embeds, attention_mask, generated_prompt_embeds)."""
    with torch.no_grad():
        embeds = pipe.encode_prompt(
            prompt=prompt,
            device=DEVICE,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=guidance,
        )
    return embeds   # (prompt_embeds, attention_mask, generated_prompt_embeds)


def encode_audio_as_prompt(wav: np.ndarray, pipe, guidance: bool) -> tuple:
    """
    Encode un chunk audio → (prompt_embeds, attention_mask, generated_prompt_embeds)
    dans le même espace que encode_prompt_text.

    Chaîne manuelle (reproduit ce que encode_prompt fait en interne) :
        wav
        → CLAP feature extractor (48kHz)
        → text_encoder.get_audio_features()  [1, 512]
        → projection_model (hidden_states)   [1, 1, proj_dim]
        → GPT2 language_model                [1, 8, 768]   ← generated_prompt_embeds

    Pour prompt_embeds (T5) on utilise un prompt vide — on veut que le
    conditioning vienne du CLAP audio, pas du texte.
    """
    dtype = next(pipe.text_encoder.parameters()).dtype

    # 1. Resample 16kHz → 48kHz
    wav_t   = torch.FloatTensor(wav).unsqueeze(0)
    wav_48k = torchaudio.functional.resample(wav_t, SAMPLE_RATE, CLAP_SR)
    wav_np  = wav_48k.squeeze(0).numpy()

    inputs = pipe.feature_extractor(
        wav_np,
        sampling_rate  = CLAP_SR,
        return_tensors = "pt",
    )

    with torch.no_grad():
        # 2. CLAP audio features → [1, 512]
        audio_feat = pipe.text_encoder.get_audio_features(
            **{k: v.to(DEVICE) for k, v in inputs.items()}
        ).to(dtype)                            # [1, 512]

        # 3. Projection CLAP → espace commun
        proj_model = pipe.projection_model

        # Lire la dim T5 depuis le config FrozenDict
        t5_dim = proj_model.config.get("cross_attention_dim_t5", None) or                  proj_model.config.get("hidden_size_t5", None) or 1024
        t5_dummy = torch.zeros(1, 1, t5_dim, dtype=dtype, device=DEVICE)

        proj_out       = proj_model(
            hidden_states   = audio_feat.unsqueeze(1),  # [1, 1, 512]
            hidden_states_1 = t5_dummy,                 # [1, 1, t5_dim]
        )
        clap_projected = proj_out.hidden_states         # [1, 1, proj_dim]

        # 4. GPT2 → generated_prompt_embeds
        gpt2_out = pipe.language_model(
            inputs_embeds = clap_projected,
        ).last_hidden_state
        generated_prompt_embeds = gpt2_out.to(dtype)   # [1, seq_audio, 768]

        # 5. T5 sur prompt vide + version unconditionnelle pour avoir la bonne taille cible
        p_embeds, p_mask, gpe_uncond = pipe.encode_prompt(
            prompt                      = "",
            device                      = DEVICE,
            num_waveforms_per_prompt    = 1,
            do_classifier_free_guidance = False,
        )
        target_seq_len = gpe_uncond.shape[1]   # longueur attendue par le UNet (ex: 8)

        # Aligner generated_prompt_embeds sur la longueur cible
        curr_len = generated_prompt_embeds.shape[1]
        if curr_len < target_seq_len:
            # Padding avec des zéros
            pad = torch.zeros(
                1, target_seq_len - curr_len, generated_prompt_embeds.shape[-1],
                dtype=dtype, device=DEVICE
            )
            generated_prompt_embeds = torch.cat([generated_prompt_embeds, pad], dim=1)
        elif curr_len > target_seq_len:
            generated_prompt_embeds = generated_prompt_embeds[:, :target_seq_len, :]

        # 6. CFG : concat uncond + cond
        if guidance:
            generated_prompt_embeds = torch.cat([gpe_uncond, generated_prompt_embeds])

    return p_embeds, p_mask, generated_prompt_embeds


def mix_embeddings(embed_prev: torch.Tensor,
                   embed_target: torch.Tensor,
                   alpha: float) -> torch.Tensor:
    """
    Mélange linéaire entre l'embedding du chunk précédent et le prompt cible.
    alpha=0 → 100% embed_prev  (début de transition)
    alpha=1 → 100% embed_target (fin de transition)
    """
    return (1 - alpha) * embed_prev + alpha * embed_target


# ─────────────────────────────────────────────
# Ajout de bruit contrôlé (forward diffusion)
# ─────────────────────────────────────────────
def add_noise(z0: torch.Tensor, scheduler, strength: float,
              n_steps: int) -> tuple[torch.Tensor, int]:
    """
    Ajoute du bruit gaussien jusqu'au timestep t = strength * n_steps.
    Retourne (z_noisy, t_start) où t_start est le step de départ
    pour le débruitage.
    """
    scheduler.set_timesteps(n_steps, device=DEVICE)

    # Timestep de départ = on ne remonte pas jusqu'au bruit pur
    t_start  = max(1, int(n_steps * strength))
    timestep = scheduler.timesteps[n_steps - t_start]

    noise = torch.randn_like(z0)
    z_noisy = scheduler.add_noise(
        z0.float(), noise.float(), timestep.unsqueeze(0)
    ).to(z0.dtype)

    print(f"      Strength={strength:.2f} → timestep={timestep.item()}, "
          f"bruit ajouté sur {t_start}/{n_steps} steps")
    return z_noisy, n_steps - t_start


# ─────────────────────────────────────────────
# Débruitage guidé par prompt (style transfer)
# ─────────────────────────────────────────────
def guided_denoise(z_noisy: torch.Tensor, t_start: int,
                   pipe, guidance_scale: float, n_steps: int,
                   prompt_embeds: torch.Tensor,
                   attention_mask: torch.Tensor,
                   generated_prompt_embeds: torch.Tensor) -> torch.Tensor:
    """
    Débruite z_noisy guidé par des embeddings pré-calculés.
    Accepte des embeddings mixés (CLAP précédent + prompt cible).
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(n_steps, device=DEVICE)

    do_cfg = guidance_scale > 1.0
    dtype  = next(pipe.unet.parameters()).dtype
    z      = z_noisy.clone()

    active_timesteps = scheduler.timesteps[t_start:]

    for t in active_timesteps:
        z_input = torch.cat([z, z]) if do_cfg else z

        with torch.no_grad():
            noise_pred = pipe.unet(
                z_input.to(dtype),
                t,
                encoder_hidden_states    = generated_prompt_embeds.to(dtype),
                encoder_hidden_states_1  = prompt_embeds.to(dtype),
                encoder_attention_mask_1 = attention_mask,
            ).sample

        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        z = scheduler.step(noise_pred, t, z).prev_sample

    return z


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(input_path: str, prompt: str, output_dir: str,
         strength: float, guidance_scale: float, n_steps: int):

    print(f"\n{'='*55}")
    print(f"  AudioLDM2 — Style Transfer")
    print(f"  Prompt   : \"{prompt}\"")
    print(f"  Strength : {strength}  |  Guidance : {guidance_scale}")
    print(f"  Device   : {DEVICE}")
    print(f"{'='*55}\n")

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Modèle ───────────────────────────────────────────
    print("[1/5] Chargement du modèle...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    vae = pipe.vae

    # ── 2. Audio source ─────────────────────────────────────
    print(f"\n[2/5] Chargement audio : {input_path}")
    wav_src = load_audio(input_path)
    print(f"      Durée : {len(wav_src)/SAMPLE_RATE:.1f}s")
    save_audio(os.path.join(output_dir, "00_original.wav"), wav_src)

    # ── 3. Test VAE sur premier chunk ──────────────────────
    total_sec = len(wav_src) / SAMPLE_RATE
    chunks_in = split_chunks(wav_src)
    print(f"\n[3/5] Audio : {total_sec:.1f}s → {len(chunks_in)} chunk(s) de {CHUNK_SECONDS}s")
    print(f"      Test reconstruction VAE (chunk 1)...")
    mel_test  = audio_to_mel(wav_src[:CHUNK_SAMPLES])
    z0_test   = vae_encode(mel_test, vae)
    wav_recon = vae_decode(z0_test, vae, pipe)
    save_audio(os.path.join(output_dir, "01_vae_reconstruction.wav"), wav_recon)

    # ── 4+5. Style transfer chunk par chunk ─────────────────
    print(f"\n[4+5/5] Style transfer chunk par chunk...")
    print(f"        Prompt   : \"{prompt}\"")
    print(f"        Strength : {strength}  |  Guidance : {guidance_scale}")

    # Pré-encoder le prompt texte cible (fait une seule fois)
    do_cfg = guidance_scale > 1.0
    print(f"      Encodage du prompt cible...")
    p_embeds_text, p_mask_text, gp_embeds_text = encode_prompt_text(prompt, pipe, do_cfg)

    # État partagé : embeddings issus du chunk précédent (audio → encode_audio_as_prompt)
    prev_embeds = [None]   # (p_embeds, p_mask, gp_embeds) du chunk précédent

    def style_chunk_progressive(chunk: np.ndarray, chunk_idx: int,
                                 n_chunks: int) -> np.ndarray:
        """
        Style transfer sur un chunk avec conditioning constant.

        Mixing au niveau de generated_prompt_embeds (espace GPT2, 768d) :
        - chunk 0 : prompt texte seul → alpha=0
        - chunk k : lerp(gpe_audio_prev, gpe_text, alpha)  alpha croissant
        - chunk N : prompt texte seul → alpha=1

        Les deux embeddings sont dans le MÊME espace (sortie GPT2)
        donc le mix est géométriquement valide.
        """

        alpha = 0.8

        if prev_embeds[0] is not None:
            p_e_prev, p_m_prev, gp_e_prev = prev_embeds[0]
            # Mix des generated_prompt_embeds (GPT2 output, même shape)
            gp_mixed = mix_embeddings(gp_e_prev, gp_embeds_text, alpha)
            # Pour p_embeds (T5) on garde le texte — T5 encode le contenu linguistique
            p_embeds = p_embeds_text
            p_mask   = p_mask_text
            print(f"        alpha={alpha:.2f} — "
                  f"{(1-alpha)*100:.0f}% audio précédent + {alpha*100:.0f}% prompt texte")
        else:
            gp_mixed = gp_embeds_text
            p_embeds = p_embeds_text
            p_mask   = p_mask_text
            print(f"        alpha={alpha:.2f} — 100% prompt texte (chunk initial)")

        mel    = audio_to_mel(chunk)
        z0     = vae_encode(mel, vae)
        z_n, t = add_noise(z0, pipe.scheduler, strength, n_steps)
        z_out  = guided_denoise(z_n, t, pipe, guidance_scale, n_steps,
                                p_embeds, p_mask, gp_mixed)
        wav_out = vae_decode(z_out, vae, pipe)

        # Encoder le chunk de sortie comme "prompt audio" pour le chunk suivant
        try:
            prev_embeds[0] = encode_audio_as_prompt(wav_out, pipe, do_cfg)
        except Exception as e:
            # Fallback si audio_prompt_embeds n'est pas supporté
            print(f"        ⚠ encode_audio_as_prompt échoué ({e}), fallback texte")
            prev_embeds[0] = (p_embeds_text, p_mask_text, gp_embeds_text)

        return wav_out

    # Traitement chunk par chunk avec progression
    chunks_in  = split_chunks(wav_src)
    n_chunks   = len(chunks_in)
    chunks_out = []
    print(f"      {n_chunks} chunk(s) à traiter...")

    for i, chunk in enumerate(chunks_in):
        print(f"\n      ── Chunk {i+1}/{n_chunks} ──")
        out = style_chunk_progressive(chunk, i, n_chunks)
        chunks_out.append(out)

    wav_styled = merge_chunks(chunks_out, len(wav_src))
    out_name   = f"styled_s{strength}_g{guidance_scale}.wav"
    save_audio(os.path.join(output_dir, out_name), wav_styled)

    print(f"\n{'─'*55}")
    print(f"  Fichiers générés dans {output_dir}/ :")
    print(f"    00_original.wav           → audio source ({total_sec:.1f}s)")
    print(f"    01_vae_reconstruction.wav → VAE seul, chunk 1 (référence qualité)")
    print(f"    {out_name}")
    print(f"{'─'*55}")
    print(f"""
  Si le résultat ne convient pas, ajuster :
    --strength   plus bas  → reste plus proche de l'original
                 plus haut → style plus marqué, moins fidèle
    --guidance   plus haut → prompt plus contraignant (3-7 recommandé)
""")




if __name__ == "__main__":
    music_path = "./musique/music4.wav"  # chemin par défaut
    style_prompt = "jazz piano, drums, soft, nocturne"  # prompt par défaut
    output_dir = "./encode_decode"  # répertoire de sortie par défaut
    strength = 0.5 # force du style par défaut
    guidance = 7.0  # guidance scale par défaut
    steps = 50
    main(music_path, style_prompt, output_dir, strength, guidance, steps)