"""
MusicLDM — Style Transfer (audio2audio)
-----------------------------------------
Version musicale du style transfer, basée sur MusicLDM au lieu d'AudioLDM2.
 
Différences clés vs AudioLDM2 :
  - Encodeur unique : CLAP musical (vs CLAP + Flan-T5 + GPT2)
  - Pas de projection_model ni language_model
  - CLAP entraîné sur données musicales → meilleur pour la musique
  - encode_prompt retourne directement (prompt_embeds, attention_mask)
  - UNet à un seul cross-attention (vs deux dans AudioLDM2)
 
Architecture :
    Prompt texte → CLAP text encoder → prompt_embeds [1, 1, 512]
    Audio source → VAE encode → z0
    z0 + bruit(strength) → UNet(prompt_embeds) → z_styled
    z_styled → VAE decode → Mel → HiFi-GAN → audio
 
Conditioning progressif par chunks :
    chunk 0   : 100% prompt texte
    chunk k   : lerp(CLAP(audio_prev), prompt, alpha)  alpha croissant
    chunk N   : 100% prompt texte
 
Usage:
    python musicldm_style_transfer.py \
        --input     musique.wav \
        --prompt    "jazz piano, soft, nocturne" \
        --strength  0.5
 
Dépendances:
    pip install diffusers transformers accelerate soundfile librosa torch torchaudio
"""
 
import argparse
import os
import torch
import numpy as np
import soundfile as sf
import librosa
try:
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except OSError:
    HAS_TORCHAUDIO = False
    print("⚠ torchaudio non disponible — resample via librosa, Mel via numpy")
 
from diffusers import MusicLDMPipeline, DDIMScheduler
 
 
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_ID    = "ucsd-reach/musicldm"
SAMPLE_RATE = 16000
CLAP_SR     = 48000      # CLAP entraîné à 48kHz
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
 
CHUNK_SECONDS = 10
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
CHUNK_OVERLAP = int(SAMPLE_RATE * 0.1)   # 100ms
 
 
# ─────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────
def load_audio(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    wav = wav / (np.abs(wav).max() + 1e-8)
    return wav
 
 
def save_audio(path: str, wav, sr: int = SAMPLE_RATE):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float().numpy()
    if wav.ndim > 1:
        wav = wav.squeeze()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wav = wav / (np.abs(wav).max() + 1e-8)
    sf.write(path, wav, sr)
    print(f"  → {path}")
 
 
def split_chunks(wav: np.ndarray) -> list[np.ndarray]:
    chunks, start = [], 0
    while start < len(wav):
        chunk = wav[start : start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunks.append(chunk)
        start += CHUNK_SAMPLES - CHUNK_OVERLAP
    return chunks
 
 
def merge_chunks(chunks: list[np.ndarray], original_length: int) -> np.ndarray:
    if len(chunks) == 1:
        return chunks[0][:original_length]
    result = chunks[0].copy()
    for chunk in chunks[1:]:
        fade_out = np.linspace(1, 0, CHUNK_OVERLAP)
        fade_in  = np.linspace(0, 1, CHUNK_OVERLAP)
        result[-CHUNK_OVERLAP:] = result[-CHUNK_OVERLAP:] * fade_out + \
                                  chunk[:CHUNK_OVERLAP] * fade_in
        result = np.concatenate([result, chunk[CHUNK_OVERLAP:]])
    return result[:original_length]
 
 
# ─────────────────────────────────────────────
# Mel spectrogram (paramètres MusicLDM = AudioLDM)
# ─────────────────────────────────────────────
_MEL_TF = None
 
def get_mel_transform():
    global _MEL_TF
    if _MEL_TF is None and HAS_TORCHAUDIO:
        _MEL_TF = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, win_length=1024,
            hop_length=160, n_mels=64, f_min=0.0, f_max=8000.0,
            power=1.0, norm="slaney", mel_scale="slaney",
        )
    return _MEL_TF
 
 
def audio_to_mel(wav: np.ndarray, target_length: int = 1000) -> torch.Tensor:
    if HAS_TORCHAUDIO:
        wav_t = torch.FloatTensor(wav).unsqueeze(0)
        mel   = get_mel_transform()(wav_t)
        mel   = torch.log(torch.clamp(mel, min=1e-5))
        mel_np = mel.squeeze(0).numpy()   # [64, T]
    else:
        # Fallback librosa
        mel_np = librosa.feature.melspectrogram(
            y=wav, sr=SAMPLE_RATE, n_fft=1024, win_length=1024,
            hop_length=160, n_mels=64, fmin=0.0, fmax=8000.0,
            power=1.0, norm="slaney",
        )                                  # [64, T]
        mel_np = np.log(np.clip(mel_np, a_min=1e-5, a_max=None))
 
    t = mel_np.shape[-1]
    if t < target_length:
        mel_np = np.pad(mel_np, ((0,0),(0, target_length - t)))
    else:
        mel_np = mel_np[:, :target_length]
 
    mel_t = torch.FloatTensor(mel_np)      # [64, T]
    mel_t = (mel_t - mel_t.mean()) / (mel_t.std() + 1e-8)
    mel_t = mel_t.T.unsqueeze(0).unsqueeze(0)   # [1, 1, T, 64]
    return mel_t.to(DEVICE)
 
 
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
# Encoding prompt texte — MusicLDM (CLAP seul)
# ─────────────────────────────────────────────
def encode_prompt_text(prompt: str, pipe, guidance: bool) -> tuple:
    """
    Encode un prompt texte via CLAP text_projection → [B, 512].
    text_model (768d) + text_projection (768→512) = espace CLAP final.
    """
    dtype = next(pipe.text_encoder.parameters()).dtype
 
    def _encode(text):
        toks = pipe.tokenizer(
            text, padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out    = pipe.text_encoder.text_model(
                input_ids      = toks["input_ids"].to(DEVICE),
                attention_mask = toks["attention_mask"].to(DEVICE),
            )
            hidden = out.last_hidden_state[:, 0, :]              # [1, 768]
            return pipe.text_encoder.text_projection(hidden).to(dtype)  # [1, 512]
 
    cond = _encode(prompt)
    if guidance:
        uncond = _encode("")
        cond   = torch.cat([uncond, cond], dim=0)   # [2, 512]
    return cond, None
 
def encode_audio_as_prompt(wav: np.ndarray, pipe, guidance: bool) -> tuple:
    """
    Encode un chunk audio via CLAP musical → même espace que encode_prompt_text.
 
    MusicLDM utilise ClapModel — get_audio_features() retourne directement
    un embedding [1, 512] dans le même espace que get_text_features().
    Pas de projection ni de GPT2 : on peut mixer directement.
    """
    dtype = next(pipe.text_encoder.parameters()).dtype
 
    # Resample 16kHz → 48kHz pour CLAP (librosa si torchaudio indisponible)
    if HAS_TORCHAUDIO:
        wav_t   = torch.FloatTensor(wav).unsqueeze(0)
        wav_np  = torchaudio.functional.resample(wav_t, SAMPLE_RATE, CLAP_SR).squeeze(0).numpy()
    else:
        wav_np = librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=CLAP_SR)
 
    inputs = pipe.feature_extractor(
        wav_np,
        sampling_rate  = CLAP_SR,
        return_tensors = "pt",
    )
 
    with torch.no_grad():
        # CLAP audio embed → [1, 512]
        audio_embed = pipe.text_encoder.get_audio_features(
            **{k: v.to(DEVICE) for k, v in inputs.items()}
        ).to(dtype)                             # [1, 512]
 
        # MusicLDM UNet attend [batch, seq, dim] → unsqueeze seq dim
        audio_embed = audio_embed.unsqueeze(1)  # [1, 1, 512]
 
        # CFG : concat (uncond, cond)
        if guidance:
            toks_u   = pipe.tokenizer(
                "", padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            out_u  = pipe.text_encoder.text_model(
                input_ids      = toks_u["input_ids"].to(DEVICE),
                attention_mask = toks_u["attention_mask"].to(DEVICE),
            )
            hidden_u = out_u.last_hidden_state[:, 0, :]
            p_uncond = pipe.text_encoder.text_projection(hidden_u).to(dtype).unsqueeze(1)  # [1,1,512]
            audio_embed = torch.cat([p_uncond, audio_embed], dim=0)
 
        # attention_mask = None pour CLAP (pas de padding variable)
        attention_mask = None
 
    return audio_embed, attention_mask
 
 
def mix_embeddings(embed_prev: torch.Tensor,
                   embed_target: torch.Tensor,
                   alpha: float) -> torch.Tensor:
    """
    alpha=0 → 100% embed_prev   (son du chunk précédent)
    alpha=1 → 100% embed_target (prompt texte cible)
    """
    return (1 - alpha) * embed_prev + alpha * embed_target
 
 
# ─────────────────────────────────────────────
# DDIM Inversion — remonte x0 → x_T exactement
# ─────────────────────────────────────────────
def ddim_inversion(z0: torch.Tensor, pipe,
                   prompt_embeds: torch.Tensor,
                   guidance_scale: float,
                   n_steps: int,
                   strength: float) -> tuple[torch.Tensor, int]:
    """
    Vraie DDIM inversion : remonte de z0 vers z_T de façon déterministe.
    Utilise le même UNet que le débruitage mais en sens inverse.
 
    Contrairement à add_noise (aléatoire), l'inversion est exacte :
    le z_T obtenu, une fois redescendu avec le même prompt, redonne z0.
    Changer le prompt à la descente donne le style transfer.
 
    strength contrôle jusqu'où on remonte :
        1.0 → remonte jusqu'à T (perd tout le contenu)
        0.5 → remonte jusqu'à T/2 (compromis contenu/style)
        0.3 → remonte peu (proche de l'original)
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(n_steps, device=DEVICE)
 
    # Nombre de steps d'inversion
    t_start   = max(1, int(n_steps * strength))
    # On inverse sur les t_start premiers timesteps (du moins bruité au plus bruité)
    inversion_timesteps = scheduler.timesteps[n_steps - t_start:].flip(0)
 
    do_cfg = guidance_scale > 1.0
    dtype  = next(pipe.unet.parameters()).dtype
    z      = z0.clone().to(dtype)
 
    print(f"        DDIM inversion : strength={strength:.2f} "
          f"→ {t_start} steps remontés sur {n_steps}")
 
    for t in inversion_timesteps:
        z_input = torch.cat([z, z]) if do_cfg else z
 
        with torch.no_grad():
            p = prompt_embeds.to(dtype)
            batch_size = z_input.shape[0]
            if p.dim() == 1:
                p = p.unsqueeze(0).expand(batch_size, -1)
            elif p.dim() == 2 and p.shape[0] != batch_size:
                p = p.expand(batch_size, -1)
            elif p.dim() == 3:
                p = p.squeeze(1)
 
            noise_pred = pipe.unet(
                z_input.to(dtype),
                t,
                encoder_hidden_states = None,
                class_labels          = p,
            ).sample
 
        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
 
        # alphas_cumprod est sur CPU — on indexe sur CPU puis on déplace
        t_cpu             = t.cpu()
        alpha_prod_t      = scheduler.alphas_cumprod[t_cpu].to(DEVICE)
 
        # Trouver le prochain timestep dans la séquence d'inversion
        inv_cpu = inversion_timesteps.cpu()
        idx     = (inv_cpu == t_cpu).nonzero(as_tuple=True)[0]
        if idx < len(inv_cpu) - 1:
            alpha_prod_t_prev = scheduler.alphas_cumprod[inv_cpu[idx + 1]].to(DEVICE)
        else:
            alpha_prod_t_prev = torch.tensor(1.0, device=DEVICE)
 
        # Formule DDIM inverse :
        # z_{t+1} = √ᾱ_{t+1} · (z_t - √(1-ᾱ_t)·ε) / √ᾱ_t + √(1-ᾱ_{t+1})·ε
        x0_pred = (z - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
        z = alpha_prod_t_prev.sqrt() * x0_pred + (1 - alpha_prod_t_prev).sqrt() * noise_pred
 
    t_denoise_start = n_steps - t_start
    print(f"        Inversion terminée → débruitage depuis step {t_denoise_start}")
    return z, t_denoise_start
 
 
# ─────────────────────────────────────────────
# Débruitage guidé (MusicLDM — UNet simple)
# ─────────────────────────────────────────────
def get_class_labels(guidance_scale: float, pipe, do_cfg: bool) -> torch.Tensor:
    """
    MusicLDM passe guidance_scale comme class_label (pas la durée).
    class_embedding est une Linear(1, 512) → attend [batch, 1].
    """
    # batch_size = 2 si CFG (uncond + cond), 1 sinon
    batch_size = 2 if do_cfg else 1
    class_labels = torch.full(
        (batch_size, 1), fill_value=guidance_scale,
        device=DEVICE, dtype=torch.float32
    )
    return class_labels
 
 
def guided_denoise(z_noisy: torch.Tensor, t_start: int,
                   pipe, guidance_scale: float, n_steps: int,
                   prompt_embeds: torch.Tensor,
                   attention_mask,
                   ) -> torch.Tensor:
    """
    MusicLDM UNet : un seul cross-attention + class_labels pour la durée.
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(n_steps, device=DEVICE)
 
    do_cfg        = guidance_scale > 1.0
    dtype         = next(pipe.unet.parameters()).dtype
    z             = z_noisy.clone()
 
    for t in scheduler.timesteps[t_start:]:
        z_input = torch.cat([z, z]) if do_cfg else z
 
        with torch.no_grad():
            # prompt_embeds doit être [batch, 512] pour Linear(512, 512)
            batch_size = z_input.shape[0]
            p = prompt_embeds.to(dtype)
            # Gérer toutes les shapes possibles → [batch, 512]
            if p.dim() == 1:
                p = p.unsqueeze(0).expand(batch_size, -1)   # [512] → [batch, 512]
            elif p.dim() == 2 and p.shape[0] != batch_size:
                p = p.expand(batch_size, -1)                # [1, 512] → [batch, 512]
            elif p.dim() == 3:
                p = p.squeeze(1)                            # [batch, 1, 512] → [batch, 512]
            noise_pred = pipe.unet(
                z_input.to(dtype),
                t,
                encoder_hidden_states = None,
                class_labels          = p,
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
    print(f"  MusicLDM — Style Transfer")
    print(f"  Prompt   : \"{prompt}\"")
    print(f"  Strength : {strength}  |  Guidance : {guidance_scale}")
    print(f"  Device   : {DEVICE}")
    print(f"{'='*55}\n")
 
    os.makedirs(output_dir, exist_ok=True)
 
    # ── 1. Modèle ───────────────────────────────────────────
    print("[1/5] Chargement MusicLDM...")
    pipe = MusicLDMPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    vae = pipe.vae
    print("      Modèle chargé.")
 
    # ── 2. Audio source ─────────────────────────────────────
    print(f"\n[2/5] Chargement audio : {input_path}")
    wav_src   = load_audio(input_path)
    total_sec = len(wav_src) / SAMPLE_RATE
    print(f"      Durée : {total_sec:.1f}s")
    save_audio(os.path.join(output_dir, "00_original.wav"), wav_src)
 
    # ── 3. Test reconstruction VAE ──────────────────────────
    chunks_in = split_chunks(wav_src)
    print(f"\n[3/5] {total_sec:.1f}s → {len(chunks_in)} chunk(s) de {CHUNK_SECONDS}s")
    print(f"      Test reconstruction VAE (chunk 1)...")
    mel_test  = audio_to_mel(wav_src[:CHUNK_SAMPLES])
    z0_test   = vae_encode(mel_test, vae)
    wav_recon = vae_decode(z0_test, vae, pipe)
    save_audio(os.path.join(output_dir, "01_vae_reconstruction.wav"), wav_recon)
 
    # ── 4+5. Style transfer progressif chunk par chunk ──────
    print(f"\n[4+5/5] Style transfer progressif...")
    print(f"        Prompt : \"{prompt}\"")
 
    do_cfg = guidance_scale > 1.0
 
    # Encoder le prompt texte cible (une seule fois)
    print(f"        Encodage prompt texte...")
    p_embeds_text, p_mask_text = encode_prompt_text(prompt, pipe, do_cfg)
 
    # État : embeddings du chunk précédent
    prev_embeds = [None]
 
    def style_chunk(chunk: np.ndarray, idx: int, n: int) -> np.ndarray:
        # alpha croissant : 0 au début → 1 à la fin
        alpha = idx / max(n - 1, 1)
 
        if prev_embeds[0] is not None:
            p_prev, _ = prev_embeds[0]
            p_mixed   = mix_embeddings(p_prev, p_embeds_text, alpha)
            print(f"        alpha={alpha:.2f} — "
                  f"{(1-alpha)*100:.0f}% CLAP audio + {alpha*100:.0f}% prompt texte")
        else:
            p_mixed = p_embeds_text
            print(f"        alpha={alpha:.2f} — 100% prompt texte (chunk initial)")
 
        mel    = audio_to_mel(chunk)
        z0     = vae_encode(mel, vae)
        # DDIM inversion exacte (remonte z0 → z_t de façon déterministe)
        z_inv, t_start = ddim_inversion(
            z0, pipe, p_mixed, guidance_scale, n_steps, strength
        )
        # Débruitage guidé par le prompt cible (descend z_t → z0_styled)
        z_out  = guided_denoise(z_inv, t_start, pipe, guidance_scale, n_steps,
                                p_mixed, p_mask_text)
        wav_out = vae_decode(z_out, vae, pipe)
 
        # Encoder la sortie via CLAP audio pour le prochain chunk
        try:
            prev_embeds[0] = encode_audio_as_prompt(wav_out, pipe, do_cfg)
        except Exception as e:
            print(f"        ⚠ CLAP audio échoué ({e}), fallback texte")
            prev_embeds[0] = (p_embeds_text, p_mask_text)
 
        return wav_out
 
    n_chunks   = len(chunks_in)
    chunks_out = []
    print(f"        {n_chunks} chunk(s) à traiter...\n")
 
    for i, chunk in enumerate(chunks_in):
        print(f"\n      ── Chunk {i+1}/{n_chunks} ──")
        chunks_out.append(style_chunk(chunk, i, n_chunks))
 
    wav_styled = merge_chunks(chunks_out, len(wav_src))
    out_name   = f"styled_s{strength}_g{guidance_scale}.wav"
    save_audio(os.path.join(output_dir, out_name), wav_styled)
 
    print(f"\n{'─'*55}")
    print(f"  Fichiers dans {output_dir}/ :")
    print(f"    00_original.wav           → source ({total_sec:.1f}s)")
    print(f"    01_vae_reconstruction.wav → VAE seul (référence qualité)")
    print(f"    {out_name}")
    print(f"{'─'*55}")
    print(f"""
  Ajustements si le résultat ne convient pas :
    --strength   0.3-0.4 → reste proche de l'original
                 0.6-0.7 → style très marqué
    --guidance   3-5     → bon compromis
                 7+      → prompt très contraignant
""")
 
 
if __name__ == "__main__":
    music_path = "./musique/music1.wav"  # chemin par défaut
    style_prompt = "electric guitar, energetic"  # prompt par défaut
    output_dir = "./encode_decode"  # répertoire de sortie par défaut
    strength = 0.4 # force du style par défaut
    guidance = 5.0  # guidance scale par défaut
    steps = 50
    main(music_path, style_prompt, output_dir, strength, guidance, steps)