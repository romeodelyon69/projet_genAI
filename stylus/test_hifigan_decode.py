"""
test_hifigan_decode.py

Teste la chaîne encode → mel → HiFi-GAN → audio en isolation,
avec les paramètres mel de Stylus ET avec les paramètres natifs HiFi-GAN,
pour diagnostiquer la source de dégradation audio.

Trois reconstructions comparées :
  (A) HiFi-GAN natif   : mel 80 bandes, n_fft=1024, hop=256  (ce pour quoi HiFi-GAN a été entraîné)
  (B) HiFi-GAN Stylus  : mel 512 bandes, n_fft=2048, hop=512 → resample vers 80 bandes avant HiFi-GAN
  (C) Griffin-Lim      : baseline de référence depuis le mel Stylus

Produit :
  ./hifigan_test/audio_original.wav
  ./hifigan_test/audio_A_hifigan_native.wav
  ./hifigan_test/audio_B_hifigan_stylus_resampled.wav
  ./hifigan_test/audio_C_griffinlim.wav
  ./hifigan_test/mel_comparison.png
  ./hifigan_test/snr_report.txt

Usage :
  python test_hifigan_decode.py --input relax.wav
  python test_hifigan_decode.py --input relax.wav --duration 5.0
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "./hifigan_test"


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres mel Stylus (depuis StylusConfig)
# ─────────────────────────────────────────────────────────────────────────────
STYLUS_SR         = 22050
STYLUS_N_FFT      = 2048
STYLUS_HOP        = 512
STYLUS_N_MELS     = 512
STYLUS_FMIN       = 0.0
STYLUS_FMAX       = 8000.0

# Paramètres mel natifs HiFi-GAN (LJ Speech / VCTK checkpoint)
HIFIGAN_SR        = 22050
HIFIGAN_N_FFT     = 1024
HIFIGAN_HOP       = 256
HIFIGAN_WIN       = 1024
HIFIGAN_N_MELS    = 80
HIFIGAN_FMIN      = 0.0
HIFIGAN_FMAX      = 8000.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers audio
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path, sr, duration=None):
    audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return audio.astype(np.float32)


def save_audio(path, audio, sr):
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    sf.write(path, audio, sr)
    print(f"  Saved: {path}")


def audio_metrics(ref, est, sr=22050):
    """
    Métriques perceptuelles adaptées à la reconstruction audio.

    Le SNR sample-à-sample est inutile pour un vocoder neural :
    HiFi-GAN produit un signal déphasé, l'échelle est arbitraire,
    et une légère différence de longueur casse tout.

    On utilise à la place :
      - Corrélation spectrale (mel) : compare les enveloppes fréquentielles
      - Log-spectral distance : distance en dB entre les mels moyennés
      - Energie RMS relative : vérifie que l'amplitude est cohérente
    """
    T = min(len(ref), len(est))
    ref, est = ref[:T] / (np.abs(ref[:T]).max() + 1e-8), \
               est[:T] / (np.abs(est[:T]).max() + 1e-8)

    # Mel des deux signaux (80 bandes, paramètres standard)
    def mel(x):
        m = librosa.feature.melspectrogram(
            y=x, sr=sr, n_fft=1024, hop_length=256, n_mels=80,
            fmin=0.0, fmax=8000.0,
        )
        return librosa.power_to_db(m + 1e-10)

    mel_ref = mel(ref)
    mel_est = mel(est)

    # 1. Corrélation de Pearson sur les mels aplatis
    corr = float(np.corrcoef(mel_ref.flatten(), mel_est.flatten())[0, 1])

    # 2. Log-spectral distance (LSD) : distance RMS entre mels moyennés en temps
    env_ref = mel_ref.mean(axis=1)  # (80,)
    env_est = mel_est.mean(axis=1)
    lsd = float(np.sqrt(np.mean((env_ref - env_est) ** 2)))

    # 3. Energie RMS relative (en dB)
    rms_ref = np.sqrt(np.mean(ref ** 2))
    rms_est = np.sqrt(np.mean(est ** 2))
    rms_diff = float(20 * np.log10((rms_est + 1e-8) / (rms_ref + 1e-8)))

    return {"mel_corr": corr, "lsd_db": lsd, "rms_diff_db": rms_diff}


# ─────────────────────────────────────────────────────────────────────────────
# Mel encode avec paramètres Stylus
# ─────────────────────────────────────────────────────────────────────────────

def encode_stylus_mel(audio):
    """audio → (mel_db, stft) avec les paramètres exacts de Stylus."""
    stft = librosa.stft(audio, n_fft=STYLUS_N_FFT, hop_length=STYLUS_HOP,
                        window='hann', center=True)
    mag = np.abs(stft)
    mel_fb = librosa.filters.mel(sr=STYLUS_SR, n_fft=STYLUS_N_FFT,
                                  n_mels=STYLUS_N_MELS,
                                  fmin=STYLUS_FMIN, fmax=STYLUS_FMAX)
    mel_power = mel_fb @ (mag ** 2)
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    return mel_db, stft   # mel_db : (512, T)


# ─────────────────────────────────────────────────────────────────────────────
# Mel encode avec paramètres HiFi-GAN natifs
# ─────────────────────────────────────────────────────────────────────────────

def encode_hifigan_mel(audio):
    """audio → mel_db avec les paramètres natifs HiFi-GAN."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=HIFIGAN_SR,
        n_fft=HIFIGAN_N_FFT, hop_length=HIFIGAN_HOP, win_length=HIFIGAN_WIN,
        n_mels=HIFIGAN_N_MELS, fmin=HIFIGAN_FMIN, fmax=HIFIGAN_FMAX,
        window='hann', center=True, pad_mode='reflect',
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db   # (80, T)


# ─────────────────────────────────────────────────────────────────────────────
# Resample mel Stylus (512 bandes) → mel HiFi-GAN (80 bandes)
# ─────────────────────────────────────────────────────────────────────────────

def resample_stylus_mel_to_hifigan(mel_db_stylus):
    """
    Convertit le mel Stylus (512 bandes, hop=512) vers le format HiFi-GAN
    (80 bandes, hop=256).

    Étapes :
      1. Mel dB Stylus → amplitude linéaire
      2. Reproject via filterbanks : 512 → magntiude STFT → 80 bandes
      3. Ajuster la résolution temporelle (hop 512→256 = ×2)
    """
    import torch, torch.nn.functional as F

    # 1. Amplitude mel Stylus
    mel_amp_stylus = librosa.db_to_amplitude(mel_db_stylus)  # (512, T_stylus)

    # 2. Inverser filterbank Stylus → magnitude STFT approx
    fb_stylus = librosa.filters.mel(sr=STYLUS_SR, n_fft=STYLUS_N_FFT,
                                     n_mels=STYLUS_N_MELS,
                                     fmin=STYLUS_FMIN, fmax=STYLUS_FMAX)
    # Transposée pondérée (plus stable que pinv)
    fb_norm = fb_stylus / (fb_stylus.sum(axis=0, keepdims=True) + 1e-10)
    mag_stft = fb_norm.T @ mel_amp_stylus  # (n_fft//2+1, T_stylus)

    # 3. Appliquer filterbank HiFi-GAN (80 bandes, n_fft=1024)
    fb_hifi = librosa.filters.mel(sr=HIFIGAN_SR, n_fft=HIFIGAN_N_FFT,
                                   n_mels=HIFIGAN_N_MELS,
                                   fmin=HIFIGAN_FMIN, fmax=HIFIGAN_FMAX)
    # fb_hifi : (80, 513), mag_stft : (1025, T_stylus)
    # On doit aligner les bins fréquentiels (1025 vs 513)
    n_bins_hifi  = HIFIGAN_N_FFT // 2 + 1   # 513
    n_bins_stylus = STYLUS_N_FFT // 2 + 1    # 1025
    # Réduire mag_stft à 513 bins par interpolation fréquentielle
    t = torch.from_numpy(mag_stft).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(n_bins_hifi, mag_stft.shape[1]),
                      mode='bilinear', align_corners=False)
    mag_stft_hifi = t.squeeze().numpy()  # (513, T_stylus)

    mel_hifi_power = fb_hifi @ (mag_stft_hifi ** 2)  # (80, T_stylus)

    # 4. Ajuster résolution temporelle : hop 512→256 = doubler T
    T_target = mel_hifi_power.shape[1] * 2
    t = torch.from_numpy(mel_hifi_power).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(HIFIGAN_N_MELS, T_target),
                      mode='bilinear', align_corners=False)
    mel_hifi_power = t.squeeze().numpy()  # (80, T_target)

    mel_hifi_db = librosa.power_to_db(mel_hifi_power, ref=np.max)
    return mel_hifi_db  # (80, T_hifigan)


# ─────────────────────────────────────────────────────────────────────────────
# HiFi-GAN decode
# ─────────────────────────────────────────────────────────────────────────────

def load_hifigan(device='cpu'):
    """
    Charge HiFi-GAN. Trois tentatives dans l'ordre :
      1. NVIDIA torch.hub  — retourne un tuple (model, denoiser), on prend [0]
      2. SpeechBrain       — API inference
      3. Hugging Face      — microsoft/speecht5_hifigan (fallback universel)
    """
    print("  Chargement HiFi-GAN...")

    # ── Option 1 : NVIDIA torch.hub ───────────────────────────────────────────
    try:
        result = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_hifigan',
            model_math='fp32',
            pretrained=True,
            force_reload=False,
            verbose=False,
        )
        # nvidia_hifigan retourne (generator, denoiser) — on veut juste le generator
        if isinstance(result, (tuple, list)):
            vocoder = result[0]
        else:
            vocoder = result
        vocoder = vocoder.to(device).eval()
        print("  HiFi-GAN charge via torch.hub (NVIDIA)")
        return vocoder, 'nvidia'
    except Exception as e1:
        print(f"  torch.hub NVIDIA echoue: {e1}")

    # ── Option 2 : SpeechBrain (patch torchaudio si nécessaire) ───────────────
    try:
        import torchaudio
        # Patch pour les versions récentes de torchaudio qui ont supprimé list_audio_backends
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: []
        from speechbrain.inference.vocoders import HIFIGAN
        vocoder = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir="./hifigan_sb",
            run_opts={"device": device},
        )
        print("  HiFi-GAN charge via SpeechBrain")
        return vocoder, 'speechbrain'
    except Exception as e2:
        print(f"  SpeechBrain echoue: {e2}")

    # ── Option 3 : Hugging Face transformers (SpeechT5 HiFi-GAN) ─────────────
    try:
        from transformers import SpeechT5HifiGan
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        vocoder = vocoder.to(device).eval()
        print("  HiFi-GAN charge via HuggingFace (SpeechT5HifiGan)")
        return vocoder, 'hf_speecht5'
    except Exception as e3:
        print(f"  HuggingFace SpeechT5 echoue: {e3}")

    return None, None


def encode_mel_for_backend(audio, backend, device='cpu'):
    """
    Encode l'audio avec le pipeline mel NATIF de chaque backend.
    On ne passe plus par notre mel Stylus — chaque vocoder a son propre
    encodeur avec ses propres paramètres de normalisation.
    """
    if backend in ('nvidia', 'speechbrain'):
        # HiFi-GAN standard : n_mels=80, sr=22050, n_fft=1024, hop=256
        # Normalisation : log naturel (pas dB librosa)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=22050, n_fft=1024, hop_length=256,
            win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0,
            window='hann', center=True,
        )
        mel_log = np.log(np.maximum(mel, 1e-5))  # log naturel
        return torch.from_numpy(mel_log).float().unsqueeze(0).to(device)  # (1, 80, T)

    elif backend == 'hf_speecht5':
        # SpeechT5HifiGan est entraîné avec le SpeechT5FeatureExtractor
        # sr=16000, n_mels=80, normalisation mean/var spécifique
        from transformers import SpeechT5FeatureExtractor
        fe = SpeechT5FeatureExtractor.from_pretrained("microsoft/speecht5_tts")
        audio_16k = librosa.resample(audio, orig_sr=HIFIGAN_SR, target_sr=16000)
        # return_tensors="pt" → dict avec "input_values" (mel features)
        feats = fe(audio=audio_16k, sampling_rate=16000,
                   return_tensors="pt", padding=False)
        # feats contient le spectrogramme log-mel normalisé (1, T, 80)
        mel = feats["input_values"].to(device)
        return mel

    raise ValueError(f"Backend inconnu: {backend}")


def hifigan_decode(vocoder, backend, audio_raw, device='cpu'):
    """
    audio_raw : waveform numpy original (on réencode avec le pipeline natif)
    Retourne  : audio numpy float32 reconstruit par le vocoder
    """
    mel = encode_mel_for_backend(audio_raw, backend, device)

    with torch.no_grad():
        if backend == 'nvidia':
            out = vocoder(mel)              # (1, 1, T) ou (1, T)
            audio = out.squeeze().cpu().numpy()

        elif backend == 'speechbrain':
            out = vocoder.decode_batch(mel)
            audio = out.squeeze().cpu().numpy()

        elif backend == 'hf_speecht5':
            # SpeechT5HifiGan attend (batch, T, n_mels) — deja dans ce format
            out = vocoder(mel)
            audio = out.squeeze().cpu().numpy()

        else:
            raise ValueError(f"Backend inconnu: {backend}")

    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Griffin-Lim depuis mel Stylus
# ─────────────────────────────────────────────────────────────────────────────

def griffinlim_from_stylus_mel(mel_db, stft_content=None):
    """
    Reconstruction Griffin-Lim depuis le mel Stylus.
    Si stft_content fourni : initialisation avec la phase content.
    """
    mel_amp = librosa.db_to_amplitude(mel_db)

    # Mel → magnitude STFT via transposée pondérée
    fb = librosa.filters.mel(sr=STYLUS_SR, n_fft=STYLUS_N_FFT,
                               n_mels=STYLUS_N_MELS,
                               fmin=STYLUS_FMIN, fmax=STYLUS_FMAX)
    fb_norm = fb / (fb.sum(axis=0, keepdims=True) + 1e-10)
    mag_stft = np.maximum(fb_norm.T @ mel_amp, 0.0)  # (1025, T)

    if stft_content is not None:
        # Initialiser avec la phase du content
        T = min(mag_stft.shape[1], stft_content.shape[1])
        phase_init = np.angle(stft_content[:, :T])
        mag_t = mag_stft[:, :T]
        stft_init = mag_t * np.exp(1j * phase_init)
        audio_tmp = librosa.istft(stft_init, n_fft=STYLUS_N_FFT,
                                   hop_length=STYLUS_HOP, window='hann', center=True)
        # Quelques itérations GL
        for _ in range(8):
            s = librosa.stft(audio_tmp, n_fft=STYLUS_N_FFT,
                              hop_length=STYLUS_HOP, window='hann', center=True)
            s = mag_t * np.exp(1j * np.angle(s))
            audio_tmp = librosa.istft(s, n_fft=STYLUS_N_FFT,
                                       hop_length=STYLUS_HOP, window='hann', center=True)
    else:
        audio_tmp = librosa.griffinlim(
            mag_stft, n_iter=32, hop_length=STYLUS_HOP,
            n_fft=STYLUS_N_FFT, window='hann', center=True,
        )

    return audio_tmp.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_mel_comparison(mel_native, mel_stylus, mel_resampled, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    panels = [
        (mel_native,     f"Mel HiFi-GAN natif\n({HIFIGAN_N_MELS} bandes, hop={HIFIGAN_HOP})"),
        (mel_stylus,     f"Mel Stylus\n({STYLUS_N_MELS} bandes, hop={STYLUS_HOP})"),
        (mel_resampled,  f"Mel Stylus → resamplé HiFi-GAN\n({HIFIGAN_N_MELS} bandes, hop={HIFIGAN_HOP})"),
    ]
    hops = [HIFIGAN_HOP, STYLUS_HOP, HIFIGAN_HOP]

    for ax, (mel, title), hop in zip(axes, panels, hops):
        dur = mel.shape[1] * hop / STYLUS_SR
        im = ax.imshow(mel, origin='lower', aspect='auto', cmap='magma',
                       extent=[0, dur, 0, STYLUS_FMAX], interpolation='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Temps (s)"); ax.set_ylabel("Fréquence (Hz)")
        fig.colorbar(im, ax=ax, format="%+2.0f dB", shrink=0.8)

    plt.suptitle("Comparaison des représentations mel", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',    default='relax.wav',  help='Fichier audio source')
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = args.device

    print(f"Input  : {args.input}")
    print(f"Device : {device}")

    # ── Chargement ────────────────────────────────────────────────────────────
    print("\n[1] Chargement audio...")
    audio = load_audio(args.input, STYLUS_SR, args.duration)
    save_audio(os.path.join(OUT_DIR, "audio_original.wav"), audio, STYLUS_SR)

    # ── Encode mel ────────────────────────────────────────────────────────────
    print("\n[2] Encodage mel...")
    mel_stylus, stft_content = encode_stylus_mel(audio)
    mel_native               = encode_hifigan_mel(audio)
    mel_resampled            = resample_stylus_mel_to_hifigan(mel_stylus)

    print(f"  mel_stylus   : {mel_stylus.shape}  ({mel_stylus.min():.1f} → {mel_stylus.max():.1f} dB)")
    print(f"  mel_native   : {mel_native.shape}   ({mel_native.min():.1f} → {mel_native.max():.1f} dB)")
    print(f"  mel_resampled: {mel_resampled.shape}  ({mel_resampled.min():.1f} → {mel_resampled.max():.1f} dB)")

    save_mel_comparison(mel_native, mel_stylus, mel_resampled,
                        os.path.join(OUT_DIR, "mel_comparison.png"))

    # ── HiFi-GAN ──────────────────────────────────────────────────────────────
    print("\n[3] HiFi-GAN...")
    vocoder, backend = load_hifigan(device)

    results = {}

    def fmt(m):
        if m is None: return "N/A"
        return (f"mel_corr={m['mel_corr']:.3f}  lsd={m['lsd_db']:.1f}dB  rms_diff={m['rms_diff_db']:+.1f}dB")

    if vocoder is not None:
        print("  (A) HiFi-GAN natif...")
        try:
            audio_A = hifigan_decode(vocoder, backend, audio, device)
            save_audio(os.path.join(OUT_DIR, "audio_A_hifigan_native.wav"), audio_A, HIFIGAN_SR)
            results['A'] = audio_metrics(audio, audio_A, HIFIGAN_SR)
            print(f"      {fmt(results['A'])}")
        except Exception as e:
            import traceback; traceback.print_exc()
            results['A'] = None
    else:
        print("  HiFi-GAN non disponible — pip install speechbrain ou vocos")
        results['A'] = None

    print("  (C) Griffin-Lim (phase content)...")
    audio_C = griffinlim_from_stylus_mel(mel_stylus, stft_content=stft_content)
    save_audio(os.path.join(OUT_DIR, "audio_C_griffinlim.wav"), audio_C, STYLUS_SR)
    results['C'] = audio_metrics(audio, audio_C, STYLUS_SR)
    print(f"      {fmt(results['C'])}")

    labels = {'A': '(A) HiFi-GAN natif', 'C': '(C) Griffin-Lim (phase content)'}
    report_path = os.path.join(OUT_DIR, "snr_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Rapport encode/decode mel\n")
        f.write("="*50 + "\n\n")
        f.write(f"Source : {args.input}  |  {args.duration}s @ {STYLUS_SR}Hz\n\n")
        f.write("Metriques (mel_corr: 1=parfait | lsd: 0=parfait | rms_diff: 0=meme volume)\n")
        f.write("-"*50 + "\n")
        for key, label in labels.items():
            m = results.get(key)
            f.write(f"\n  {label}\n")
            if m:
                f.write(f"    mel_corr = {m['mel_corr']:.4f}\n")
                f.write(f"    lsd      = {m['lsd_db']:.2f} dB\n")
                f.write(f"    rms_diff = {m['rms_diff_db']:+.2f} dB\n")
            else:
                f.write("    N/A\n")

    print(f"\nRapport : {report_path}")
    print("\n-- Metriques perceptuelles --")
    for key, label in labels.items():
        print(f"  {label:<35}: {fmt(results.get(key))}")


if __name__ == "__main__":
    main()