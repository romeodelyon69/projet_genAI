import torch
import torchaudio
import os
import urllib.request
import soundfile as sf  # Ajoute cet import

MODEL_PATH = 'musicnet.ts'  # Chemin pour le modèle RAVE TorchScript

def load_rave_model(model_path='musicnet.ts'):
    """Charge le modèle RAVE (TorchScript)"""
    if not os.path.exists(MODEL_PATH):
        print(f"Téléchargement du modèle {MODEL_PATH} (80h de musique musicnet)...")
        # URL directe pour le modèle musicnet v1
        url = "https://github.com/acids-ircam/RAVE/releases/download/v1.0/musicnet.ts"
        urllib.request.urlretrieve(url, MODEL_PATH)
    
    model = torch.jit.load(MODEL_PATH)
    model.eval()
    return model



def prepare_audio(audio_path, target_sr=44100):
    """Charge l'audio sans utiliser le backend problématique de torchaudio"""
    # 1. Charger avec soundfile (renvoie un numpy array)
    data, sr = sf.read(audio_path)
    
    # Convertir en Tensor [Channels, Samples]
    waveform = torch.from_numpy(data).float()
    if waveform.ndim == 1: # Si c'est déjà mono
        waveform = waveform.unsqueeze(0)
    else: # Si c'est [Samples, Channels], on permute
        waveform = waveform.transpose(0, 1)

    # 2. Rééchantillonnage si nécessaire
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # 3. Mixdown mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 4. Normalisation
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-7)
    
    return waveform.unsqueeze(0) # [1, 1, samples]

def run_rave_inference(input_file, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    # 1. Charger le modèle et l'audio
    model = load_rave_model().to(device)
    x = prepare_audio(input_file).to(device)

    # 2. Inférence
    with torch.no_grad():
        # ENCODAGE
        # z aura une shape [1, 128, N] ou [1, 256, N] selon le modèle
        z = model.encode(x)
        print(f"--- ANALYSE LATENTE ---")
        print(f"Dimension du vecteur z : {z.shape[1]}")
        print(f"Nombre de pas temporels : {z.shape[2]}")

        # --- ICI : Emplacement pour ton code d'interpolation / sampling ---
        # Exemple : transition_z = (z1 * 0.5) + (z2 * 0.5)
        
        # DÉCODAGE
        x_res = model.decode(z)

    # 3. Sauvegarde avec soundfile au lieu de torchaudio
    output_waveform = x_res.squeeze(0).cpu().numpy() # Conversion en numpy pour soundfile
    
    # soundfile attend souvent (samples, channels)
    if output_waveform.ndim > 1:
        output_waveform = output_waveform.T
        
    import soundfile as sf
    sf.write(output_file, output_waveform, 44100)
    
    print(f"Succès ! Fichier reconstruit sauvegardé : {output_file}")

if __name__ == "__main__":
    # Remplace par ton fichier audio pop
    INPUT_WAV = "Vois-sur-ton-chemin.wav"
    
    if os.path.exists(INPUT_WAV):
        run_rave_inference(INPUT_WAV, "reconstruction_musicnet.wav")
    else:
        print(f"Erreur : Place un fichier nommé '{INPUT_WAV}' dans le dossier.")