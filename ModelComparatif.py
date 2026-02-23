import torch
import numpy as np
import cv2 
from PIL import Image

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

print("###################################################################################")


# 1. Charger le modèle (SDXL Inpainting est idéal ici)
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda") # Utilise "mps" pour Mac M1/M2/M3 ou "cpu" sinon

# 2. Charger l'image originale et créer un masque pour la zone à modifier
init_image = load_image("image.png").resize((1024, 1024))
#créer un masque blanc (255) pour la zone à modifier et noir (0) pour le reste
mask_image = np.zeros((1024, 1024), dtype=np.uint8)
# Dessiner un rectangle blanc sur le masque pour la zone à modifier (exemple)
cv2.rectangle(mask_image, (350, 350), (650, 650), 255, thickness=-1) # Remplacez les coordonnées par celles de votre zone à modifier
mask_image = Image.fromarray(mask_image, mode='L') # Convertir le masque numpy en PIL Image (grayscale)

# Afficher les images pour vérifier
init_image.show(title="Image Originale")
mask_image.show(title="Masque de Modification")
#afficher l'image sans le masque pour vérifier
masked_image = Image.composite(init_image, Image.new("RGB", init_image.size, (0, 0, 0)), mask_image)
masked_image.show(title="Image avec Masque Appliqué")

print("###################################################################################")


# 3. Définir le prompt pour la zone masquée
prompt = "a marriage proposal on a moutain between Tom (a regular student with glasses) and Vicky (a blonde beautiful woman from Greece), in the style of a romantic painting, with a beautiful sunset in the background"
prompt = "Groot from Guardians of the Galaxy, in the style of a Pixar animation, with a colorful background"
prompt = "a lake with a wooden boat, in the style of a watercolor painting, with a serene atmosphere"

print(f"Prompt utilisé pour l'inpainting : '{prompt}'")

# 4. Exécuter la diffusion locale
image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    strength=1.0,    # Force de la modification (0.0 à 1.0)
    guidance_scale=7.5
).images[0]

image.save("resultat_inpainting_" + prompt.replace(" ", "_") + ".png")