"""
Scores CLAP pour évaluation du transfert de style audio.

Supporte laion_clap et msclap (fallback gracieux si non installé).
Le modèle est chargé une seule fois (singleton).

Trois métriques complémentaires :

  clap_style_score(output, style, sr)
      cos(e_out, e_style)
      → "Le résultat ressemble-t-il au style ?"
      Insuffisant seul : un output identique au style aurait un score parfait.

  clap_content_score(output, content, sr)
      cos(e_out, e_content)
      → "Le contenu est-il préservé ?"

  clap_directional_score(output, style, content, sr)
      cos(e_out - e_content, e_style - e_content)
      → "L'output s'est-il déplacé dans la bonne direction, du contenu vers le style ?"
      C'est la métrique centrale pour évaluer le transfert (analogue au DirectionalCLIP).

  clap_scores(output, style, content, sr)
      → dict regroupant les trois scores, point d'entrée recommandé.
"""

import os
import tempfile
import numpy as np
import soundfile as sf


_clap_model = None  # singleton


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _load_clap():
    global _clap_model
    if _clap_model is not None:
        return _clap_model
    try:
        import laion_clap

        _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        _clap_model.load_ckpt()
        print("CLAP loaded (laion_clap)")
    except ImportError:
        try:
            from msclap import CLAP

            _clap_model = CLAP(version="2023", use_cuda=False)
            print("CLAP loaded (msclap)")
        except ImportError:
            _clap_model = None
            print(
                "WARNING: CLAP not available (install laion_clap or msclap). "
                "CLAP score will be skipped."
            )
    return _clap_model


def _write_tmp(audio: np.ndarray, sr: int) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, audio, sr)
    f.close()
    return f.name


def _get_embed_fn(model):
    """Retourne une fonction embed(audio, sr) -> np.ndarray selon le backend CLAP."""
    try:
        import laion_clap  # noqa: F401

        def embed(audio, sr):
            path = _write_tmp(audio, sr)
            emb = model.get_audio_embedding_from_filelist([path], use_tensor=False)
            os.unlink(path)
            return emb[0]

        return embed
    except ImportError:
        pass

    # msclap fallback
    def embed(audio, sr):
        path = _write_tmp(audio, sr)
        emb = model.get_audio_embeddings([path])
        os.unlink(path)
        return emb[0].numpy()

    return embed


# ─────────────────────────────────────────────────────────────────────────────
# API publique
# ─────────────────────────────────────────────────────────────────────────────


def clap_scores(
    output: np.ndarray,
    style_ref: np.ndarray,
    content_ref: np.ndarray,
    sr: int,
) -> dict | None:
    """
    Calcule les trois scores CLAP en un seul appel (3 embeddings au total).
    Retourne None si CLAP n'est pas disponible.

    Clés du dict retourné :
      'style'       : cos(e_out, e_style)            — ressemblance au style
      'content'     : cos(e_out, e_content)          — préservation du contenu
      'directional' : cos(e_out - e_content, e_style - e_content)
                      — direction du transfert (métrique principale)
    """
    model = _load_clap()
    if model is None:
        return None

    try:
        embed = _get_embed_fn(model)
        e_out = embed(output, sr)
        e_style = embed(style_ref, sr)
        e_content = embed(content_ref, sr)

        direction_taken = e_out - e_content
        direction_target = e_style - e_content

        return {
            "style": _cosine(e_out, e_style),
            "content": _cosine(e_out, e_content),
            "directional": _cosine(direction_taken, direction_target),
        }
    except Exception as e:
        print(f"CLAP scoring failed: {e}")
        return None


def clap_style_score(
    output: np.ndarray, style_ref: np.ndarray, content_ref: np.ndarray, sr: int
) -> float | None:
    """cos(e_out, e_style). Voir clap_scores() pour la métrique complète."""
    scores = clap_scores(output, style_ref, content_ref, sr)
    return scores["style"] if scores else None


def clap_content_score(
    output: np.ndarray, style_ref: np.ndarray, content_ref: np.ndarray, sr: int
) -> float | None:
    """cos(e_out, e_content). Voir clap_scores() pour la métrique complète."""
    scores = clap_scores(output, style_ref, content_ref, sr)
    return scores["content"] if scores else None


def clap_directional_score(
    output: np.ndarray, style_ref: np.ndarray, content_ref: np.ndarray, sr: int
) -> float | None:
    """
    cos(e_out - e_content, e_style - e_content).
    Mesure si l'output s'est déplacé dans la direction contenu→style.
    C'est la métrique principale pour évaluer la qualité du transfert.
    """
    scores = clap_scores(output, style_ref, content_ref, sr)
    return scores["directional"] if scores else None
