"""
Grid Search Stylus-AudioLDM2 — 8×8 (alpha × gamma) = 64 générations
=====================================================================
Lance le style transfer pour toutes les combinaisons et produit
un CSV récapitulatif pour analyse CLAP score en aval.
"""

import os
import sys
import json
import time
import itertools
import numpy as np
from dataclasses import asdict

from stylus_audioldm2_v4 import StylusAudioLDM2Config, StylusAudioLDM2Pipeline


def main():
    # ── Paramètres de la grille ──────────────────────────────────────────
    N = 8
    alphas = np.linspace(0.0, 1.0, N).round(4)
    gammas = np.linspace(0.0, 1.0, N).round(4)

    style_path   = "musicTI_dataset/audios/timbre/chime/chime1.wav"
    content_path = "musicTI_dataset/audios/content/violin/violin1.wav"
    base_out_dir = "grid_search_output"
    os.makedirs(base_out_dir, exist_ok=True)

    total = len(alphas) * len(gammas)
    print("=" * 60)
    print(f"  Grid Search : {N} alphas × {N} gammas = {total} runs")
    print(f"  Alpha  : {alphas.tolist()}")
    print(f"  Gamma  : {gammas.tolist()}")
    print(f"  Style  : {style_path}")
    print(f"  Content: {content_path}")
    print(f"  Output : {base_out_dir}/")
    print("=" * 60)

    # ── Charger le modèle UNE SEULE FOIS ─────────────────────────────────
    cfg_base = StylusAudioLDM2Config(
        style_audio_path=style_path,
        content_audio_path=content_path,
        skip_roundtrip_check=True,   # on skip les checks pour les 64 runs
        num_inference_steps=50,
    )

    pipeline = StylusAudioLDM2Pipeline(cfg_base)
    pipeline.load_model()

    # ── CSV results ──────────────────────────────────────────────────────
    csv_path = os.path.join(base_out_dir, "grid_results.csv")
    with open(csv_path, "w") as f:
        f.write("run_id,alpha,gamma,use_audio_prompt,output_path,duration_s,status\n")

    # Copier les inputs dans le dossier racine pour référence
    pipeline.proc.save_audio(
        pipeline.proc.load_audio(style_path),
        os.path.join(base_out_dir, "input_style.wav")
    )
    pipeline.proc.save_audio(
        pipeline.proc.load_audio(content_path),
        os.path.join(base_out_dir, "input_content.wav")
    )

    # ── Boucle principale ────────────────────────────────────────────────
    results = []
    for idx, (alpha, gamma) in enumerate(itertools.product(alphas, gammas)):
        alpha = float(alpha)
        gamma = float(gamma)
        run_id  = f"a{alpha:.2f}_g{gamma:.2f}"
        run_dir = os.path.join(base_out_dir, run_id)

        print(f"\n{'━'*60}")
        print(f"  [{idx+1}/{total}]  α={alpha:.4f}  γ={gamma:.4f}")
        print(f"{'━'*60}")

        # Mettre à jour les params sans recharger le modèle
        pipeline.cfg.alpha = alpha
        pipeline.cfg.gamma = gamma
        pipeline.store.alpha = alpha
        pipeline.store.gamma = gamma
        pipeline.store.clear()

        t0 = time.time()
        status = "ok"
        out_path = ""

        try:
            wav_out = pipeline.transfer(
                style_path=style_path,
                content_path=content_path,
                output_dir=run_dir,
            )
            suffix = "audioprompt" if pipeline.cfg.use_audio_prompt else "uncond"
            out_path = os.path.join(
                run_dir, f"stylized_g{gamma}_a{alpha}_{suffix}.wav"
            )
        except Exception as e:
            status = f"error: {e}"
            print(f"  ⚠ ERREUR: {e}")

        elapsed = time.time() - t0

        row = {
            "run_id": run_id,
            "alpha": alpha,
            "gamma": gamma,
            "use_audio_prompt": pipeline.cfg.use_audio_prompt,
            "output_path": out_path,
            "duration_s": round(elapsed, 1),
            "status": status,
        }
        results.append(row)

        with open(csv_path, "a") as f:
            f.write(f"{row['run_id']},{row['alpha']},{row['gamma']},"
                    f"{row['use_audio_prompt']},{row['output_path']},"
                    f"{row['duration_s']},{row['status']}\n")

        print(f"  Durée: {elapsed:.1f}s  Status: {status}")

    # ── Récapitulatif ────────────────────────────────────────────────────
    ok_count  = sum(1 for r in results if r["status"] == "ok")
    err_count = total - ok_count

    print(f"\n{'='*60}")
    print(f"  GRID SEARCH TERMINÉ")
    print(f"  Réussies : {ok_count}/{total}")
    if err_count:
        print(f"  Erreurs  : {err_count}/{total}")
    print(f"  CSV      : {csv_path}")
    print(f"  Dossier  : {base_out_dir}/")
    print(f"{'='*60}")

    # ── Sauvegarder aussi en JSON pour faciliter l'analyse ───────────────
    json_path = os.path.join(base_out_dir, "grid_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "alphas": alphas.tolist(),
            "gammas": gammas.tolist(),
            "style_path": style_path,
            "content_path": content_path,
            "results": results,
        }, f, indent=2)
    print(f"  JSON     : {json_path}")


if __name__ == "__main__":
    main()
