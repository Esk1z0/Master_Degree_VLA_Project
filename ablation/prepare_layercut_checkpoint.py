"""
Crea un directorio wrapper que permite usar SmolVLALayercutPolicy
sobre un checkpoint entrenado como smolvla vanilla.

El script:
  1. Lee el config.json del checkpoint original
  2. Cambia el type a "smolvla_layercut" y añade los campos de ablación
  3. Escribe un nuevo config.json en el directorio de salida
  4. Crea un hardlink de model.safetensors (sin copiar, ocupa 0 bytes extra)
     — si el hardlink falla (sistemas de ficheros distintos), copia el fichero.

Uso:
    python prepare_layercut_checkpoint.py \\
        --source outputs/train/tfm_layer_ablation_expert_only_v3/checkpoints/last/pretrained_model \\
        --output outputs/eval/layercut_layers_4_8 \\
        --ablate_layer_range 4 8

    # O con índices sueltos:
    python prepare_layercut_checkpoint.py \\
        --source outputs/train/tfm_layer_ablation_expert_only_v3/checkpoints/last/pretrained_model \\
        --output outputs/eval/layercut_0_1_2 \\
        --ablate_layer_indices 0 1 2

    # Se pueden combinar:
    python prepare_layercut_checkpoint.py \\
        --source ... --output ... \\
        --ablate_layer_indices 0 \\
        --ablate_layer_range 4 8
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepara un checkpoint SmolVLALayercut.")
    parser.add_argument("--source", required=True,
                        help="Ruta al checkpoint smolvla entrenado (contiene config.json y model.safetensors).")
    parser.add_argument("--output", required=True,
                        help="Directorio de salida para el wrapper layercut.")
    parser.add_argument("--ablate_layer_indices", nargs="*", type=int, default=[],
                        help="Índices sueltos del action expert a saltar (ej: --ablate_layer_indices 0 3 7).")
    parser.add_argument("--ablate_layer_range", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="Intervalo cerrado [start, end] a saltar (ej: --ablate_layer_range 4 8).")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    # ── Validar source ────────────────────────────────────────────────────────
    config_src = source / "config.json"
    weights_src = source / "model.safetensors"

    if not config_src.exists():
        print(f"ERROR: No se encuentra config.json en {source}", file=sys.stderr)
        sys.exit(1)
    if not weights_src.exists():
        print(f"ERROR: No se encuentra model.safetensors en {source}", file=sys.stderr)
        sys.exit(1)

    # ── Leer y modificar config ───────────────────────────────────────────────
    config = json.loads(config_src.read_text(encoding="utf-8"))

    orig_type = config.get("type", "?")
    if orig_type not in ("smolvla", "smolvla_layercut"):
        print(f"ADVERTENCIA: type original es '{orig_type}', se esperaba 'smolvla'.")

    config["type"] = "smolvla_layercut"
    config["ablate_layer_indices"] = args.ablate_layer_indices or []
    config["ablate_layer_range"] = args.ablate_layer_range  # None o [start, end]

    # ── Crear directorio de salida ────────────────────────────────────────────
    output.mkdir(parents=True, exist_ok=True)

    config_dst = output / "config.json"
    config_dst.write_text(json.dumps(config, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"config.json escrito en: {config_dst}")

    # ── Enlazar / copiar pesos ────────────────────────────────────────────────
    weights_dst = output / "model.safetensors"
    if weights_dst.exists() or weights_dst.is_symlink():
        weights_dst.unlink()

    try:
        os.link(weights_src, weights_dst)
        print(f"model.safetensors hardlinkeado desde: {weights_src}")
    except OSError:
        shutil.copy2(weights_src, weights_dst)
        print(f"model.safetensors copiado desde: {weights_src}")

    # ── Copiar todos los ficheros del source (excepto config.json y model.safetensors) ──
    for src_extra in source.iterdir():
        if src_extra.name in ("config.json", "model.safetensors"):
            continue  # ya manejados arriba
        dst_extra = output / src_extra.name
        if src_extra.suffix == ".safetensors":
            # hardlink para .safetensors adicionales (normalizadores, etc.)
            if dst_extra.exists() or dst_extra.is_symlink():
                dst_extra.unlink()
            try:
                os.link(src_extra, dst_extra)
                print(f"  hardlinkeado: {src_extra.name}")
            except OSError:
                shutil.copy2(src_extra, dst_extra)
                print(f"  copiado: {src_extra.name}")
        else:
            shutil.copy2(src_extra, dst_extra)
            print(f"  copiado: {src_extra.name}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    indices = sorted(set(args.ablate_layer_indices or []))
    if args.ablate_layer_range:
        s, e = args.ablate_layer_range
        indices = sorted(set(indices) | set(range(s, e + 1)))

    print(f"\n✓ Wrapper listo en: {output}")
    print(f"  type original     : {orig_type}")
    print(f"  type nuevo        : smolvla_layercut")
    print(f"  capas a saltar    : {indices if indices else '(ninguna — vanilla)'}")
    if args.ablate_layer_range:
        print(f"  ablate_layer_range: [{args.ablate_layer_range[0]}, {args.ablate_layer_range[1]}] (cerrado)")
    print(f"\nUsa este wrapper en lerobot-record con:")
    print(f"  --policy.path={output}")


if __name__ == "__main__":
    main()
