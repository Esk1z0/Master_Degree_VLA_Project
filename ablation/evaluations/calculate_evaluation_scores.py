"""
Calcula puntuacion_bruta, puntuacion_normalizada y exito_completo a partir
de los 9 criterios binarios del CSV de evaluacion. Muestra resumen por modelo
y por tipo de configuracion (training / no_visto / stress).

Uso:
    python calculate_evaluation_scores.py [ruta_csv]

Si no se especifica ruta, usa eval_v2_checklist.csv en el mismo directorio.
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).parent / "eval_checklist_layercut_0.csv"

BINARY_COLS = [
    "estrella_negra_agarre",
    "estrella_negra_destino",
    "estrella_naranja_agarre",
    "estrella_naranja_destino",
    "cubo_negro_agarre",
    "cubo_negro_destino",
    "cubo_naranja_agarre",
    "cubo_naranja_destino",
    "finalizacion_limpia",
]

GRASP_STAR_COLS  = ["estrella_negra_agarre",  "estrella_naranja_agarre"]
PLACE_STAR_COLS  = ["estrella_negra_destino",  "estrella_naranja_destino"]
GRASP_CUBE_COLS  = ["cubo_negro_agarre",       "cubo_naranja_agarre"]
PLACE_CUBE_COLS  = ["cubo_negro_destino",      "cubo_naranja_destino"]


def _parse_binary(val: str, row_label: str, col: str) -> int | None:
    v = val.strip()
    if v == "":
        return None
    if v in ("0", "1"):
        return int(v)
    raise ValueError(f"[{row_label}] columna '{col}': valor invalido '{v}' (se esperaba 0, 1 o vacio)")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def compute_scores(rows: list[dict]) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    for i, row in enumerate(rows):
        label = f"fila {i + 2} ({row.get('test_id', '?')} modelo={row.get('modelo', '?')} intento={row.get('intento', '?')})"
        binary_vals: list[int | None] = []
        parse_ok = True

        for col in BINARY_COLS:
            try:
                v = _parse_binary(row.get(col, ""), label, col)
                binary_vals.append(v)
            except ValueError as e:
                errors.append(str(e))
                parse_ok = False
                binary_vals.append(None)

        if not parse_ok:
            continue

        filled = [v for v in binary_vals if v is not None]

        if len(filled) == 0:
            continue

        bruta = sum(filled)  # type: ignore[arg-type]

        if len(filled) == 9:
            row["puntuacion_bruta"]       = str(bruta)
            row["puntuacion_normalizada"] = f"{bruta / 9:.2f}"
            row["exito_completo"]         = "1" if bruta == 9 else "0"
        else:
            # Datos parciales: calcula bruta pero no normaliza
            row["puntuacion_bruta"]       = str(bruta)
            row["puntuacion_normalizada"] = ""
            row["exito_completo"]         = ""

    return rows, errors


def _collect_stats(rows: list[dict]) -> dict:
    scores: list[float]       = []
    successes: list[int]      = []
    grasp_star: list[int]     = []
    place_star: list[int]     = []
    grasp_cube: list[int]     = []
    place_cube: list[int]     = []

    for row in rows:
        norm = row.get("puntuacion_normalizada", "").strip()
        if norm:
            try:
                scores.append(float(norm))
            except ValueError:
                pass

        exito = row.get("exito_completo", "").strip()
        if exito in ("0", "1"):
            successes.append(int(exito))

        for col, lst in [
            (GRASP_STAR_COLS, grasp_star),
            (PLACE_STAR_COLS, place_star),
            (GRASP_CUBE_COLS, grasp_cube),
            (PLACE_CUBE_COLS, place_cube),
        ]:
            for c in col:
                v = row.get(c, "").strip()
                if v in ("0", "1"):
                    lst.append(int(v))

    return {
        "n": len(scores),
        "scores": scores,
        "successes": successes,
        "grasp_star": grasp_star,
        "place_star": place_star,
        "grasp_cube": grasp_cube,
        "place_cube": place_cube,
    }


def _print_stats(label: str, stats: dict) -> None:
    n = stats["n"]
    if n == 0:
        print(f"  {label}: sin datos")
        return

    scores    = stats["scores"]
    successes = stats["successes"]
    print(f"  {label}")
    print(f"    Ejecuciones con datos : {n}")
    print(f"    Puntuacion media      : {_mean(scores):.3f}")
    print(f"    Desviacion estandar   : {_std(scores):.3f}")
    if successes:
        rate = _mean(successes)
        print(f"    Tasa exito completo   : {rate:.3f}  ({sum(successes)}/{len(successes)})")
    if stats["grasp_star"]:
        print(f"    Tasa agarre estrellas : {_mean(stats['grasp_star']):.3f}")
    if stats["place_star"]:
        print(f"    Tasa coloc. estrellas : {_mean(stats['place_star']):.3f}")
    if stats["grasp_cube"]:
        print(f"    Tasa agarre cubos     : {_mean(stats['grasp_cube']):.3f}")
    if stats["place_cube"]:
        print(f"    Tasa coloc. cubos     : {_mean(stats['place_cube']):.3f}")


def _print_failure_analysis(rows: list[dict]) -> None:
    modo_counts:  dict[str, int] = defaultdict(int)
    fase_counts:  dict[str, int] = defaultdict(int)
    objeto_counts: dict[str, int] = defaultdict(int)

    total = len(rows)
    for row in rows:
        modo   = row.get("modo_fallo",   "").strip()
        fase   = row.get("fase_fallo",   "").strip()
        objeto = row.get("objeto_fallo", "").strip()
        if modo   and modo   != "ninguno": modo_counts[modo]     += 1
        if fase   and fase   != "ninguna": fase_counts[fase]     += 1
        if objeto and objeto != "ninguno": objeto_counts[objeto] += 1

    def _print_counter(label: str, counts: dict[str, int]) -> None:
        if not counts:
            print(f"  {label}: sin fallos registrados")
            return
        print(f"  {label}:")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            pct = v / total * 100
            print(f"    {k:<25} {v:>3} ejecuciones  ({pct:.1f} %)")

    _print_counter("\nTipos de fallo (modo_fallo)",     modo_counts)
    _print_counter("\nFases con mas errores (fase_fallo)", fase_counts)
    _print_counter("\nObjetos con mas fallos (objeto_fallo)", objeto_counts)


def print_summary(rows: list[dict]) -> None:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        modelo = row.get("modelo", "").strip()
        if modelo:
            by_model[modelo].append(row)

    if not by_model:
        print("\n[Sin datos de modelos todavia — rellena la columna 'modelo' en el CSV]")
        return

    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  RESUMEN GLOBAL MODELO: {modelo}")
    print(sep)
    for modelo in sorted(by_model):
        print(f"\n  Modelo: {modelo}")
        _print_stats("Global", _collect_stats(by_model[modelo]))

        for tipo in ("training", "no_visto", "stress"):
            subset = [r for r in by_model[modelo] if r.get("tipo_configuracion") == tipo]
            if subset:
                _print_stats(tipo, _collect_stats(subset))

    print(f"\n{sep}")
    print(f"  ANALISIS DE FALLOS MODELO: {modelo}")
    print(sep)
    _print_failure_analysis(rows)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH

    if not path.exists():
        print(f"Error: no se encuentra '{path}'")
        sys.exit(1)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    rows, errors = compute_scores(rows)

    if errors:
        print("\nERRORES DE VALIDACION:")
        for e in errors:
            print(f"  - {e}")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV actualizado: {path}")
    print_summary(rows)


if __name__ == "__main__":
    main()
