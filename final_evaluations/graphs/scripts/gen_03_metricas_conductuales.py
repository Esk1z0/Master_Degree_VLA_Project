"""Panel de metricas conductuales (tasas 0-1) por modelo: exito completo,
timeout, intervencion de emergencia, memoria de oclusion (recuerda/completa) y
rechazo de la foto. Todas se leen de resumen_global_evaluacion.csv (fila
'global' por modelo), salvo tasa_rechazo_foto que viene de la fila
'combinado_con_foto' (solo tiene sentido en esos episodios)."""
import csv
import os
import sys

from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(__file__))
from style_eval import (EVAL_DIR, MODEL_COLORS, MODEL_HATCH, HATCH_EDGE, INK, MUTED,
                         FS_AXIS_LABEL, FS_TICK, FS_LEGEND, FS_VALUE,
                         discover_models, new_fig, note, save, style_axes)

METRICS = [
    ("tasa_exito", "Éxito completo", "global"),
    ("timeouts_rate", "Timeout", "global"),
    ("intervenciones_rate", "Intervención", "global"),
    ("tasa_recuerdo_ocluido", "Recuerda ocluido", "global"),
    ("tasa_completado_ocluido", "Completa ocluido", "global"),
    ("tasa_rechazo_foto", "Rechaza foto", "combinado_con_foto"),
]

resumen_path = os.path.join(EVAL_DIR, "resumen_global_evaluacion.csv")
if not os.path.exists(resumen_path):
    raise SystemExit(
        "resumen_global_evaluacion.csv no existe todavia — hace falta evaluar "
        ">=2 modelos a la vez con calcular_resultados_evaluacion.py para generarlo."
    )

with open(resumen_path, encoding="utf-8") as f:
    resumen_rows = list(csv.DictReader(f))

models = discover_models()
if not models:
    raise SystemExit("No hay archivos evaluacion_*_calculado.csv todavia.")

model_file_key = {name: os.path.basename(path).replace("_calculado.csv", "") for name, path in models}


def get_value(model_key, metric_key, escenario):
    row = next((r for r in resumen_rows if r["modelo"] == model_key and r["escenario"] == escenario), None)
    if row is None:
        return None
    if metric_key == "timeouts_rate":
        n_total = int(row["n_total"])
        return int(row["timeouts"]) / n_total if n_total else None
    if metric_key == "intervenciones_rate":
        n_total = int(row["n_total"])
        return int(row["intervenciones_emergencia"]) / n_total if n_total else None
    val = row.get(metric_key, "")
    return float(val) if val not in ("", None) else None


fig, ax = new_fig(width_frac=1.35, aspect=0.53)

n_metrics = len(METRICS)
n_models = len(models)
group_w = 0.74 if n_models <= 2 else 0.92
bar_w = group_w / n_models
value_fs = FS_VALUE if n_models <= 2 else FS_VALUE - 3.2
x = list(range(n_metrics))

for mi, (metric_key, label, escenario) in enumerate(METRICS):
    for gi, (name, _) in enumerate(models):
        color = MODEL_COLORS[name]
        hatch = MODEL_HATCH.get(name)
        val = get_value(model_file_key[name], metric_key, escenario)
        if val is None:
            continue
        xpos = x[mi] - group_w / 2 + bar_w * gi + bar_w / 2
        ax.bar(xpos, val, width=bar_w * 0.72, color=color,
               edgecolor=(HATCH_EDGE if hatch else color),
               linewidth=0.9, hatch=hatch, zorder=3)
        # pequeno escalonado alterno para que valores identicos en barras
        # contiguas (p. ej. 0.42/0.42) no se toquen
        stagger = 0.035 if gi % 2 else 0.0
        ax.text(xpos, val + 0.02 + stagger, f"{val:.2f}", ha="center", va="bottom",
                fontsize=value_fs, color=INK, rotation=0)

# separador discontinuo entre cada metrica (Exito completo | Timeout |
# Intervencion | ...), para que se lean como bloques distintos y no como
# una fila continua de barras
for b in range(1, n_metrics):
    ax.axvline(x[b] - 0.5, color=MUTED, linewidth=1.1,
               linestyle=(0, (5, 3)), zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([label for _, label, _ in METRICS], fontsize=FS_TICK, color=INK,
                    rotation=25, ha="right", rotation_mode="anchor")
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Tasa (0–1)", fontsize=FS_AXIS_LABEL, color=INK)
style_axes(ax)

handles = [
    Patch(facecolor=MODEL_COLORS[name],
          edgecolor=(HATCH_EDGE if MODEL_HATCH.get(name) else MODEL_COLORS[name]),
          hatch=MODEL_HATCH.get(name), label=name)
    for name, _ in models
]
ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, labelcolor=INK,
          loc="upper center", ncol=len(models))

note(ax, "\"Rechaza la foto\" solo se calcula sobre los 6 episodios combinado_con_foto; el resto son globales (12 episodios).",
     y=-0.32)

if n_models < 4:
    pendientes = [m for m in ["Vanilla", "M", "D", "MD"] if m not in [n for n, _ in models]]
    note(ax, f"Evaluación en curso — pendientes: {', '.join(pendientes)}", y=-0.44)

save(fig, "03_metricas_conductuales.png")
