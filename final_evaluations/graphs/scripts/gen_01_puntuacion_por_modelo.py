"""Grafica principal: puntuacion normalizada media por modelo, separada en
sin_foto / con_foto. Lee directamente los *_calculado.csv de cada modelo ya
evaluado (no hace falta editar el script al anadir D o MD: basta con volver a
ejecutarlo)."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from style_eval import (ESCENARIOS, ESCENARIO_ALPHA, MODEL_COLORS,
                         MODEL_HATCH, HATCH_EDGE, INK, FS_AXIS_LABEL, FS_TICK,
                         FS_VALUE, discover_models, escenario_legend, new_fig,
                         save, style_axes)


def load_scores(path):
    """{escenario: [puntuacion_normalizada, ...]} para un modelo."""
    scores = {esc: [] for esc in ESCENARIOS}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            esc = row.get("escenario")
            if esc in scores and row.get("puntuacion_normalizada"):
                scores[esc].append(float(row["puntuacion_normalizada"]))
    return scores


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


models = discover_models()
if not models:
    raise SystemExit("No hay archivos evaluacion_*_calculado.csv todavia.")

fig, ax = new_fig(width_frac=1.0, aspect=0.6)

n_models = len(models)
group_w = 0.62
bar_w = group_w / len(ESCENARIOS)
x = list(range(n_models))

for gi, (name, path) in enumerate(models):
    scores = load_scores(path)
    color = MODEL_COLORS[name]
    hatch = MODEL_HATCH.get(name)
    for si, esc in enumerate(ESCENARIOS):
        val = mean(scores[esc])
        xpos = x[gi] - group_w / 2 + bar_w * si + bar_w / 2
        ax.bar(xpos, val, width=bar_w * 0.92, color=color,
               alpha=ESCENARIO_ALPHA[esc],
               edgecolor=(HATCH_EDGE if hatch else color), linewidth=0.9,
               hatch=hatch, zorder=3)
        ax.text(xpos, val + 0.02, f"{val:.2f}", ha="center", va="bottom",
                fontsize=FS_VALUE, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([name for name, _ in models], fontsize=FS_TICK + 1, weight="bold", color=INK)
ax.set_ylim(0, 1.32)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Puntuación normalizada media", fontsize=FS_AXIS_LABEL, color=INK)
style_axes(ax)
escenario_legend(ax, loc="upper right", ncol=1)

save(fig, "01_puntuacion_por_modelo.png")
