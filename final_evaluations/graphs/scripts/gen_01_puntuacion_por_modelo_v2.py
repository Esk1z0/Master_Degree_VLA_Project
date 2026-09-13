"""Variante de gen_01_puntuacion_por_modelo.py que corrige la identificacion
visual de sin_foto/con_foto: el original codificaba la transparencia al
reves (sin_foto atenuado, con_foto solido) y el rayado diagonal iba atado al
modelo Vanilla en vez de al escenario, así que no marcaba de forma
consistente "con foto" en las demas barras. Aqui se seguido el mismo esquema
que ablation/graphs/generar_graficos_ablacion.py (01_ablacion_capa_unica.png):
la barra IZQUIERDA (sin foto) va en color solido, la DERECHA (con foto) va
con transparencia + rayado diagonal, igual en los 4 modelos.

Salida: 01_puntuacion_por_modelo_v2.png (nuevo nombre, no pisa
01_puntuacion_por_modelo.png)."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from style_eval import (ESCENARIOS, ESCENARIO_LABELS, MODEL_COLORS, INK,
                         FS_AXIS_LABEL, FS_LEGEND, FS_TICK, FS_VALUE,
                         discover_models, new_fig, save, style_axes)
from matplotlib.patches import Rectangle

# esquema izquierda=solido / derecha=transparente+rayada, igual para los 4
# modelos (en vez de ESCENARIO_ALPHA + MODEL_HATCH de style_eval.py, que
# databan la transparencia al reves y el rayado solo a Vanilla)
ESC_ALPHA = {"combinado_sin_foto": 1.0, "combinado_con_foto": 0.55}
ESC_HATCH = {"combinado_sin_foto": None, "combinado_con_foto": "//"}


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
    for si, esc in enumerate(ESCENARIOS):
        val = mean(scores[esc])
        xpos = x[gi] - group_w / 2 + bar_w * si + bar_w / 2
        ax.bar(xpos, val, width=bar_w * 0.92, color=color,
               alpha=ESC_ALPHA[esc],
               edgecolor=color, linewidth=0.9,
               hatch=ESC_HATCH[esc], zorder=3)
        ax.text(xpos, val + 0.02, f"{val:.2f}", ha="center", va="bottom",
                fontsize=FS_VALUE, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([name for name, _ in models], fontsize=FS_TICK + 1, weight="bold", color=INK)
ax.set_ylim(0, 1.32)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Puntuación normalizada media", fontsize=FS_AXIS_LABEL, color=INK)
style_axes(ax)

legend_handles = [
    Rectangle((0, 0), 1, 1, facecolor="#888", alpha=ESC_ALPHA[esc], hatch=ESC_HATCH[esc],
              edgecolor="#888", label=ESCENARIO_LABELS[esc])
    for esc in ESCENARIOS
]
ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=FS_LEGEND, labelcolor=INK)

save(fig, "01_puntuacion_por_modelo_v2.png")
