"""Variante de gen_01_puntuacion_por_modelo.py que anade barras de error
(+/- 1 desviacion estandar) sobre cada barra, con el mismo estilo de "linea
que marca la dispersion" que ya se usa en las graficas de ablacion
(ablation/graphs/generar_graficos_ablacion.py: yerr + capsize + error_kw),
para poder comparar de un vistazo no solo la media por modelo/escenario sino
tambien lo dispersos que son los resultados entre episodios.

Tambien corrige la identificacion visual de sin_foto/con_foto (mismo fallo
que gen_01_puntuacion_por_modelo.py: la transparencia iba al reves y el
rayado dependia del modelo en vez del escenario). Sigue el esquema de
01_ablacion_capa_unica.png: barra IZQUIERDA (sin foto) solida, DERECHA (con
foto) con transparencia + rayado diagonal, igual en los 4 modelos. La
etiqueta de la media va pegada a la barra (no al extremo de la barra de
error) y la linea de dispersion es gris fina, no negra gruesa.

Lee los mismos *_calculado.csv que gen_01_puntuacion_por_modelo.py (no hace
falta editar el script al anadir un modelo nuevo).

Salida: 01b_puntuacion_por_modelo_dispersion.png (nuevo nombre, no pisa
01_puntuacion_por_modelo.png)."""
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from style_eval import (ESCENARIOS, ESCENARIO_LABELS, MODEL_COLORS, INK,
                         FS_AXIS_LABEL, FS_LEGEND, FS_TICK, FS_VALUE,
                         discover_models, new_fig, style_axes)
from matplotlib.patches import Rectangle

# esquema izquierda=solido / derecha=transparente+rayada, igual para los 4
# modelos (en vez de ESCENARIO_ALPHA + MODEL_HATCH de style_eval.py, que
# databan la transparencia al reves y el rayado solo a Vanilla)
ESC_ALPHA = {"combinado_sin_foto": 1.0, "combinado_con_foto": 0.55}
ESC_HATCH = {"combinado_sin_foto": None, "combinado_con_foto": "//"}

DISPERSION_COLOR = "#9aa3af"  # gris, no negro


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


def std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


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
        err = std(scores[esc])
        xpos = x[gi] - group_w / 2 + bar_w * si + bar_w / 2
        ax.bar(xpos, val, width=bar_w * 0.92, color=color,
               alpha=ESC_ALPHA[esc],
               edgecolor=color, linewidth=0.9,
               hatch=ESC_HATCH[esc], zorder=3,
               yerr=err, capsize=3,
               error_kw=dict(elinewidth=0.7, ecolor=DISPERSION_COLOR, zorder=4))
        # la etiqueta va pegada al final de la barra, no al extremo de la
        # barra de error (que puede quedar muy por encima de la media)
        ax.text(xpos, val + 0.02, f"{val:.2f}", ha="center", va="bottom",
                fontsize=FS_VALUE, color=INK, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([name for name, _ in models], fontsize=FS_TICK + 1, weight="bold", color=INK)
ax.set_ylim(0, 1.32)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Puntuación normalizada media (± 1σ)", fontsize=FS_AXIS_LABEL, color=INK)
style_axes(ax)

legend_handles = [
    Rectangle((0, 0), 1, 1, facecolor="#888", alpha=ESC_ALPHA[esc], hatch=ESC_HATCH[esc],
              edgecolor="#888", label=ESCENARIO_LABELS[esc])
    for esc in ESCENARIOS
]
ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=FS_LEGEND, labelcolor=INK)

# save() con bbox_inches="tight" recorta la etiqueta rotada del eje Y
# ("...± 1σ)") por arriba con el pad_inches por defecto; guardamos con un
# margen algo mayor en vez de usar el helper compartido.
_path = os.path.join(os.path.dirname(__file__), "..", "01b_puntuacion_por_modelo_dispersion.png")
fig.savefig(_path, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight", pad_inches=0.2)
print(f"saved {_path}")
