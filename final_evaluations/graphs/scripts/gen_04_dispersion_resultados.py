"""Dispersion de resultados: los 4 modelos en un solo grafico (no en pequenos
multiplos como 02_distribucion_episodios.py), con un boxplot por modelo x
escenario (sin/con foto) para ver mediana, cuartiles y rango de un vistazo —
un scatter/strip puro con solo 6 puntos por grupo se leia poco (version
anterior), el boxplot resume mejor la dispersion. Se superponen los 12
episodios como puntos pequenos y semitransparentes encima de cada caja para
no perder la trazabilidad a episodios individuales."""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from style_eval import (ESCENARIOS, ESCENARIO_ALPHA, MODEL_COLORS, MODEL_HATCH,
                         HATCH_EDGE, INK, FS_AXIS_LABEL, FS_TICK,
                         discover_models, escenario_legend, new_fig,
                         save, style_axes)


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


models = discover_models()
if not models:
    raise SystemExit("No hay archivos evaluacion_*_calculado.csv todavia.")

fig, ax = new_fig(width_frac=1.0, aspect=0.68)

group_w = 0.6
box_w = group_w / len(ESCENARIOS) * 0.8
jitter_rnd = random.Random(11)

entries = []  # (position, values, color, alpha, hatch)
for gi, (name, path) in enumerate(models):
    color = MODEL_COLORS[name]
    hatch = MODEL_HATCH.get(name)
    rows = load_rows(path)
    for si, esc in enumerate(ESCENARIOS):
        vals = [float(r["puntuacion_normalizada"]) for r in rows if r["escenario"] == esc]
        xpos = gi - group_w / 2 + (group_w / len(ESCENARIOS)) * si + (group_w / len(ESCENARIOS)) / 2
        entries.append((xpos, vals, color, ESCENARIO_ALPHA[esc], hatch))

bp = ax.boxplot(
    [e[1] for e in entries],
    positions=[e[0] for e in entries],
    widths=box_w,
    patch_artist=True,
    showfliers=False,
    whis=(0, 100),  # bigotes hasta min/max real (n=6 por caja, no hay outliers que recortar)
    zorder=3,
)

for i, (xpos, vals, color, alpha, hatch) in enumerate(entries):
    box = bp["boxes"][i]
    box.set_facecolor(color)
    box.set_alpha(alpha)
    box.set_edgecolor(HATCH_EDGE if hatch else color)
    box.set_hatch(hatch)
    box.set_linewidth(1.1)
    bp["medians"][i].set_color(HATCH_EDGE if hatch else INK)
    bp["medians"][i].set_linewidth(2.0)
    for w in bp["whiskers"][2 * i:2 * i + 2]:
        w.set_color(HATCH_EDGE if hatch else color)
        w.set_linewidth(1.1)
    for c in bp["caps"][2 * i:2 * i + 2]:
        c.set_color(HATCH_EDGE if hatch else color)
        c.set_linewidth(1.1)
    # puntos individuales encima, con jitter, para no perder los 12 episodios
    xs = [xpos + jitter_rnd.uniform(-box_w * 0.28, box_w * 0.28) for _ in vals]
    ax.scatter(xs, vals, s=20, color=HATCH_EDGE if hatch else color, alpha=0.55,
               edgecolor="none", zorder=4)

ax.set_xlim(-0.5, len(models) - 0.5)
ax.set_xticks(range(len(models)))
ax.set_xticklabels([name for name, _ in models], fontsize=FS_TICK + 2, weight="bold", color=INK)
ax.set_ylim(-0.04, 1.08)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Puntuación normalizada (por episodio)", fontsize=FS_AXIS_LABEL, color=INK)
style_axes(ax)
escenario_legend(ax, loc="upper right", ncol=1)

save(fig, "04_dispersion_resultados.png")
