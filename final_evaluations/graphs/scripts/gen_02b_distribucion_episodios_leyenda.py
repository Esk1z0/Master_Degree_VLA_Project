"""Variante B de la distribucion por episodio: igual que
gen_02_distribucion_episodios.py, pero el valor de la media de cada bloque
(linea discontinua) se muestra en una leyenda por fila, no escrito sobre la
propia grafica. Comparar con la variante C (valor encima de la linea) y con
02_distribucion_episodios.png (sin valor) para elegir estilo final."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from style_eval import (ESCENARIOS, ESCENARIO_LABELS, ESCENARIO_ALPHA, MODEL_COLORS,
                         MODEL_HATCH, HATCH_EDGE, BG, INK, MUTED, TEXTWIDTH_IN,
                         FS_TITLE, FS_AXIS_LABEL, FS_TICK, FS_VALUE, FS_LEGEND,
                         discover_models, fig_note, save)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


models = discover_models()
if not models:
    raise SystemExit("No hay archivos evaluacion_*_calculado.csv todavia.")

n_models = len(models)
row_h = 1.55
fig_h = row_h * n_models + 1.5
fig, axes = plt.subplots(n_models, 1, figsize=(TEXTWIDTH_IN * 1.15, fig_h),
                          dpi=300, sharex=True)
if n_models == 1:
    axes = [axes]
fig.patch.set_facecolor(BG)

fig.suptitle("Puntuación normalizada por episodio, en el orden real de ejecución",
             fontsize=FS_TITLE, weight="bold", color=INK, y=0.995)

for ridx, (name, path) in enumerate(models):
    ax = axes[ridx]
    ax.set_facecolor(BG)
    color = MODEL_COLORS[name]
    hatch = MODEL_HATCH.get(name)
    rows = load_rows(path)
    rows.sort(key=lambda r: r["test_id"])

    xs, vals, escs = [], [], []
    for i, r in enumerate(rows):
        xs.append(i)
        vals.append(float(r["puntuacion_normalizada"]))
        escs.append(r["escenario"])

    for i, (val, esc) in enumerate(zip(vals, escs)):
        ax.bar(i, val, width=0.72, color=color, alpha=ESCENARIO_ALPHA[esc],
               edgecolor=(HATCH_EDGE if hatch else color), linewidth=0.9,
               hatch=hatch, zorder=3)
        ax.text(i, val + 0.04, f"{val:.2f}", ha="center", va="bottom",
                fontsize=FS_VALUE - 2.5, color=INK)

    # media de cada bloque, como referencia horizontal punteada; el valor se
    # lleva a la leyenda de la fila (no se escribe sobre el grafico)
    legend_handles = []
    for esc in ESCENARIOS:
        block_vals = [v for v, e in zip(vals, escs) if e == esc]
        block_idx = [i for i, e in enumerate(escs) if e == esc]
        if not block_vals:
            continue
        mean_v = sum(block_vals) / len(block_vals)
        line_alpha = ESCENARIO_ALPHA[esc]
        ax.plot([min(block_idx) - 0.45, max(block_idx) + 0.45], [mean_v, mean_v],
                color=INK, linewidth=1.1, linestyle=(0, (4, 2)), zorder=4, alpha=line_alpha)
        legend_handles.append(
            Line2D([0], [0], color=INK, alpha=line_alpha, linewidth=1.6,
                   linestyle=(0, (4, 2)),
                   label=f"media {ESCENARIO_LABELS[esc]}: {mean_v:.2f}")
        )

    # separador visual entre el bloque sin_foto y con_foto
    ax.axvline(5.5, color=MUTED, linewidth=1.0, linestyle=(0, (1, 1.5)), zorder=2)

    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=FS_TICK - 1, colors=INK)
    ax.set_ylabel(name, fontsize=FS_AXIS_LABEL, weight="bold", color=color, rotation=0,
                  ha="right", va="center", labelpad=14)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
        ax.spines[spine].set_linewidth(0.8)
    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(handles=legend_handles, loc="upper right", frameon=False,
              fontsize=FS_LEGEND - 1, labelcolor=INK, handlelength=2.2,
              bbox_to_anchor=(1.0, 1.18), ncol=2)

axes[-1].set_xticks(range(12))
axes[-1].set_xticklabels([r["test_id"] for r in load_rows(models[0][1])],
                          fontsize=FS_TICK - 2, color=INK, rotation=0)
axes[-1].set_xlabel("Episodio (orden de ejecución)", fontsize=FS_AXIS_LABEL, color=INK, labelpad=8)

footer = ("Cada barra es un episodio (C-01…C-12); las 6 primeras columnas son sin foto, "
          "las 6 últimas con foto — no alternadas. La media de cada bloque está en la leyenda de cada fila.")
if n_models < 4:
    pendientes = [m for m in ["Vanilla", "M", "D", "MD"] if m not in [n for n, _ in models]]
    footer += f"\nEvaluación en curso — pendientes: {', '.join(pendientes)}"
fig_note(fig, footer, y=-0.04)

fig.tight_layout(rect=[0.06, 0.06, 1, 0.92])
save(fig, "02b_distribucion_episodios_leyenda.png")
