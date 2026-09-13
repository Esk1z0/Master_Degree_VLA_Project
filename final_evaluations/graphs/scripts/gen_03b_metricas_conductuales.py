"""Panel de metricas conductuales (tasas 0-1) por modelo, version consistente
de 03_metricas_conductuales.png.

Ese primer panel calculaba las 6 tasas con dos formulas distintas sin decirlo
claramente: exito/recuerdo/completado/rechazo dividian por el numero de
episodios VALIDOS (registro_valido=1), pero timeout e intervencion dividian
por el numero TOTAL de episodios (incluyendo los invalidos). Con los datos
actuales da igual (no hay episodios invalidos todavia), pero es una trampa en
cuanto aparezca uno.

Aqui las 6 se calculan con la MISMA regla: tasa = (episodios validos con el
evento) / (episodios validos donde el campo aplica). Se lee directamente de
los *_calculado.csv por episodio (via discover_models(), que ya apunta a
esos ficheros) en vez de la fila agregada de resumen_global_evaluacion.csv,
para no heredar la inconsistencia de como se sumaron esos totales.

"Recuerda/completa ocluido" aplican a los 12 episodios (con y sin foto: la
oclusion no depende de si hay foto). "Rechaza foto" solo puede existir en los
6 episodios con foto (evita_intento_agarre_foto esta vacio en los 6 sin foto
porque ahi no hay foto que rechazar) — no es una inconsistencia de formula,
es que la metrica no esta definida fuera de ese subconjunto; por eso su
denominador es mas pequeno y se anota en el pie.
"""
import csv
import os
import sys

from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(__file__))
from style_eval import (MODEL_COLORS, MODEL_HATCH, HATCH_EDGE, INK, MUTED,
                         FS_AXIS_LABEL, FS_TICK, FS_LEGEND, FS_VALUE,
                         discover_models, new_fig, note, save, style_axes)

# (columna en *_calculado.csv, etiqueta, campo que debe estar definido para
#  que el episodio cuente en el denominador — None = siempre definido)
METRICS = [
    ("exito_completo", "Éxito completo", None),
    ("timeout", "Timeout", None),
    ("intervencion_emergencia", "Intervención", None),
    ("recuerda_objeto_ocluido", "Recuerda ocluido", None),
    ("completa_objeto_ocluido", "Completa ocluido", None),
    ("evita_intento_agarre_foto", "Rechaza foto", "evita_intento_agarre_foto"),
]


def compute_rate(rows, event_col, requires_col):
    """tasa = episodios validos con event_col=1 / episodios validos donde
    requires_col (o event_col si no se especifica otro) esta definido."""
    check_col = requires_col or event_col
    valid = [r for r in rows if r.get("registro_valido") == "1"]
    applicable = [r for r in valid if r.get(check_col, "") not in ("", None)]
    if not applicable:
        return None
    hits = sum(1 for r in applicable if r.get(event_col) == "1")
    return hits / len(applicable)


models = discover_models()
if not models:
    raise SystemExit("No hay archivos evaluacion_*_calculado.csv todavia.")

model_rows = {}
for name, path in models:
    with open(path, encoding="utf-8") as f:
        model_rows[name] = list(csv.DictReader(f))

fig, ax = new_fig(width_frac=1.35, aspect=0.53)

n_metrics = len(METRICS)
n_models = len(models)
group_w = 0.74 if n_models <= 2 else 0.92
bar_w = group_w / n_models
value_fs = FS_VALUE if n_models <= 2 else FS_VALUE - 3.2
x = list(range(n_metrics))

for mi, (event_col, label, requires_col) in enumerate(METRICS):
    for gi, (name, _) in enumerate(models):
        color = MODEL_COLORS[name]
        hatch = MODEL_HATCH.get(name)
        val = compute_rate(model_rows[name], event_col, requires_col)
        if val is None:
            continue
        xpos = x[mi] - group_w / 2 + bar_w * gi + bar_w / 2
        ax.bar(xpos, val, width=bar_w * 0.72, color=color,
               edgecolor=(HATCH_EDGE if hatch else color),
               linewidth=0.9, hatch=hatch, zorder=3)
        stagger = 0.035 if gi % 2 else 0.0
        ax.text(xpos, val + 0.02 + stagger, f"{val:.2f}", ha="center", va="bottom",
                fontsize=value_fs, color=INK, rotation=0)

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

note(ax, "Tasa = episodios válidos con el evento / episodios válidos donde aplica. "
         "\"Rechaza foto\" solo aplica a los 6 episodios con foto.",
     y=-0.32)

if n_models < 4:
    pendientes = [m for m in ["Vanilla", "M", "D", "MD"] if m not in [n for n, _ in models]]
    note(ax, f"Evaluación en curso — pendientes: {', '.join(pendientes)}", y=-0.48)

save(fig, "03b_metricas_conductuales.png")
