"""Variante de la celda 6 ("Modos de fallo por modelo") de
../resultados_ablacion.ipynb, con tres cambios pedidos sobre la version
original (12_notebook_modos_fallo_todos_modelos.png):

  - notacion de capas: "LC-N" -> "Capa N", "layercut_range_X_Y" ->
    "Capas X y Y" (o "Capas X, Y y Z" para rangos de 3 capas), en vez de la
    notacion interna layercut_N / layercut_range_X_Y;
  - paleta nueva, menos saturada que tab10, con un color distinto para cada
    modo de fallo real;
  - se quita "ninguno" de la distribucion: es una fila con
    exito_completo=0 pero sin modo de fallo registrado (1 de las 277
    ejecuciones fallidas), un fallo de captura de datos, no un modo de
    fallo real -- no tiene sentido que aparezca en la distribucion de
    modos de fallo por modelo si no representa ningun fallo concreto.

Salida: 12_notebook_modos_fallo_todos_modelos_v2.png (nuevo nombre, no pisa
el original).
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EVAL_DIR = HERE.parent / "evaluations"

# ── Carga de datos (igual que la celda 0 del notebook) ──────────────────────
csv_files = sorted(EVAL_DIR.glob("eval_checklist_*.csv"))
csv_files = [f for f in csv_files if "template" not in f.name]
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


def _sort_key(name):
    if name == "base":
        return -1
    m = re.match(r"layercut_(\d+)$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"range_([\d_]+)$", name)
    if m:
        return 100 + int(m.group(1).split("_")[0])
    return 9999


model_order = sorted(df["modelo"].unique().tolist(), key=_sort_key)


def _label(name):
    if name == "base":
        return "Base"
    m = re.match(r"layercut_(\d+)$", name)
    if m:
        return f"Capa {m.group(1)}"
    m = re.search(r"range_([\d_]+)$", name)
    if m:
        nums = m.group(1).split("_")
        return f"Capas {', '.join(nums[:-1])} y {nums[-1]}" if len(nums) > 2 else f"Capas {nums[0]} y {nums[1]}"
    return name


model_labels = [_label(m) for m in model_order]

# ── 6. Modos de fallo por modelo (stacked bar) ───────────────────────────────
# "ninguno" descartado ANTES de agrupar: no es un modo de fallo real (ver
# docstring), asi que no debe contar ni en el numerador ni en el denominador
# de las proporciones.
failed = df[(df["exito_completo"] == 0) & (df["modo_fallo"] != "ninguno")].copy()
fail_counts = (
    failed.groupby(["modelo", "modo_fallo"])
    .size()
    .unstack(fill_value=0)
    .reindex(model_order)
)
fail_pct = fail_counts.div(fail_counts.sum(axis=1), axis=0).fillna(0)

# paleta propia, menos saturada que tab10, un color por modo de fallo real
MUTED_PALETTE = {
    "bloqueo": "#5B7C99",
    "agarre_fallido": "#C97064",
    "timeout": "#D4A64B",
    "sin_aproximacion": "#6FA287",
    "objeto_caido": "#9B7EBD",
    "otro": "#A8A8A8",
    "colision": "#C77DA8",
    "varios": "#7D9471",
}

fig, ax = plt.subplots(figsize=(12.5, 5.5))
bottom = np.zeros(len(model_order))
for col in fail_pct.columns:
    color = MUTED_PALETTE.get(col, "#888888")
    ax.bar(range(len(model_order)), fail_pct[col].values, bottom=bottom,
           label=col, color=color, edgecolor="white", linewidth=0.5)
    bottom += fail_pct[col].values

ax.set_xticks(range(len(model_order)))
ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=13)
ax.tick_params(axis="y", labelsize=12)
ax.set_ylabel("Proporción de fallos", fontsize=13)
ax.set_title("Distribución de modos de fallo por modelo", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12, title="Modo de fallo", title_fontsize=12.5)
ax.set_xlabel("Modelo", fontsize=13)
plt.tight_layout()

out = HERE / "12_notebook_modos_fallo_todos_modelos_v2.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
