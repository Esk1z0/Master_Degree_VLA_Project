"""Estilo compartido para las graficas de resultados de la evaluacion final
(final_evaluations/graphs/). Reutiliza la paleta por modelo ya establecida en
las figuras de metodologia (docs/.../figuras/metodologia/scripts/gen_f18_...py:
Vanilla=gris, M=naranja, D=azul, MD=morado) para que la identidad de color de
cada modelo sea consistente en todo el TFM. El naranja se oscurece un poco
respecto al de metodologia (#f0a500 -> #e08e00) porque en las graficas de barras
aqui el color rellena un area grande (no solo una linea), y el tono original no
pasaba el validador de accesibilidad (banda de luminosidad); ver la skill de
dataviz. El gris de Vanilla es una excepcion deliberada (no un fallo del
validador que se ignora): funciona como "referencia/base", igual que en
gen_f18, no como un cuarto color categorico mas.
"""
import glob
import os
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "DejaVu Sans"

BG = "#f7f9fb"
INK = "#1f2937"
MUTED = "#6b7280"
# Texto de nota/pie: mas oscuro que MUTED y con tamano propio mas grande —
# el gris claro original no se leia bien impreso/en pantalla.
NOTE_COLOR = "#454b54"
GRID = "#e2e8f0"

# Tamanos de fuente centralizados (puntos). Subidos respecto a la primera
# version porque la letra, sobre todo la de las notas al pie, no se leia bien.
FS_TITLE = 15
FS_AXIS_LABEL = 12
FS_TICK = 11
FS_LEGEND = 10.5
FS_VALUE = 10
FS_NOTE = 10.5

MODEL_ORDER = ["Vanilla", "M", "D", "MD"]
MODEL_COLORS = {
    "Vanilla": "#5b6472",
    "M": "#e08e00",
    "D": "#4a9eff",
    "MD": "#9b59b6",
}
# Vanilla se dibuja con un leve rayado en vez de depender solo del gris,
# para que la condicion "sin foto / con foto" (alpha) no la deje casi
# invisible en algunas graficas.
MODEL_HATCH = {"Vanilla": "//"}
# El hatch se dibuja con este color (no con el color del modelo): si usaramos
# el mismo gris de relleno como color de trama, las lineas del hatch
# quedarian invisibles sobre su propio relleno.
HATCH_EDGE = "#2f333a"

ESCENARIOS = ["combinado_sin_foto", "combinado_con_foto"]
ESCENARIO_LABELS = {"combinado_sin_foto": "sin foto", "combinado_con_foto": "con foto"}
ESCENARIO_ALPHA = {"combinado_sin_foto": 0.55, "combinado_con_foto": 1.0}

TEXTWIDTH_IN = 6.102  # geometry: a4paper, left=3cm right=2.5cm -> 15.5cm

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evaluations")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..")


def _display_name(model_key):
    """'evaluacion_vanilla' -> 'Vanilla', 'evaluacion_md' -> 'MD', etc."""
    key = re.sub(r"^evaluacion_", "", model_key).lower()
    mapping = {"vanilla": "Vanilla", "m": "M", "d": "D", "md": "MD"}
    return mapping.get(key, model_key)


def discover_models():
    """Devuelve la lista de modelos disponibles (con *_calculado.csv ya generado),
    en el orden fijo MODEL_ORDER, filtrando a los que realmente existen todavia."""
    paths = glob.glob(os.path.join(EVAL_DIR, "evaluacion_*_calculado.csv"))
    found = {}
    for p in paths:
        key = os.path.basename(p).replace("_calculado.csv", "")
        found[_display_name(key)] = p
    return [(name, found[name]) for name in MODEL_ORDER if name in found]


def new_fig(width_frac=1.0, aspect=0.62):
    w = TEXTWIDTH_IN * width_frac
    h = w * aspect
    fig, ax = plt.subplots(figsize=(w, h), dpi=300)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


def style_axes(ax, ygrid=True):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=FS_TICK)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def escenario_legend(ax, **kw):
    handles = [
        Patch(facecolor=MUTED, alpha=ESCENARIO_ALPHA[esc], edgecolor="none",
              label=ESCENARIO_LABELS[esc])
        for esc in ESCENARIOS
    ]
    ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, labelcolor=INK, **kw)


def _wrap_to_width(text, width_in, fontsize):
    """Envuelve 'text' en varias lineas para que quepa en 'width_in' pulgadas
    a 'fontsize' puntos, en vez de dejar una sola linea larga que desborda el
    ancho del grafico (bbox_inches='tight' agranda el PNG para acomodarla).
    Estimacion de ancho medio de caracter en DejaVu Sans italic ~0.52*fontsize,
    con un margen de seguridad del 8%."""
    max_width_pt = width_in * 72 * 0.92
    avg_char_width_pt = fontsize * 0.52
    max_chars = max(20, int(max_width_pt / avg_char_width_pt))
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=max_chars) or [""])
    return "\n".join(lines)


def note(ax, text, y=-0.18):
    """Texto de nota al pie, bajo los ejes — mas grande y oscuro que un
    'gris claro' para que se lea bien impreso, envuelto para no desbordar el
    ancho del grafico."""
    fig = ax.figure
    bbox = ax.get_position()
    axes_width_in = bbox.width * fig.get_figwidth()
    wrapped = _wrap_to_width(text, axes_width_in, FS_NOTE)
    ax.text(0.5, y, wrapped, transform=ax.transAxes, ha="center", va="top",
            fontsize=FS_NOTE, style="italic", color=NOTE_COLOR)


def fig_note(fig, text, y=-0.04, width_in=None):
    """Igual que note(), pero para figuras con varios ejes (pequenos
    multiplos), donde el texto se ancla a la figura entera en vez de a un
    Axes concreto."""
    if width_in is None:
        width_in = fig.get_figwidth()
    wrapped = _wrap_to_width(text, width_in, FS_NOTE)
    fig.text(0.5, y, wrapped, ha="center", va="top", fontsize=FS_NOTE,
              style="italic", color=NOTE_COLOR)


def save(fig, name, width_frac=1.0):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
