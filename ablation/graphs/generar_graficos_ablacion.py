"""
Genera las graficas del estudio de ablacion (skip-block analysis) del Action Expert
de SmolVLA, usadas para justificar el diseno de SmolVLA-D / SmolVLA-MD.

Lee todos los `eval_checklist_*.csv` de ../evaluations (excepto la plantilla) y
distingue explicitamente dos fases del estudio:

  - Ablacion de CAPA UNICA  (layercut_0 .. layercut_15): se desactiva 1 de las
    16 capas del Action Expert por ejecucion (skip-block).
  - Ablacion de RANGO / VARIAS CAPAS (layercut_range_6_7, _6_7_8, _7_8_9,
    _9_10_11): se desactivan 2-3 capas consecutivas simultaneamente.

(Nota: el CSV `eval_checklist_layercut_range_7_8_9.csv` tiene un typo en la
columna `modelo` -> "lyercut_range_7_8_9". Se normaliza aqui para que no
aparezca como un modelo distinto en las graficas.)

Salidas (en este mismo directorio):
  01_ablacion_capa_unica_puntuacion.png
  02_ablacion_capa_unica_exito.png
  03_individual_vs_combinado.png
  04_degradacion_no_lineal.png
  05_curva_degradacion_capas.png
  06_heatmap_config_capas_clave.png
  07_modos_fallo_capas_clave.png

Uso:
    cd <repo>/ablation/graphs
    python generar_graficos_ablacion.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EVAL_DIR = HERE.parent / "evaluations"

# Capas elegidas en SmolVLA-D / SmolVLA-MD para la inyeccion de profundidad
# (depth_injection_layers = [6, 7, 8] en configuration_smolvla_d.py / _md.py)
INJECTION_LAYERS = [6, 7, 8]

COLOR_BASE = "#2196F3"
COLOR_INJECTION = "#2E7D32"      # capas elegidas para D/MD
COLOR_CRITICAL = "#C62828"       # capas catastroficas (exito=0, score bajo)
COLOR_OTHER = "#90A4AE"          # resto de capas individuales
COLOR_RANGE = "#8E44AD"          # cortes de varias capas
COLOR_EXPECTED = "#B0BEC5"

CONFIG_TYPES = ["training", "no_visto", "stress"]
CONFIG_LABELS = {"training": "Training", "no_visto": "No visto", "stress": "Stress"}


# ══════════════════════════════════════════════════════════════════════════════
# Carga y normalizacion de datos
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    csv_files = sorted(EVAL_DIR.glob("eval_checklist_*.csv"))
    csv_files = [f for f in csv_files if "template" not in f.name]
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    # Normaliza el typo "lyercut_range_7_8_9" -> "layercut_range_7_8_9"
    df["modelo"] = df["modelo"].replace({"lyercut_range_7_8_9": "layercut_range_7_8_9"})

    # Solo filas con puntuacion normalizada valida (evaluacion completa)
    df["puntuacion_normalizada"] = pd.to_numeric(df["puntuacion_normalizada"], errors="coerce")
    df["exito_completo"] = pd.to_numeric(df["exito_completo"], errors="coerce")
    return df


def is_single_layer(modelo: str) -> bool:
    return bool(re.fullmatch(r"layercut_\d+", modelo))


def layer_index(modelo: str) -> int | None:
    m = re.fullmatch(r"layercut_(\d+)", modelo)
    return int(m.group(1)) if m else None


def is_range(modelo: str) -> bool:
    return modelo.startswith("layercut_range_")


def range_layers(modelo: str) -> list[int]:
    m = re.match(r"layercut_range_(.+)", modelo)
    return [int(x) for x in m.group(1).split("_")] if m else []


def bar_label(ax, bars, values, fmt="{:.2f}", dy=0.015):
    for bar, v in zip(bars, values):
        if np.isnan(v):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + dy, fmt.format(v),
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# 1-2. Ablacion de capa unica: puntuacion y tasa de exito por capa (0..15)
# ══════════════════════════════════════════════════════════════════════════════

def _critical_layers(df: pd.DataFrame, threshold: float = 0.28) -> set[int]:
    """Capas cuya puntuacion normalizada media (no la tasa de exito) cae por debajo del umbral.
    Se calcula SIEMPRE a partir de la puntuacion para que la clasificacion de color sea la misma
    en todos los graficos, independientemente de la metrica que se este representando."""
    single = df[df["modelo"].apply(is_single_layer)].copy()
    single["layer"] = single["modelo"].apply(layer_index)
    means = single.groupby("layer")["puntuacion_normalizada"].mean()
    return set(means[means < threshold].index)


def plot_single_layer(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path, fmt="{:.2f}"):
    single = df[df["modelo"].apply(is_single_layer) | (df["modelo"] == "base")].copy()
    single["layer"] = single["modelo"].apply(lambda m: -1 if m == "base" else layer_index(m))
    stats = single.groupby("layer")[metric].agg(["mean", "std", "count"]).sort_index()
    critical = _critical_layers(df)

    labels = ["Base"] + [f"Capa {i}" for i in stats.index if i >= 0]
    colors = []
    for i in stats.index:
        if i < 0:
            colors.append(COLOR_BASE)
        elif i in INJECTION_LAYERS:
            colors.append(COLOR_INJECTION)
        elif i in critical:
            colors.append(COLOR_CRITICAL)
        else:
            colors.append(COLOR_OTHER)

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(stats))
    yerr = stats["std"].fillna(0).values if metric == "puntuacion_normalizada" else None
    bars = ax.bar(x, stats["mean"].values, yerr=yerr, capsize=4, color=colors, edgecolor="white", linewidth=0.8,
                   error_kw=dict(elinewidth=1.0, ecolor="#333"))
    bar_label(ax, bars, stats["mean"].values, fmt=fmt)

    base_val = stats.loc[-1, "mean"]
    ax.axhline(base_val, color=COLOR_BASE, linestyle="--", linewidth=1.2, alpha=0.6,
               label=f"Base ({fmt.format(base_val)})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=13.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BASE, label="Base (sin ablacion)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_INJECTION, label=f"Capas {INJECTION_LAYERS}: elegidas para inyeccion de profundidad (D / MD)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_CRITICAL, label="Capas criticas (degradacion severa al desactivarlas solas)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_OTHER, label="Resto de capas"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.8, framealpha=0.9)

    fig.suptitle("Ablacion de CAPA UNICA del Action Expert (16 capas, una desactivada por ejecucion)",
                  fontsize=10.5, color="#555", y=0.965)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Individual (6-11) vs combinado (rangos) — comparacion directa
# ══════════════════════════════════════════════════════════════════════════════

def plot_individual_vs_combined(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    indiv_layers = [6, 7, 8, 9, 10, 11]
    indiv_models = [f"layercut_{i}" for i in indiv_layers]
    range_models = ["layercut_range_6_7", "layercut_range_6_7_8", "layercut_range_7_8_9", "layercut_range_9_10_11"]
    range_labels = ["Capas 6+7", "Capas 6+7+8", "Capas 7+8+9", "Capas 9+10+11"]

    base_score = df.loc[df["modelo"] == "base", "puntuacion_normalizada"].mean()
    base_success = df.loc[df["modelo"] == "base", "exito_completo"].mean()

    # ── izquierda: capas individuales 6-11 ───────────────────────────────────
    ax = axes[0]
    indiv_score = [df.loc[df["modelo"] == m, "puntuacion_normalizada"].mean() for m in indiv_models]
    indiv_success = [df.loc[df["modelo"] == m, "exito_completo"].mean() for m in indiv_models]
    x = np.arange(len(indiv_layers))
    w = 0.38
    colors_indiv = [COLOR_INJECTION if i in INJECTION_LAYERS else COLOR_OTHER for i in indiv_layers]
    b1 = ax.bar(x - w / 2, indiv_score, w, label="Puntuacion media", color=colors_indiv, edgecolor="white")
    b2 = ax.bar(x + w / 2, indiv_success, w, label="Tasa de exito", color=colors_indiv, alpha=0.5, edgecolor="white",
                hatch="//")
    bar_label(ax, b1, indiv_score)
    bar_label(ax, b2, indiv_success)
    ax.axhline(base_score, color=COLOR_BASE, linestyle="--", lw=1.2, alpha=0.7, label=f"Base score ({base_score:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Capa {i}" for i in indiv_layers], fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("Desactivando UNA capa a la vez (6 a 11)", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Valor (0-1)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ── derecha: rangos combinados ───────────────────────────────────────────
    ax = axes[1]
    range_score = [df.loc[df["modelo"] == m, "puntuacion_normalizada"].mean() for m in range_models]
    range_success = [df.loc[df["modelo"] == m, "exito_completo"].mean() for m in range_models]
    x = np.arange(len(range_models))
    b1 = ax.bar(x - w / 2, range_score, w, label="Puntuacion media", color=COLOR_RANGE, edgecolor="white")
    b2 = ax.bar(x + w / 2, range_success, w, label="Tasa de exito", color=COLOR_RANGE, alpha=0.5, edgecolor="white",
                hatch="//")
    bar_label(ax, b1, range_score)
    bar_label(ax, b2, range_success)
    ax.axhline(base_score, color=COLOR_BASE, linestyle="--", lw=1.2, alpha=0.7, label=f"Base score ({base_score:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("Desactivando VARIAS capas consecutivas a la vez", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Capa unica vs. combinacion de capas: el dano NO es aditivo", fontsize=14.5, fontweight="bold", y=1.04)
    plt.tight_layout(rect=(0, 0.16, 1, 1))
    fig.text(0.5, 0.02,
              "Las capas 6, 7 y 8 toleran razonablemente bien ser desactivadas UNA A LA VEZ (puntuacion y exito\n"
              "muy por encima del resto de capas). Al desactivar 2-3 de ellas A LA VEZ, el rendimiento colapsa muy\n"
              "por debajo de lo que sugeriria cada capa por separado -> evidencia de que SOLO se puede intervenir\n"
              "ahi con una perturbacion pequena (suma residual zero-init), nunca eliminando o reemplazando esas capas.",
              ha="center", fontsize=9.5, color="#444",
              bbox=dict(facecolor="#EDEDED", edgecolor="#CCC", boxstyle="round,pad=0.5"))

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Degradacion no lineal: esperado (independiente) vs observado (combinado)
# ══════════════════════════════════════════════════════════════════════════════

def plot_nonlinear_degradation(df: pd.DataFrame, out_path: Path):
    combos = [
        ("layercut_range_6_7", [6, 7], "Capas\n6+7"),
        ("layercut_range_6_7_8", [6, 7, 8], "Capas\n6+7+8"),
        ("layercut_range_7_8_9", [7, 8, 9], "Capas\n7+8+9"),
        ("layercut_range_9_10_11", [9, 10, 11], "Capas\n9+10+11"),
    ]

    def mean_of(models, metric):
        return df.loc[df["modelo"].isin(models), metric].mean()

    expected_score, actual_score, expected_succ, actual_succ, gaps = [], [], [], [], []
    for range_model, layers, _ in combos:
        indiv_models = [f"layercut_{i}" for i in layers]
        exp_s = np.mean([df.loc[df["modelo"] == m, "puntuacion_normalizada"].mean() for m in indiv_models])
        act_s = df.loc[df["modelo"] == range_model, "puntuacion_normalizada"].mean()
        exp_e = np.mean([df.loc[df["modelo"] == m, "exito_completo"].mean() for m in indiv_models])
        act_e = df.loc[df["modelo"] == range_model, "exito_completo"].mean()
        expected_score.append(exp_s)
        actual_score.append(act_s)
        expected_succ.append(exp_e)
        actual_succ.append(act_e)
        gaps.append(exp_s - act_s)

    labels = [c[2] for c in combos]
    x = np.arange(len(combos))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    b1 = ax.bar(x - w / 2, expected_score, w, label="Esperado (media de capas individuales)", color=COLOR_EXPECTED,
                edgecolor="white")
    b2 = ax.bar(x + w / 2, actual_score, w, label="Observado (corte combinado real)", color=COLOR_RANGE,
                edgecolor="white")
    bar_label(ax, b1, expected_score)
    bar_label(ax, b2, actual_score)
    for xi, e, a in zip(x, expected_score, actual_score):
        ax.annotate("", xy=(xi + w / 2, a + 0.02), xytext=(xi - w / 2, e + 0.02),
                    arrowprops=dict(arrowstyle="->", color=COLOR_CRITICAL, lw=1.6))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Puntuacion normalizada")
    ax.set_title("Puntuacion: esperado vs. observado", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8.8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax = axes[1]
    b1 = ax.bar(x - w / 2, expected_succ, w, label="Esperado (media de capas individuales)", color=COLOR_EXPECTED,
                edgecolor="white")
    b2 = ax.bar(x + w / 2, actual_succ, w, label="Observado (corte combinado real)", color=COLOR_RANGE,
                edgecolor="white")
    bar_label(ax, b1, expected_succ)
    bar_label(ax, b2, actual_succ)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 0.3)
    ax.set_ylabel("Tasa de exito completo")
    ax.set_title("Tasa de exito: esperado vs. observado", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8.8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("La degradacion combinada es SUPER-ADITIVA (peor de lo esperado) y empeora fuera de 6-8",
                  fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout(rect=(0, 0.14, 1, 1))
    fig.text(0.5, 0.02,
              f"Brecha esperado-observado (puntuacion): {gaps[0]:.2f} (6+7)  ->  {gaps[1]:.2f} (6+7+8)  ->  "
              f"{gaps[2]:.2f} (7+8+9)  ->  {gaps[3]:.2f} (9+10+11).\n"
              "La brecha CRECE al desplazar la ventana hacia capas 9-11 -> refuerza que 6-8 es la ventana menos "
              "fragil disponible para intervenir en el Action Expert.",
              ha="center", fontsize=9.5, color="#444",
              bbox=dict(facecolor="#EDEDED", edgecolor="#CCC", boxstyle="round,pad=0.5"))

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Curva de degradacion por capa (0..15), con rangos como referencia aparte
# ══════════════════════════════════════════════════════════════════════════════

def plot_degradation_curve(df: pd.DataFrame, out_path: Path):
    single = df[df["modelo"].apply(is_single_layer)].copy()
    single["layer"] = single["modelo"].apply(layer_index)
    stats = single.groupby("layer").agg(score=("puntuacion_normalizada", "mean"),
                                         success=("exito_completo", "mean")).sort_index()

    base_score = df.loc[df["modelo"] == "base", "puntuacion_normalizada"].mean()
    base_success = df.loc[df["modelo"] == "base", "exito_completo"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))

    for ax, col, base_val, title in [
        (axes[0], "score", base_score, "Puntuacion normalizada media"),
        (axes[1], "success", base_success, "Tasa de exito completo"),
    ]:
        ax.axvspan(INJECTION_LAYERS[0] - 0.5, INJECTION_LAYERS[-1] + 0.5, color=COLOR_INJECTION, alpha=0.12,
                   label="Ventana elegida para D/MD (capas 6-8)")
        ax.plot(stats.index, stats[col], marker="o", markersize=7, linewidth=2, color="#444", zorder=3)
        for i in INJECTION_LAYERS:
            ax.plot(i, stats.loc[i, col], marker="o", markersize=10, color=COLOR_INJECTION, zorder=4)
        ax.axhline(base_val, color=COLOR_BASE, linestyle="--", linewidth=1.3, alpha=0.7,
                   label=f"Base ({base_val:.2f})")
        ax.set_xticks(stats.index)
        ax.set_xlabel("Capa desactivada (indice, de 16 en el Action Expert)", fontsize=10)
        ax.set_ylabel(title, fontsize=10.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title} por capa desactivada", fontsize=12.5, fontweight="bold")
        ax.legend(fontsize=8.8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle("Mapa de fragilidad de las 16 capas del Action Expert (ablacion de capa unica)", fontsize=14.5,
                  fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Heatmap modelo x tipo_configuracion, restringido a modelos clave
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap_key_models(df: pd.DataFrame, out_path: Path):
    key_models = ["base", "layercut_6", "layercut_7", "layercut_8", "layercut_range_6_7", "layercut_range_6_7_8",
                  "layercut_range_7_8_9", "layercut_range_9_10_11"]
    key_labels = ["Base", "Capa 6\n(individual)", "Capa 7\n(individual)", "Capa 8\n(individual)",
                  "Capas 6+7\n(combinado)", "Capas 6+7+8\n(combinado)", "Capas 7+8+9\n(combinado)",
                  "Capas 9+10+11\n(combinado)"]

    heat = (
        df[df["modelo"].isin(key_models)]
        .groupby(["modelo", "tipo_configuracion"])["puntuacion_normalizada"]
        .mean()
        .unstack("tipo_configuracion")
        .reindex(key_models)[CONFIG_TYPES]
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(heat.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Puntuacion normalizada media", fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(CONFIG_TYPES)))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_TYPES], fontsize=10.5)
    ax.set_yticks(range(len(key_models)))
    ax.set_yticklabels(key_labels, fontsize=9.5)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < 0.35 or val > 0.75 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    # separador visual entre individuales y combinados (el grupo ya esta indicado en cada etiqueta de fila)
    ax.axhline(3.5, color="black", lw=1.5)

    ax.set_title("Robustez por tipo de configuracion — capas clave para D/MD", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Modos de fallo, restringido a modelos clave
# ══════════════════════════════════════════════════════════════════════════════

def plot_failure_modes_key_models(df: pd.DataFrame, out_path: Path):
    key_models = ["base", "layercut_6", "layercut_7", "layercut_8", "layercut_range_6_7", "layercut_range_6_7_8",
                  "layercut_range_7_8_9", "layercut_range_9_10_11"]
    key_labels = ["Base", "Capa 6", "Capa 7", "Capa 8", "Capas\n6+7", "Capas\n6+7+8", "Capas\n7+8+9",
                  "Capas\n9+10+11"]

    failed = df[(df["modelo"].isin(key_models)) & (df["exito_completo"] == 0)].copy()
    fail_counts = failed.groupby(["modelo", "modo_fallo"]).size().unstack(fill_value=0).reindex(key_models)
    fail_pct = fail_counts.div(fail_counts.sum(axis=1), axis=0).fillna(0)

    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(key_models))
    for i, col in enumerate(fail_pct.columns):
        ax.bar(range(len(key_models)), fail_pct[col].values, bottom=bottom, label=col,
               color=palette[i % len(palette)], edgecolor="white", linewidth=0.6)
        bottom += fail_pct[col].values

    ax.axvline(3.5, color="black", lw=1.3, linestyle=":")
    ax.set_xticks(range(len(key_models)))
    ax.set_xticklabels(key_labels, fontsize=10)
    ax.set_ylabel("Proporcion de ejecuciones fallidas", fontsize=10.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Modo de fallo dominante — individual vs. combinado (capas 6-11)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(1.5, 1.02, "individual", ha="center", fontsize=9, color="#555", style="italic")
    ax.text(5.5, 1.02, "combinado (rango)", ha="center", fontsize=9, color="#555", style="italic")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Cargando datos de ../evaluations ...")
    df = load_data()
    print(f"  {len(df)} filas, modelos: {sorted(df['modelo'].unique())}\n")

    print("[1/7] Ablacion de capa unica -- puntuacion ...")
    plot_single_layer(df, "puntuacion_normalizada", "Puntuacion normalizada media",
                       "Puntuacion media al desactivar UNA capa del Action Expert (de 16)",
                       HERE / "01_ablacion_capa_unica_puntuacion.png")

    print("[2/7] Ablacion de capa unica -- tasa de exito ...")
    plot_single_layer(df, "exito_completo", "Tasa de exito completo",
                       "Tasa de exito completo al desactivar UNA capa del Action Expert (de 16)",
                       HERE / "02_ablacion_capa_unica_exito.png")

    print("[3/7] Individual (6-11) vs combinado (rangos) ...")
    plot_individual_vs_combined(df, HERE / "03_individual_vs_combinado.png")

    print("[4/7] Degradacion no lineal (esperado vs observado) ...")
    plot_nonlinear_degradation(df, HERE / "04_degradacion_no_lineal.png")

    print("[5/7] Curva de degradacion por capa ...")
    plot_degradation_curve(df, HERE / "05_curva_degradacion_capas.png")

    print("[6/7] Heatmap de robustez por configuracion (capas clave) ...")
    plot_heatmap_key_models(df, HERE / "06_heatmap_config_capas_clave.png")

    print("[7/7] Modos de fallo (capas clave) ...")
    plot_failure_modes_key_models(df, HERE / "07_modos_fallo_capas_clave.png")

    print("\nListo.")


if __name__ == "__main__":
    main()
