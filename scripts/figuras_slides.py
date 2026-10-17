"""Figuras extras para os slides (PNG grandes, fundo branco)."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

RAIZ = Path(r"C:\Users\offic\Downloads\triaxis\olhovivo\dados_semana")
SAIDA = Path(r"C:\Users\offic\Downloads\triaxis\olhovivo\docs\figuras\slides")
SAIDA.mkdir(parents=True, exist_ok=True)
AZUL, LARANJA, AQUA, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
TINTA, TINTA2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
                     "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": CINZA, "xtick.color": TINTA2, "ytick.color": TINTA2,
                     "axes.labelcolor": TINTA2, "savefig.bbox": "tight", "savefig.dpi": 180})


def salvar(fig, nome):
    fig.savefig(SAIDA / f"{nome}.png", facecolor="white")
    plt.close(fig)
    print(SAIDA / f"{nome}.png")


# 1) corredores: grupos do DBSCAN sobre os tracados
tracos = []
for arq in sorted((RAIZ / "derivado/tracados").glob("*.json")):
    t = json.loads(arq.read_text(encoding="utf-8"))
    if len(t.get("lat") or []) > 1:
        tracos.append(np.column_stack([t["lon"], t["lat"]]))
import pandas as pd
import matplotlib.patheffects as pe
ag = pd.read_parquet(RAIZ / "derivado/amostras_regioes_agregado.parquet")
fig, ax = plt.subplots(figsize=(9, 9))
ax.add_collection(LineCollection(tracos, colors=CINZA, linewidths=1.2))
fora = ag[ag["cluster_dbscan"] < 0]
dentro = ag[ag["cluster_dbscan"] >= 0]
ax.scatter(fora["lon"], fora["lat"], s=12, color="#d9d8d2", zorder=2)
ax.scatter(dentro["lon"], dentro["lat"], s=60, color=LARANJA, edgecolor="white", linewidth=0.6, zorder=3)
halo = [pe.withStroke(linewidth=4, foreground="white")]
for texto, (x, y) in {"Sul\nSto. Amaro – Parelheiros": (-46.62, -23.80),
                      "Centro-sul\nBandeira – Capelinha": (-46.73, -23.63),
                      "Leste\nPq. D. Pedro II – Itaquera": (-46.46, -23.59)}.items():
    ax.annotate(texto, xy=(x, y), fontsize=14, color=TINTA, ha="center", weight="bold",
                path_effects=halo, zorder=5, annotation_clip=False)
ax.set_aspect(1 / np.cos(np.radians(-23.6)))
ax.autoscale_view()
ax.axis("off")
salvar(fig, "corredores")

# 2) modelos: MAE no teste, com os atributos de vizinhanca (comparacao_atraso.json)
nomes = ["Trivial\n(mediana)", "Atraso\nrecente", "XGBoost", "GNN", "LSTM"]
mae = [30.5, 28.8, 24.9, 25.3, 24.6]
cores = [CINZA, CINZA, AZUL, AQUA, LARANJA]
fig, ax = plt.subplots(figsize=(11, 6))
b = ax.bar(nomes, mae, color=cores, width=0.6)
for r, v in zip(b, mae):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.3, f"{v:.1f} s".replace(".", ","),
            ha="center", fontsize=16, color=TINTA, weight="bold")
ax.set_ylim(20, 32)
ax.set_ylabel("erro absoluto médio (s)")
ax.grid(axis="y", color="#e1e0d9")
ax.set_axisbelow(True)
salvar(fig, "modelos")

# 3) achado: viagem x trecho
nomes = ["Só a memória\nda viagem", "Só a memória\ndo trecho", "As duas\njuntas"]
mae = [30.1, 26.2, 25.5]
fig, ax = plt.subplots(figsize=(10, 6))
b = ax.barh(nomes[::-1], mae[::-1], color=[LARANJA, AZUL, CINZA], height=0.55)
for r, v in zip(b, mae[::-1]):
    ax.text(v + 0.2, r.get_y() + r.get_height() / 2, f"{v:.1f} s".replace(".", ","),
            va="center", fontsize=17, color=TINTA, weight="bold")
ax.axvline(30.5, color=TINTA2, linestyle="--", linewidth=1.5)
ax.text(30.5, 2.45, "trivial 30,5 s", ha="center", fontsize=13, color=TINTA2)
ax.set_xlim(20, 33)
ax.set_xlabel("erro absoluto médio da LSTM (s)")
salvar(fig, "achado")

# 4) diagrama da reconstrucao da chegada
fig, ax = plt.subplots(figsize=(12, 4.2))
x = np.linspace(0, 10, 200)
ax.plot(x, 0.25 * np.sin(x / 1.7), color=CINZA, linewidth=6, solid_capstyle="round")
ax.text(0, -0.75, "traçado da linha (GTFS)", fontsize=14, color=TINTA2)
for xp, rot in ((2.2, "GPS às 10:00:00"), (6.8, "GPS às 10:00:30")):
    ax.plot(xp, 0.25 * np.sin(xp / 1.7), "o", color=AZUL, markersize=16)
    ax.text(xp, 0.55, rot, ha="center", fontsize=14, color=AZUL)
xs = 4.9
ax.plot(xs, 0.25 * np.sin(xs / 1.7), "s", color=LARANJA, markersize=18)
ax.text(xs, -0.6, "parada\nchegada interpolada: 10:00:18", ha="center", fontsize=14,
        color=LARANJA, weight="bold")
ax.set_xlim(-0.3, 10.3)
ax.set_ylim(-1.0, 0.9)
ax.axis("off")
salvar(fig, "reconstrucao")
