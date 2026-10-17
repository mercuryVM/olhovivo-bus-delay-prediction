"""Gera as figuras da secao de resultados preliminares, em PDF (vetorial, para o
LaTeX) e PNG (para conferir).

    python scripts/figuras_resultados.py [pasta_dos_dados] [pasta_de_saida]

Le as saidas ja sintetizadas: percursos.parquet, os tracados e, se estiver por
perto, o previsao_realizado.parquet.
"""
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

RAIZ = Path(sys.argv[1] if len(sys.argv) > 1 else "dados")
SAIDA = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/figuras")
SAIDA.mkdir(parents=True, exist_ok=True)
FUSO = "America/Sao_Paulo"
FERIADO = date(2026, 9, 7)
MIN_CELULA = 30  # observacoes minimas para a celula entrar no grafico

# paleta validada: tres primeiras posicoes categoricas + rampa sequencial azul
AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, MUDO = "#0b0b0b", "#52514e", "#898781"
GRADE, EIXO = "#e1e0d9", "#c3c2b7"
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
RAMPA_LARANJA = ["#fbe3d5", "#f6c3a5", "#ef9468", "#eb6834", "#c14f23", "#8f3a17"]
SEQ_AZUL = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
# invertida: velocidade baixa (o caso ruim) fica escura, como o atraso alto
SEQ_LARANJA = LinearSegmentedColormap.from_list("laranja", RAMPA_LARANJA[::-1])
# a janela de casamento entre previsao e chegada e de 60 min: acima disto a
# chegada muito atrasada nao casa e o erro medido cai artificialmente
HORIZONTE_MAX_MIN = 45

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.labelcolor": TINTA2,
    "axes.edgecolor": EIXO,
    "axes.titlecolor": TINTA,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.6,
    "xtick.color": MUDO,
    "ytick.color": MUDO,
    "xtick.labelcolor": TINTA2,
    "ytick.labelcolor": TINTA2,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def salvar(fig, nome):
    for ext in ("pdf", "png"):
        fig.savefig(SAIDA / f"{nome}.{ext}")
    plt.close(fig)
    print("  ", SAIDA / f"{nome}.pdf")


def carregar_percursos() -> pd.DataFrame:
    p = pd.read_parquet(RAIZ / "derivado/percursos.parquet")
    local = p["t_chegada"].dt.tz_convert(FUSO)
    p["data"] = local.dt.date
    dow = local.dt.dayofweek
    p["tipo_dia"] = np.select(
        [(dow == 6) | (p["data"] == FERIADO), dow == 5],
        ["domingo/feriado", "sábado"], "dia útil",
    )
    return p


def por(df, chaves):
    g = df.groupby(chaves, observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "taxa": g["atrasado"].mean(),
        "p90": g["atraso_s"].quantile(0.9),
        # em proporcao do tempo tipico do trecho: o p90 em segundos acompanha o
        # tamanho do trecho e faria parada de trecho longo parecer pior
        "p90_rel": g["atraso_rel"].quantile(0.9) * 100,
        "vel": g["velocidade_kmh"].median(),
    }).reset_index()


# --------------------------------------------------------------------------
# Figura 1 — perfil ao longo do dia, por tipo de dia
# --------------------------------------------------------------------------
def figura_perfil(p):
    ordem = ["dia útil", "sábado", "domingo/feriado"]
    cores = dict(zip(ordem, [AZUL, LARANJA, AQUA]))
    h = por(p, ["tipo_dia", "hora"])
    h = h[h["n"] >= MIN_CELULA]

    fig, eixos = plt.subplots(1, 2, figsize=(6.4, 2.15), constrained_layout=True)
    for ax, campo, rotulo, fmt in (
        (eixos[0], "taxa", "trechos atrasados (%)", lambda v: v * 100),
        (eixos[1], "vel", "velocidade mediana (km/h)", lambda v: v),
    ):
        for tipo in ordem:
            # horas sem dado viram NaN para a linha quebrar, e nao ligar 0h a 4h
            # por cima da pausa da coleta
            d = h[h["tipo_dia"] == tipo].set_index("hora").reindex(range(24)).reset_index()
            ax.plot(d["hora"], fmt(d[campo]), color=cores[tipo], linewidth=2,
                    marker="o", markersize=3.2, markeredgecolor="white",
                    markeredgewidth=0.6, label=tipo, zorder=3)
        ax.set_xlabel("hora do dia")
        ax.set_ylabel(rotulo)
        ax.set_xticks(range(0, 24, 4))
        ax.set_xlim(-0.5, 23.5)
        ax.grid(axis="x", visible=False)
        # faixas de pico ao fundo
        for ini, fim in ((6, 8), (16, 19)):
            ax.axvspan(ini - 0.5, fim + 0.5, color=GRADE, alpha=0.55, zorder=0, lw=0)
    eixos[0].set_ylim(0, None)
    eixos[0].annotate("faixas de pico", xy=(17.5, eixos[0].get_ylim()[1] * 0.06),
                      color=MUDO, fontsize=7.5, ha="center")
    # legenda acima dos dois paineis, fora da area das linhas
    fig.legend(*eixos[0].get_legend_handles_labels(), loc="outside upper center",
               ncols=3, fontsize=8, labelcolor=TINTA2, handlelength=1.6)
    salvar(fig, "perfil-dia")


# --------------------------------------------------------------------------
# Figura 2 — erro da previsao da API por antecedencia
# --------------------------------------------------------------------------
def figura_previsao():
    import duckdb

    arq = None
    for cand in (RAIZ / "derivado/previsao_realizado.parquet",
                 Path("../release-dados-v1/previsao_realizado.parquet"),
                 Path("release-dados-v1/previsao_realizado.parquet")):
        if cand.exists():
            arq = cand
            break
    if arq is None:
        print("   previsao_realizado.parquet nao encontrado — figura 2 pulada")
        return
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=4;")
    d = con.execute(f"""
        SELECT floor(horizonte_s / 120) * 2 + 1 AS minutos,
               count(*) AS n,
               avg(erro_abs_s) AS mae,
               avg(CASE WHEN erro_s >= 300 THEN 1 ELSE 0 END) AS p5
        FROM read_parquet('{arq.as_posix()}')
        WHERE horizonte_s < {HORIZONTE_MAX_MIN * 60}
        GROUP BY 1 ORDER BY 1
    """).df()
    d = d[d["n"] >= 1000]

    fig, eixos = plt.subplots(1, 2, figsize=(6.4, 1.85), constrained_layout=True)
    eixos[0].plot(d["minutos"], d["mae"], color=AZUL, linewidth=2, zorder=3)
    eixos[0].set_ylabel("erro absoluto médio (s)")
    eixos[1].plot(d["minutos"], d["p5"] * 100, color=AZUL, linewidth=2, zorder=3)
    eixos[1].set_ylabel("chegam ≥ 5 min depois (%)")
    for ax in eixos:
        ax.set_xlabel("antecedência da previsão (min)")
        ax.set_xlim(0, HORIZONTE_MAX_MIN)
        ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
    salvar(fig, "erro-previsao")
    print("   pares usados:", int(d["n"].sum()))


# --------------------------------------------------------------------------
# Figura 3 — mapa: atraso p90 e velocidade mediana por parada
# --------------------------------------------------------------------------
def figura_mapa(p, min_trechos=100):
    ag = por(p, "cp_destino").rename(columns={"cp_destino": "cp"})
    ag = ag[ag["n"] >= min_trechos]
    pos = p.groupby("cp_destino")[["lat", "lon"]].mean()
    ag = ag.join(pos, on="cp")
    cat = RAIZ / "catalogo/paradas.parquet"
    if cat.exists():
        par = pd.read_parquet(cat, columns=["cp", "lat", "lon"]).drop_duplicates("cp")
        par["cp"] = par["cp"].astype("int64")
        par = par.set_index("cp").reindex(ag["cp"])
        for c in ("lat", "lon"):
            oficial = par[c].to_numpy(dtype=float)
            ag[c] = np.where(np.isfinite(oficial), oficial, ag[c].to_numpy(dtype=float))

    tracos = []
    for arq in sorted((RAIZ / "derivado/tracados").glob("*.json")):
        t = json.loads(arq.read_text(encoding="utf-8"))
        if len(t.get("lat") or []) > 1:
            tracos.append(np.column_stack([t["lon"], t["lat"]]))

    fig, eixos = plt.subplots(1, 2, figsize=(6.4, 3.0), constrained_layout=True)
    aspecto = 1 / np.cos(np.radians(ag["lat"].mean()))
    for ax, campo, cmap, rotulo in (
        (eixos[0], "p90_rel", SEQ_AZUL, "atraso p90 (% do tempo típico) — escuro é pior"),
        (eixos[1], "vel", SEQ_LARANJA, "velocidade mediana (km/h) — escuro é pior"),
    ):
        ax.add_collection(LineCollection(tracos, colors=GRADE, linewidths=0.8, zorder=1))
        lo, hi = np.nanpercentile(ag[campo], [5, 95])
        pc = ax.scatter(ag["lon"], ag["lat"], c=ag[campo], cmap=cmap,
                        norm=Normalize(vmin=lo, vmax=hi),
                        s=13, linewidths=0.3, edgecolors="white", zorder=3)
        barra = fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.02,
                             fraction=0.05, aspect=28, extend="both")
        barra.set_label(rotulo, color=TINTA2, fontsize=8)
        barra.outline.set_visible(False)
        barra.ax.tick_params(labelsize=7.5, color=MUDO, labelcolor=TINTA2)
        ax.set_aspect(aspecto)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.autoscale_view()

    contorno = [pe.withStroke(linewidth=2.2, foreground="white")]
    for ax in eixos:
        for texto, (x, y) in {
            "Sto. Amaro –\nParelheiros": (-46.78, -23.82),
            "Bandeira –\nCapelinha": (-46.71, -23.63),
            "Pq. D. Pedro II –\nItaquera": (-46.46, -23.59),
        }.items():
            ax.annotate(texto, xy=(x, y), fontsize=6.5, color=TINTA2, ha="center",
                        path_effects=contorno, zorder=5)
        # escala de 5 km
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        km = 5 / (111.32 * np.cos(np.radians(ag["lat"].mean())))
        xe, ye = x0 + 0.04 * (x1 - x0), y0 + 0.05 * (y1 - y0)
        ax.plot([xe, xe + km], [ye, ye], color=TINTA2, linewidth=1.4, zorder=5)
        ax.annotate("5 km", xy=(xe + km / 2, ye + 0.012 * (y1 - y0)), fontsize=6.5,
                    color=TINTA2, ha="center", zorder=5)
    salvar(fig, "mapa-paradas")
    print("   paradas no mapa:", len(ag))


if __name__ == "__main__":
    print("lendo", RAIZ)
    p = carregar_percursos()
    print("percursos:", p.shape)
    figura_perfil(p)
    figura_previsao()
    figura_mapa(p)

    # numeros que vao no texto e nas tabelas
    print("\n== por tipo de dia e pico")
    print(por(p, ["tipo_dia", "pico"]).round(3).to_string(index=False))
    print("\n== extremos do perfil horario (dia util)")
    u = por(p[p["tipo_dia"] == "dia útil"], "hora")
    u = u[u["n"] >= MIN_CELULA]
    print("   maior taxa:", u.loc[u["taxa"].idxmax(), ["hora", "taxa", "vel"]].to_dict())
    print("   menor velocidade:", u.loc[u["vel"].idxmin(), ["hora", "taxa", "vel"]].to_dict())
