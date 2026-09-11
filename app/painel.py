"""Painel do estudo: mapa de São Paulo com o atraso por parada, as regiões
encontradas pelos agrupamentos, os fluxos origem -> destino, os padrões no tempo
e o erro da previsão da API.

    streamlit run app/painel.py

Lê só as saídas já sintetizadas (derivado/ e catalogo/). Usa dados_semana/ se
existir, senão dados/; a variável OLHOVIVO_DADOS aponta outra pasta.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
FUSO = "America/Sao_Paulo"
# feriados dentro do período coletado — contam como domingo
FERIADOS = [date(2026, 9, 7)]

FAIXAS = {
    "madrugada": "madrugada 0–5h",
    "pico_manha": "pico manhã 6–8h",
    "entrepico": "entrepico 9–15h",
    "pico_tarde": "pico tarde 16–19h",
    "noite": "noite 20–23h",
}
TIPOS_DIA = ["útil", "sábado", "domingo/feriado"]
DIAS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
HORIZONTES = ["0-5 min", "5-15 min", "15-30 min", "30-60 min"]

ALGORITMOS = {
    "kmeans": "K-means",
    "dbscan": "DBSCAN",
    "hdbscan": "HDBSCAN",
    "st-dbscan": "ST-DBSCAN (espaço + tempo)",
}
DESCRICOES = {
    "kmeans": "Particiona as paradas em k grupos pela posição. Toda parada entra em "
    "algum grupo — serve de base para comparar com os métodos por densidade.",
    "dbscan": "Só entram as paradas com probabilidade de atraso acima da taxa geral. "
    "Agrupa as que ficam próximas umas das outras; as isoladas viram ruído.",
    "hdbscan": "Mesmas paradas do DBSCAN, mas sem raio fixo: o algoritmo escolhe a "
    "densidade de cada grupo.",
    "st-dbscan": "Roda sobre as ocorrências de atraso, exigindo proximidade no espaço "
    "e no tempo. Cada grupo é um episódio — um lugar num intervalo de tempo —, não "
    "uma região fixa. A recorrência mostra onde os episódios se repetem.",
}

# nome -> (coluna, formatador, maior é melhor?)
METRICAS = {
    "Atraso p90 (s)": ("atraso_p90_s", lambda v: f"{v:.0f} s", False),
    "Atraso médio (s)": ("atraso_medio_s", lambda v: f"{v:.0f} s", False),
    "Taxa de atraso": ("taxa", lambda v: f"{v:.0%}", False),
    "Velocidade mediana (km/h)": ("vel_mediana", lambda v: f"{v:.1f} km/h", True),
}
METRICAS_PREV = {
    "Erro absoluto médio (s)": ("mae_s", lambda v: f"{v:.0f} s"),
    "Chegou 5 min ou mais depois do previsto": ("p_atraso5", lambda v: f"{v:.0%}"),
    "Chegou 1 min ou mais antes do previsto": ("p_adiantado1", lambda v: f"{v:.0%}"),
}

# verde -> amarelo -> vermelho
RAMPA = np.array(
    [[26, 152, 80], [145, 207, 96], [254, 224, 139], [252, 141, 89], [215, 48, 39]],
    dtype=float,
)
# tab20 com os tons escuros primeiro, para grupos vizinhos não saírem parecidos
_TAB20 = [
    [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
    [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
    [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
    [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
    [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229],
]
PALETA = _TAB20[0::2] + _TAB20[1::2]
COR_TRACADO = [70, 90, 120, 110]
COR_RUIDO = [150, 150, 150, 150]

TOOLTIP = {
    "html": "<b>{titulo}</b><br/>{corpo}",
    "style": {
        "backgroundColor": "#1f2937",
        "color": "white",
        "fontSize": "12px",
        "maxWidth": "360px",
        "whiteSpace": "pre-line",
    },
}

COLUNAS = [
    "cl", "letreiro", "sentido", "viagem_id", "cp_origem", "cp_destino",
    "ordem_origem", "ordem_destino", "t_chegada", "atraso_s", "atrasado",
    "velocidade_kmh", "hora", "faixa", "regiao_origem", "regiao_destino",
    "lat", "lon",
]

CFG_TABELA = {
    "nome": st.column_config.TextColumn("parada"),
    "linhas": st.column_config.TextColumn("linhas"),
    "n": st.column_config.NumberColumn("trechos", format="localized"),
    "taxa": st.column_config.NumberColumn("taxa de atraso", format="percent"),
    "atraso_p90_s": st.column_config.NumberColumn("atraso p90 (s)", format="%.0f"),
    "atraso_medio_s": st.column_config.NumberColumn("atraso médio (s)", format="%.1f"),
    "vel_mediana": st.column_config.NumberColumn("velocidade (km/h)", format="%.1f"),
    "regiao_origem": st.column_config.TextColumn("origem"),
    "regiao_destino": st.column_config.TextColumn("destino"),
}


# --------------------------------------------------------------------------
# utilitários
# --------------------------------------------------------------------------
def milhar(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".")


def pct(v: float) -> str:
    return f"{v:.1%}".replace(".", ",")


def rotulo_data(d: date) -> str:
    extra = " (feriado)" if d in FERIADOS else ""
    return f"{DIAS[d.weekday()]} {d:%d/%m}{extra}"


def escala(valores, maior_melhor: bool = False, limites=None):
    """Leva os valores para 0..1, onde 1 é o pior. Corta nos percentis 5 e 95
    para um valor extremo não achatar a escala inteira."""
    v = np.asarray(valores, dtype=float)
    if limites is None:
        finitos = v[np.isfinite(v)]
        limites = tuple(np.percentile(finitos, [5, 95])) if finitos.size else (0.0, 1.0)
    lo, hi = limites
    if hi <= lo:
        hi = lo + 1e-9
    x = np.clip((v - lo) / (hi - lo), 0, 1)
    if maior_melhor:
        x = 1 - x
    return np.nan_to_num(x, nan=0.0), (lo, hi)


def cores(x, alpha: int = 220) -> list[list[int]]:
    pos = np.asarray(x, dtype=float) * (len(RAMPA) - 1)
    i = np.minimum(pos.astype(int), len(RAMPA) - 2)
    f = (pos - i)[:, None]
    rgb = RAMPA[i] * (1 - f) + RAMPA[i + 1] * f
    return [[int(r), int(g), int(b), alpha] for r, g, b in rgb]


def legenda(titulo: str, limites, fmt, maior_melhor: bool = False) -> None:
    lo, hi = limites
    grad = ", ".join(f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in RAMPA)
    esq, dir_ = (fmt(hi), fmt(lo)) if maior_melhor else (fmt(lo), fmt(hi))
    st.markdown(
        f"<div style='font-size:0.8rem;margin-bottom:2px'>{titulo}</div>"
        f"<div style='height:10px;border-radius:3px;"
        f"background:linear-gradient(90deg,{grad})'></div>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.75rem'>"
        f"<span>{esq} ou melhor</span><span>{dir_} ou pior</span></div>",
        unsafe_allow_html=True,
    )


def circulo(lat: float, lon: float, raio_m: float, lados: int = 24) -> list:
    dlat = raio_m / 111_320
    dlon = raio_m / (111_320 * math.cos(math.radians(lat)))
    return [
        [lon + dlon * math.cos(a), lat + dlat * math.sin(a)]
        for a in np.linspace(0, 2 * math.pi, lados)
    ]


def linhas_por(df: pd.DataFrame, chave) -> pd.Series:
    """Letreiros que passam por cada valor da chave, como texto."""
    chaves = [chave] if isinstance(chave, str) else list(chave)
    pares = df[chaves + ["letreiro"]].drop_duplicates()
    pares = pares.assign(letreiro=pares["letreiro"].astype(str)).sort_values("letreiro")
    return pares.groupby(chaves, observed=True)["letreiro"].agg(", ".join)


def agrega(df: pd.DataFrame, chaves) -> pd.DataFrame:
    g = df.groupby(chaves, observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "taxa": g["atrasado"].mean(),
        "atraso_medio_s": g["atraso_s"].mean(),
        "atraso_p90_s": g["atraso_s"].quantile(0.9),
        "vel_mediana": g["velocidade_kmh"].median(),
    }).reset_index()


def com_catalogo(ag: pd.DataFrame, paradas: pd.DataFrame) -> pd.DataFrame:
    """Nome e posição oficiais da parada; o ponto médio do trecho fica de reserva."""
    cat = paradas.reindex(ag["cp"].to_numpy())
    nomes = cat["nome"].to_numpy(dtype=object)
    ag["nome"] = [
        n if isinstance(n, str) and n else f"parada {cp}"
        for n, cp in zip(nomes, ag["cp"])
    ]
    for c in ("lat", "lon"):
        oficial = cat[c].to_numpy(dtype=float)
        if c in ag:
            ag[c] = np.where(np.isfinite(oficial), oficial, ag[c].to_numpy(dtype=float))
        else:
            ag[c] = oficial
    return ag


def mapa(camadas, altura: int = 600, pitch: int = 0, zoom: float = 9.7) -> None:
    deck = pdk.Deck(
        layers=[c for c in camadas if c is not None],
        initial_view_state=pdk.ViewState(
            latitude=-23.66, longitude=-46.60, zoom=zoom, pitch=pitch
        ),
        map_provider="carto",
        map_style="light",
        tooltip=TOOLTIP,
    )
    st.pydeck_chart(deck, height=altura)


def camada(tipo: str, data: pd.DataFrame, **kw) -> pdk.Layer:
    """O tooltip do pydeck no Streamlit escapa o HTML que vem nos dados. A dica vai
    em texto puro na coluna `tip`; a primeira linha vira o título em negrito."""
    if "tip" in data:
        partes = data["tip"].fillna("").astype(str).str.split("\n", n=1)
        data = data.drop(columns="tip").assign(
            titulo=partes.str[0].fillna(""),
            corpo=partes.str[1].fillna(""),
        )
    return pdk.Layer(tipo, data=data, **kw)


def camada_tracados(tracados: pd.DataFrame, linhas, sentidos):
    if tracados.empty:
        return None
    d = tracados[tracados["letreiro"].isin(linhas) & tracados["sentido"].isin(sentidos)]
    return camada(
        "PathLayer",
        data=d[["path", "tip"]],
        get_path="path",
        get_color=COR_TRACADO,
        get_width=15,
        width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
    )


# --------------------------------------------------------------------------
# leitura (uma vez por processo)
# --------------------------------------------------------------------------
def pasta_dados() -> Path:
    if os.environ.get("OLHOVIVO_DADOS"):
        return Path(os.environ["OLHOVIVO_DADOS"])
    semana = RAIZ_PROJETO / "dados_semana"
    if (semana / "derivado" / "percursos.parquet").exists():
        return semana
    return RAIZ_PROJETO / "dados"


@st.cache_resource(show_spinner="Lendo os percursos…")
def carregar_percursos(raiz: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(raiz) / "derivado" / "percursos.parquet", columns=COLUNAS)
    local = df["t_chegada"].dt.tz_convert(FUSO)
    datas = local.dt.date
    df["dia_semana"] = local.dt.dayofweek
    domingo = (df["dia_semana"] == 6) | datas.isin(FERIADOS)
    df["tipo_dia"] = np.select(
        [domingo, df["dia_semana"] == 5], ["domingo/feriado", "sábado"], "útil"
    )
    df["data"] = pd.Categorical(datas)
    for c in ("letreiro", "faixa", "regiao_origem", "regiao_destino", "tipo_dia"):
        df[c] = df[c].astype(str).astype("category")
    df.attrs["periodo"] = (local.min(), local.max())
    return df.drop(columns="t_chegada")


@st.cache_resource
def carregar_paradas(raiz: str) -> pd.DataFrame:
    arq = Path(raiz) / "catalogo" / "paradas.parquet"
    if not arq.exists():
        return pd.DataFrame(columns=["nome", "lat", "lon"], index=pd.Index([], name="cp"))
    df = pd.read_parquet(arq, columns=["cp", "nome", "lat", "lon"])
    df["cp"] = df["cp"].astype("int64")
    df["nome"] = df["nome"].fillna("").astype(str).str.strip()
    return df.drop_duplicates("cp").set_index("cp")


@st.cache_resource
def carregar_tracados(raiz: str) -> pd.DataFrame:
    perc = carregar_percursos(raiz)
    linhas = perc[["cl", "letreiro", "sentido"]].drop_duplicates("cl")
    regs = []
    for cl, lt, sentido in linhas.itertuples(index=False):
        arq = Path(raiz) / "derivado" / "tracados" / f"{cl}.json"
        if not arq.exists():
            continue
        t = json.loads(arq.read_text(encoding="utf-8"))
        if len(t.get("lat") or []) < 2:
            continue
        regs.append({
            "letreiro": str(lt),
            "sentido": int(sentido),
            "path": [[lo, la] for la, lo in zip(t["lat"], t["lon"])],
            "tip": f"linha {lt} · sentido {sentido}\n"
            f"{(t.get('comprimento_m') or 0) / 1000:.1f} km de itinerário",
        })
    return pd.DataFrame(regs, columns=["letreiro", "sentido", "path", "tip"])


@st.cache_resource
def rotulos_das_linhas(raiz: str) -> dict[str, str]:
    perc = carregar_percursos(raiz)
    s1 = perc.loc[perc["sentido"] == 1, ["letreiro", "regiao_origem", "regiao_destino"]]
    s1 = s1.drop_duplicates("letreiro")
    return {
        str(lt): f"{o} ↔ {d}"
        for lt, o, d in zip(s1["letreiro"], s1["regiao_origem"], s1["regiao_destino"])
    }


@st.cache_resource
def tipo_por_dia_semana(raiz: str) -> dict[int, str]:
    """A previsão está agregada por dia da semana; herda o tipo de dia do percurso."""
    perc = carregar_percursos(raiz)
    return {
        int(dow): str(g.mode().iat[0])
        for dow, g in perc.groupby("dia_semana")["tipo_dia"]
    }


@st.cache_resource
def carregar_regioes(raiz: str) -> pd.DataFrame:
    arq = Path(raiz) / "derivado" / "regioes.geojson"
    if not arq.exists():
        return pd.DataFrame()
    regs = []
    for f in json.loads(arq.read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        geo = f.get("geometry") or {}
        if geo.get("type") == "Polygon":
            anel = geo["coordinates"][0]
        elif geo.get("type") == "MultiPolygon":
            anel = geo["coordinates"][0][0]
        else:
            anel = circulo(p["lat"], p["lon"], p.get("raio_m") or 300)
        fluxos = p.get("fluxos") or []
        regs.append({
            "algoritmo": p.get("algoritmo"),
            "rotulo": p.get("rotulo"),
            "n_paradas": len(p.get("paradas") or []),
            "n_amostras": p.get("n_amostras"),
            "n_obs": p.get("n_observacoes", p.get("n_amostras")),
            "atraso_medio_s": p.get("atraso_medio_s"),
            "atraso_p90_s": p.get("atraso_p90_s"),
            "prob_atraso": p.get("prob_atraso"),
            "fluxo_principal": " · ".join(
                f"{x['origem']} → {x['destino']} ({x['n']})" for x in fluxos[:2]
            ),
            "linhas": ", ".join(p.get("linhas") or []),
            "inicio": p.get("inicio"),
            "fim": p.get("fim"),
            "poligono": anel,
        })
    df = pd.DataFrame(regs)
    for c in ("inicio", "fim"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="ISO8601")
        df[c] = df[c].dt.tz_convert(FUSO)
    return df


@st.cache_resource
def carregar_agregado(raiz: str) -> pd.DataFrame:
    arq = Path(raiz) / "derivado" / "amostras_regioes_agregado.parquet"
    if not arq.exists():
        return pd.DataFrame()
    return com_catalogo(pd.read_parquet(arq), carregar_paradas(raiz))


@st.cache_resource
def carregar_eventos(raiz: str) -> pd.DataFrame:
    arq = Path(raiz) / "derivado" / "amostras_regioes_eventos.parquet"
    if not arq.exists():
        return pd.DataFrame()
    ev = pd.read_parquet(
        arq, columns=["t", "cp", "lat", "lon", "valor_s", "letreiro", "cluster_st-dbscan"]
    )
    ev["data"] = ev["t"].dt.tz_convert(FUSO).dt.date
    return ev.rename(columns={"cluster_st-dbscan": "rotulo"})


@st.cache_resource
def carregar_previsao(raiz: str) -> pd.DataFrame | None:
    arq = Path(raiz) / "derivado" / "painel_previsao.parquet"
    if not arq.exists():
        return None
    df = pd.read_parquet(arq)
    df["cp"] = df["cp"].astype("int64")
    df["letreiro"] = df["letreiro"].astype(str)
    return df


@st.cache_resource
def terminais(raiz: str) -> pd.DataFrame:
    """Posição dos terminais de cada fluxo: primeira e última parada da linha."""
    perc = carregar_percursos(raiz)
    paradas = carregar_paradas(raiz)
    i_o = perc.groupby("cl")["ordem_origem"].idxmin().to_numpy()
    i_d = perc.groupby("cl")["ordem_destino"].idxmax().to_numpy()
    o = perc.loc[i_o, ["cl", "cp_origem", "regiao_origem", "regiao_destino"]].set_index("cl")
    d = perc.loc[i_d, ["cl", "cp_destino"]].set_index("cl")
    e = o.join(d)
    for sufixo, col in (("o", "cp_origem"), ("d", "cp_destino")):
        cat = paradas.reindex(e[col].to_numpy())
        e[f"lat_{sufixo}"] = cat["lat"].to_numpy(dtype=float)
        e[f"lon_{sufixo}"] = cat["lon"].to_numpy(dtype=float)
    e = e.dropna(subset=["lat_o", "lon_o", "lat_d", "lon_d"])
    e["regiao_origem"] = e["regiao_origem"].astype(str)
    e["regiao_destino"] = e["regiao_destino"].astype(str)
    return (
        e.groupby(["regiao_origem", "regiao_destino"])[["lat_o", "lon_o", "lat_d", "lon_d"]]
        .mean()
        .reset_index()
    )


# --------------------------------------------------------------------------
# recortes — cacheados pela combinação de filtros (o _sel não entra no hash,
# porque é função dos filtros)
# --------------------------------------------------------------------------
@st.cache_data(max_entries=64, show_spinner=False)
def agrega_recorte(raiz: str, filtros: tuple, chaves, _sel: pd.DataFrame) -> pd.DataFrame:
    return agrega(_sel, list(chaves) if isinstance(chaves, tuple) else chaves)


@st.cache_data(max_entries=32, show_spinner=False)
def indicadores(raiz: str, filtros: tuple, _sel: pd.DataFrame) -> tuple:
    return (
        len(_sel),
        _sel["viagem_id"].nunique(),
        _sel["atrasado"].mean(),
        _sel["atraso_s"].quantile(0.9),
        _sel["velocidade_kmh"].median(),
    )


@st.cache_data(max_entries=32, show_spinner=False)
def paradas_do_recorte(raiz: str, filtros: tuple, _sel: pd.DataFrame) -> pd.DataFrame:
    ag = agrega(_sel, "cp_destino").rename(columns={"cp_destino": "cp"})
    ag = ag.join(_sel.groupby("cp_destino")[["lat", "lon"]].mean(), on="cp")
    ag["linhas"] = ag["cp"].map(linhas_por(_sel, "cp_destino"))
    ag = com_catalogo(ag, carregar_paradas(raiz))
    ag["tip"] = [
        f"{nome}\nlinhas {lts}\n{milhar(n)} trechos · taxa {tx:.0%}\n"
        f"atraso p90 {p90:.0f} s · médio {md:+.0f} s\nvelocidade mediana {vel:.1f} km/h"
        for nome, lts, n, tx, p90, md, vel in zip(
            ag["nome"], ag["linhas"], ag["n"], ag["taxa"], ag["atraso_p90_s"],
            ag["atraso_medio_s"], ag["vel_mediana"],
        )
    ]
    return ag


@st.cache_data(max_entries=32, show_spinner=False)
def fluxos_do_recorte(raiz: str, filtros: tuple, _sel: pd.DataFrame) -> pd.DataFrame:
    chaves = ["regiao_origem", "regiao_destino"]
    od = agrega(_sel, chaves)
    od["linhas"] = pd.MultiIndex.from_frame(od[chaves]).map(linhas_por(_sel, chaves))
    for c in chaves:
        od[c] = od[c].astype(str)
    od = od.merge(terminais(raiz), on=chaves, how="left")
    od["tip"] = [
        f"{o} → {d}\nlinhas {lts}\n{milhar(n)} trechos · taxa {tx:.0%}"
        f"\natraso p90 {p90:.0f} s · velocidade {vel:.1f} km/h"
        for o, d, lts, n, tx, p90, vel in zip(
            od["regiao_origem"], od["regiao_destino"], od["linhas"], od["n"],
            od["taxa"], od["atraso_p90_s"], od["vel_mediana"],
        )
    ]
    return od


@st.cache_data(max_entries=32, show_spinner=False)
def histograma(raiz: str, filtros: tuple, _sel: pd.DataFrame) -> pd.DataFrame:
    bordas = np.arange(-300, 930, 30)
    cont, _ = np.histogram(_sel["atraso_s"].clip(-300, 899), bins=bordas)
    return pd.DataFrame({"de": bordas[:-1], "ate": bordas[1:], "trechos": cont})


def agrega_previsao(p: pd.DataFrame, chaves) -> pd.DataFrame:
    """Média ponderada pelo número de pares — MAE e proporções se recombinam exato."""
    w = p.assign(
        _mae=p["mae_s"] * p["n"],
        _a5=p["p_atraso5"] * p["n"],
        _ad=p["p_adiantado1"] * p["n"],
    )
    s = w.groupby(chaves, observed=True)[["n", "_mae", "_a5", "_ad"]].sum()
    return pd.DataFrame({
        "n": s["n"],
        "mae_s": s["_mae"] / s["n"],
        "p_atraso5": s["_a5"] / s["n"],
        "p_adiantado1": s["_ad"] / s["n"],
    }).reset_index()


@st.cache_data(max_entries=32, show_spinner=False)
def resumo_previsao(raiz: str, linhas: tuple, sentidos: tuple, faixas: tuple,
                    tipos: tuple, horizontes: tuple):
    pv = carregar_previsao(raiz)
    tipo = pv["dia_semana"].map(tipo_por_dia_semana(raiz))
    p = pv[
        pv["letreiro"].isin(linhas)
        & pv["sentido"].isin(sentidos)
        & pv["faixa"].isin(faixas)
        & tipo.isin(tipos)
        & pv["horizonte"].isin(horizontes)
    ]
    if p.empty:
        return None
    n = p["n"].sum()
    kpis = (
        int(n),
        (p["mae_s"] * p["n"]).sum() / n,
        (p["p_atraso5"] * p["n"]).sum() / n,
        (p["p_adiantado1"] * p["n"]).sum() / n,
    )
    por_cp = agrega_previsao(p, "cp")
    por_cp["linhas"] = por_cp["cp"].map(linhas_por(p, "cp"))
    por_cp = com_catalogo(por_cp, carregar_paradas(raiz))
    por_cp["tip"] = [
        f"{nome}\nlinhas {lts}\n{milhar(n)} pares\n"
        f"erro absoluto médio {mae:.0f} s\n≥ 5 min depois {a5:.0%} · "
        f"≥ 1 min antes {ad:.0%}"
        for nome, lts, n, mae, a5, ad in zip(
            por_cp["nome"], por_cp["linhas"], por_cp["n"], por_cp["mae_s"],
            por_cp["p_atraso5"], por_cp["p_adiantado1"],
        )
    ]
    return kpis, por_cp, agrega_previsao(p, ["letreiro", "horizonte"])


# --------------------------------------------------------------------------
# abas — cada uma é um fragmento: mexer num controle dela não refaz a página
# --------------------------------------------------------------------------
@st.fragment
def aba_mapa(sel, filtros, linhas, sentidos, min_n):
    c1, c2, c3 = st.columns([2, 3, 1])
    nome_met = c1.selectbox("Cor das paradas", list(METRICAS), key="met_mapa")
    modo = c2.radio(
        "Mostrar", ["Paradas", "Hexágonos"], horizontal=True, key="modo_mapa",
        help="Hexágonos contam os trechos atrasados em cada área de ~350 m. "
        "Refletem o volume de ônibus, além do atraso.",
    )
    tres_d = c3.toggle("3D", key="3d_mapa")
    campo, fmt, maior_melhor = METRICAS[nome_met]

    ag = paradas_do_recorte(RAIZ, filtros, sel)
    ag = ag[ag["n"] >= min_n].reset_index(drop=True)

    camadas = [camada_tracados(tracados, linhas, sentidos)]
    if modo == "Paradas":
        x, lim = escala(ag[campo], maior_melhor)
        ag["cor"] = cores(x)
        if tres_d:
            ag["altura"] = 100 + x * 3000
            camadas.append(camada(
                "ColumnLayer", data=ag[["lon", "lat", "cor", "altura", "tip"]],
                get_position=["lon", "lat"], get_elevation="altura", radius=90,
                get_fill_color="cor", extruded=True, pickable=True, auto_highlight=True,
            ))
        else:
            ag["raio"] = 60 + 140 * np.sqrt(ag["n"] / max(ag["n"].max(), 1))
            camadas.append(camada(
                "ScatterplotLayer", data=ag[["lon", "lat", "cor", "raio", "tip"]],
                get_position=["lon", "lat"], get_radius="raio",
                radius_min_pixels=3, radius_max_pixels=16, get_fill_color="cor",
                stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1,
                pickable=True, auto_highlight=True,
            ))
        legenda(nome_met, lim, fmt, maior_melhor)
    else:
        pts = sel.loc[sel["atrasado"] == 1, ["lon", "lat"]]
        if len(pts) > 80_000:
            pts = pts.sample(80_000, random_state=0)
        camadas.append(camada(
            "HexagonLayer", data=pts, get_position=["lon", "lat"], radius=350,
            coverage=0.85, extruded=tres_d, elevation_scale=6,
            elevation_range=[0, 2500], opacity=0.55,
            color_range=[[int(r), int(g), int(b)] for r, g, b in RAMPA],
        ))
        legenda(
            "Trechos atrasados na área", (0, 1),
            lambda v: "menos" if v == 0 else "mais",
        )

    mapa(camadas, pitch=45 if tres_d else 0)
    st.caption(
        "A referência de cada trecho é a mediana dele mesmo na semana, então a taxa "
        "de atraso fica perto de 25 % em quase todo lugar. Para comparar regiões, o "
        "p90 e a velocidade dizem mais. A chegada é reconstruída do GPS e tende a "
        "sair alguns segundos atrasada nas paradas de muito embarque."
    )

    st.subheader("Paradas mais críticas")
    st.dataframe(
        ag.sort_values(campo, ascending=maior_melhor).head(20)[
            ["nome", "linhas", "n", "taxa", "atraso_p90_s", "atraso_medio_s", "vel_mediana"]
        ],
        hide_index=True, column_config=CFG_TABELA,
    )


def _regioes_espaciais(r: pd.DataFrame, alg: str, camadas: list) -> None:
    cor_por = st.radio(
        "Cor das regiões", ["Atraso médio (s)", "Probabilidade de atraso"],
        horizontal=True, key="cor_reg",
    )
    col = "atraso_medio_s" if cor_por.startswith("Atraso") else "prob_atraso"
    fmt_reg = (lambda v: f"{v:.0f} s") if col == "atraso_medio_s" else (lambda v: f"{v:.0%}")
    x, lim = escala(r[col])
    r["cor"] = cores(x, alpha=70)
    r["borda"] = cores(x, alpha=230)
    r["tip"] = [
        f"{ALGORITMOS[alg]} · grupo {rot}\n{npar} paradas · "
        f"{milhar(nobs)} trechos\natraso médio {am:.1f} s · "
        f"prob. de atraso {pa:.0%}\n{fl}\nlinhas {lts}"
        for rot, npar, nobs, am, pa, fl, lts in zip(
            r["rotulo"], r["n_paradas"], r["n_obs"].fillna(0), r["atraso_medio_s"],
            r["prob_atraso"], r["fluxo_principal"], r["linhas"],
        )
    ]
    camadas.append(camada(
        "PolygonLayer", data=r[["poligono", "cor", "borda", "tip"]],
        get_polygon="poligono", get_fill_color="cor", get_line_color="borda",
        line_width_min_pixels=1.5, stroked=True, filled=True,
        pickable=True, auto_highlight=True,
    ))

    am = carregar_agregado(RAIZ)
    coluna = f"cluster_{alg}"
    if coluna in am:
        pts = am[am[coluna] != -3].copy()  # -3: parada fora do recorte do método
        pts["cor"] = [
            PALETA[int(c) % len(PALETA)] + [230] if c >= 0 else COR_RUIDO
            for c in pts[coluna]
        ]
        pts["tip"] = [
            f"{nome}\n{'grupo ' + str(int(c)) if c >= 0 else 'ruído'}"
            f"\n{milhar(n)} trechos · prob. de atraso {pa:.0%}"
            f"\natraso p90 {p90:.0f} s"
            for nome, c, n, pa, p90 in zip(
                pts["nome"], pts[coluna], pts["n"], pts["prob_atraso"], pts["valor_p90_s"],
            )
        ]
        camadas.append(camada(
            "ScatterplotLayer", data=pts[["lon", "lat", "cor", "tip"]],
            get_position=["lon", "lat"], get_radius=70, radius_min_pixels=3,
            get_fill_color="cor", stroked=True, get_line_color=[255, 255, 255],
            line_width_min_pixels=0.5, pickable=True, auto_highlight=True,
        ))
    legenda(f"Cor das regiões: {cor_por.lower()}", lim, fmt_reg)
    mapa(camadas)
    st.caption("Pontos: paradas, com a cor do seu grupo; cinza = ruído.")

    st.subheader("Grupos")
    st.dataframe(
        r.sort_values(col, ascending=False)[
            ["rotulo", "n_paradas", "n_obs", "atraso_medio_s", "prob_atraso",
             "fluxo_principal", "linhas"]
        ],
        hide_index=True,
        column_config={
            "rotulo": st.column_config.NumberColumn("grupo", format="%d"),
            "n_paradas": st.column_config.NumberColumn("paradas"),
            "n_obs": st.column_config.NumberColumn("trechos", format="localized"),
            "atraso_medio_s": st.column_config.NumberColumn("atraso médio (s)", format="%.1f"),
            "prob_atraso": st.column_config.NumberColumn("prob. de atraso", format="percent"),
            "fluxo_principal": st.column_config.TextColumn("fluxos principais (origem → destino)"),
            "linhas": st.column_config.TextColumn("linhas"),
        },
    )


def _episodios(r: pd.DataFrame, camadas: list) -> None:
    c1, c2, c3 = st.columns(3)
    min_oc = c1.slider(
        "Mínimo de ocorrências por episódio", 15, 300, 60, step=5, key="min_oc"
    )
    datas_ep = sorted(r["inicio"].dt.date.dropna().unique())
    dias_ep = c2.multiselect(
        "Dia do episódio", datas_ep, default=datas_ep, format_func=rotulo_data,
        key="dias_ep",
    )
    horas = c3.slider("Hora de início", 0, 23, (0, 23), key="horas_ep")
    ver_rec = st.toggle(
        "Recorrência por parada (em quantos dias a parada entrou em algum dos "
        "episódios filtrados)", value=True, key="rec",
    )
    r = r[
        (r["n_amostras"] >= min_oc)
        & r["inicio"].dt.date.isin(dias_ep)
        & r["inicio"].dt.hour.between(*horas)
    ].copy()

    x, lim = escala(r["atraso_medio_s"])
    r["cor"] = cores(x, alpha=60)
    r["borda"] = cores(x, alpha=220)
    r["tip"] = [
        f"episódio {rot}\n{rotulo_data(i.date())} {i:%Hh%M} → "
        f"{f:%d/%m %Hh%M}\n{milhar(n)} ocorrências · atraso médio {am:.0f} s · "
        f"p90 {p90:.0f} s\n{fl}\nlinhas {lts}"
        for rot, i, f, n, am, p90, fl, lts in zip(
            r["rotulo"], r["inicio"], r["fim"], r["n_amostras"], r["atraso_medio_s"],
            r["atraso_p90_s"], r["fluxo_principal"], r["linhas"],
        )
    ]
    camadas.append(camada(
        "PolygonLayer", data=r[["poligono", "cor", "borda", "tip"]],
        get_polygon="poligono", get_fill_color="cor", get_line_color="borda",
        line_width_min_pixels=1, stroked=True, filled=True,
        pickable=True, auto_highlight=True,
    ))

    ev = carregar_eventos(RAIZ)
    rec = pd.DataFrame()
    if ver_rec and not ev.empty and not r.empty:
        e = ev[ev["rotulo"].isin(r["rotulo"])]
        g = e.groupby("cp")
        rec = pd.DataFrame({
            "dias": g["data"].nunique(),
            "episodios": g["rotulo"].nunique(),
            "ocorrencias": g.size(),
            "atraso_mediano_s": g["valor_s"].median(),
            "lat": g["lat"].mean(),
            "lon": g["lon"].mean(),
        }).reset_index()
        rec["linhas"] = rec["cp"].map(linhas_por(e, "cp"))
        rec = com_catalogo(rec, paradas)
        xr, _ = escala(rec["dias"], limites=(1, max(len(dias_ep), 2)))
        rec["cor"] = cores(xr, alpha=235)
        rec["raio"] = 50 + 150 * np.sqrt(rec["ocorrencias"] / rec["ocorrencias"].max())
        rec["tip"] = [
            f"{nome}\nem episódio em {d} dia(s) · {ep} episódio(s)"
            f"\n{oc} ocorrências na amostra · atraso mediano {md:.0f} s"
            f"\nlinhas {lts}"
            for nome, d, ep, oc, md, lts in zip(
                rec["nome"], rec["dias"], rec["episodios"], rec["ocorrencias"],
                rec["atraso_mediano_s"], rec["linhas"],
            )
        ]
        camadas.append(camada(
            "ScatterplotLayer", data=rec[["lon", "lat", "cor", "raio", "tip"]],
            get_position=["lon", "lat"], get_radius="raio",
            radius_min_pixels=3, radius_max_pixels=14, get_fill_color="cor",
            stroked=True, get_line_color=[40, 40, 40], line_width_min_pixels=0.5,
            pickable=True, auto_highlight=True,
        ))

    cl1, cl2 = st.columns(2)
    with cl1:
        legenda("Episódios: atraso médio", lim, lambda v: f"{v:.0f} s")
    if not rec.empty:
        with cl2:
            legenda(
                "Paradas: dias com episódio", (1, max(len(dias_ep), 2)),
                lambda v: f"{v:.0f} dia(s)",
            )
    mapa(camadas)
    st.caption(
        f"{len(r)} episódios com esses filtros. O agrupamento rodou sobre uma amostra "
        f"de {milhar(len(ev))} ocorrências de atraso, então as contagens são da "
        "amostra, não do total."
    )

    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Onde se repete")
        if not rec.empty:
            st.dataframe(
                rec.sort_values(["dias", "ocorrencias"], ascending=False).head(15)[
                    ["nome", "linhas", "dias", "episodios", "ocorrencias", "atraso_mediano_s"]
                ],
                hide_index=True,
                column_config={
                    "nome": st.column_config.TextColumn("parada"),
                    "dias": st.column_config.NumberColumn("dias"),
                    "episodios": st.column_config.NumberColumn("episódios"),
                    "ocorrencias": st.column_config.NumberColumn("ocorrências"),
                    "atraso_mediano_s": st.column_config.NumberColumn(
                        "atraso mediano (s)", format="%.0f"
                    ),
                },
            )
    with t2:
        st.subheader("Episódios mais fortes")
        top = r.sort_values("atraso_medio_s", ascending=False).head(15).copy()
        top["quando"] = [
            f"{rotulo_data(i.date())} {i:%Hh%M}–{f:%Hh%M}"
            for i, f in zip(top["inicio"], top["fim"])
        ]
        st.dataframe(
            top[["quando", "n_amostras", "atraso_medio_s", "fluxo_principal"]],
            hide_index=True,
            column_config={
                "quando": st.column_config.TextColumn("quando"),
                "n_amostras": st.column_config.NumberColumn("ocorrências"),
                "atraso_medio_s": st.column_config.NumberColumn("atraso médio (s)", format="%.0f"),
                "fluxo_principal": st.column_config.TextColumn("fluxo principal"),
            },
        )


@st.fragment
def aba_regioes(todas_linhas):
    reg = carregar_regioes(RAIZ)
    if reg.empty:
        st.info("Sem `regioes.geojson`. Rode `python -m olhovivo regioes`.")
        return
    disponiveis = [a for a in ALGORITMOS if a in set(reg["algoritmo"])]
    alg = st.radio(
        "Algoritmo", disponiveis, format_func=ALGORITMOS.get, horizontal=True, key="alg"
    )
    st.caption(
        DESCRICOES[alg] + " Os agrupamentos usam a semana inteira; os filtros da "
        "barra lateral não se aplicam aqui."
    )
    r = reg[reg["algoritmo"] == alg].copy()
    camadas = [camada_tracados(tracados, todas_linhas, [1, 2])]
    if alg == "st-dbscan":
        _episodios(r, camadas)
    else:
        _regioes_espaciais(r, alg, camadas)


@st.fragment
def aba_fluxos(sel, filtros, linhas, sentidos):
    c1, c2 = st.columns(2)
    nome_od = c1.selectbox("Cor dos arcos", list(METRICAS), key="met_od")
    min_od = c2.number_input(
        "Mínimo de trechos por fluxo", 0, 50_000, 500, step=100, key="min_od"
    )
    campo, fmt, maior_melhor = METRICAS[nome_od]

    od = fluxos_do_recorte(RAIZ, filtros, sel)
    od = od[od["n"] >= min_od]
    geo = od.dropna(subset=["lat_o", "lon_o", "lat_d", "lon_d"]).copy()
    x, lim = escala(geo[campo], maior_melhor)
    geo["cor"] = cores(x, alpha=230)
    geo["largura"] = 2 + 10 * np.sqrt(geo["n"] / max(geo["n"].max(), 1))
    term = pd.concat([
        geo[["regiao_origem", "lat_o", "lon_o"]].set_axis(["nome", "lat", "lon"], axis=1),
        geo[["regiao_destino", "lat_d", "lon_d"]].set_axis(["nome", "lat", "lon"], axis=1),
    ]).groupby("nome", as_index=False).mean()
    term["tip"] = term["nome"]

    legenda(nome_od, lim, fmt, maior_melhor)
    mapa([
        camada_tracados(tracados, linhas, sentidos),
        camada(
            "ArcLayer",
            data=geo[["lon_o", "lat_o", "lon_d", "lat_d", "cor", "largura", "tip"]],
            get_source_position=["lon_o", "lat_o"], get_target_position=["lon_d", "lat_d"],
            get_source_color="cor", get_target_color="cor", get_width="largura",
            get_tilt=15, pickable=True, auto_highlight=True,
        ),
        camada(
            "ScatterplotLayer", data=term[["lon", "lat", "tip"]],
            get_position=["lon", "lat"], get_radius=250, radius_min_pixels=4,
            get_fill_color=[30, 41, 59, 230], pickable=True,
        ),
    ], pitch=40)
    st.caption(
        "Cada arco liga o terminal de origem ao de destino das linhas; a espessura é "
        "o número de trechos. Ida e volta aparecem como arcos separados."
    )
    st.dataframe(
        od.sort_values(campo, ascending=maior_melhor)[
            ["regiao_origem", "regiao_destino", "linhas", "n", "taxa",
             "atraso_p90_s", "atraso_medio_s", "vel_mediana"]
        ],
        hide_index=True, column_config=CFG_TABELA,
    )


@st.fragment
def aba_tempo(sel, filtros):
    nome_t = st.selectbox("Métrica", list(METRICAS), key="met_tempo")
    campo, fmt, maior_melhor = METRICAS[nome_t]
    minimo = 30

    def escala_cor(valores):
        _, (lo, hi) = escala(valores)
        return alt.Scale(scheme="redyellowgreen", reverse=not maior_melhor,
                         domain=[lo, hi], clamp=True)

    dica = [alt.Tooltip("n", title="trechos"),
            alt.Tooltip(f"{campo}:Q", title=nome_t, format=".2f")]

    hm = agrega_recorte(RAIZ, filtros, ("data", "hora"), sel)
    hm = hm[hm["n"] >= minimo].copy()
    hm["dia"] = [rotulo_data(d) for d in hm["data"]]
    ordem_dias = [rotulo_data(d) for d in sorted(hm["data"].unique())]
    st.subheader("Dia × hora")
    st.altair_chart(
        alt.Chart(hm.drop(columns="data")).mark_rect().encode(
            x=alt.X("hora:O", title="hora"),
            y=alt.Y("dia:N", sort=ordem_dias, title=None),
            color=alt.Color(f"{campo}:Q", scale=escala_cor(hm[campo]), title=nome_t),
            tooltip=["dia", "hora"] + dica,
        ).properties(height=280),
        width="stretch",
    )
    st.caption(
        f"Células com menos de {minimo} trechos ficam em branco — inclusive a pausa "
        "diária da coleta, das 01h às 04h."
    )

    a, b = st.columns(2)
    with a:
        st.subheader("Perfil ao longo do dia")
        perfil = agrega_recorte(RAIZ, filtros, ("tipo_dia", "hora"), sel)
        perfil = perfil[perfil["n"] >= minimo]
        st.altair_chart(
            alt.Chart(perfil).mark_line(point=True).encode(
                x=alt.X("hora:O", title="hora"),
                y=alt.Y(f"{campo}:Q", title=nome_t),
                color=alt.Color("tipo_dia:N", title="tipo de dia", sort=TIPOS_DIA),
                tooltip=["tipo_dia", "hora"] + dica,
            ).properties(height=300),
            width="stretch",
        )
    with b:
        st.subheader("Linha × faixa horária")
        lf = agrega_recorte(RAIZ, filtros, ("letreiro", "faixa"), sel)
        lf = lf[lf["n"] >= minimo].copy()
        lf["faixa_nome"] = lf["faixa"].astype(str).map(FAIXAS)
        lf["letreiro"] = lf["letreiro"].astype(str)
        lf["rotulo"] = [fmt(v) for v in lf[campo]]
        base = alt.Chart(lf).encode(
            x=alt.X("faixa_nome:N", sort=list(FAIXAS.values()), title=None),
            y=alt.Y("letreiro:N", title="linha"),
        )
        st.altair_chart(
            (
                base.mark_rect().encode(
                    color=alt.Color(f"{campo}:Q", scale=escala_cor(lf[campo]), legend=None),
                    tooltip=["letreiro", "faixa_nome"] + dica,
                )
                + base.mark_text(fontSize=11).encode(text="rotulo:N")
            ).properties(height=300),
            width="stretch",
        )

    st.subheader("Distribuição do desvio de tempo de percurso")
    st.altair_chart(
        alt.Chart(histograma(RAIZ, filtros, sel)).mark_bar().encode(
            x=alt.X("de:Q", bin="binned", title="atraso_s (s) — extremos somados às pontas"),
            x2="ate:Q",
            y=alt.Y("trechos:Q", title="trechos"),
            color=alt.condition(alt.datum.de >= 0, alt.value("#d73027"), alt.value("#1a9850")),
            tooltip=["de", "ate", "trechos"],
        ).properties(height=220),
        width="stretch",
    )
    st.caption(
        "A mediana é zero por construção. O que distingue os recortes é o tamanho da "
        "cauda à direita."
    )


@st.fragment
def aba_previsao(linhas, sentidos, faixas, tipos):
    if carregar_previsao(RAIZ) is None:
        st.info(
            "Sem `painel_previsao.parquet`. Gere com "
            "`python scripts/agrega_previsao.py <pasta dos dados>`."
        )
        return
    st.caption(
        "Medida complementar: a qualidade da previsão publicada pela SPTrans, comparando "
        "o horário prometido com a chegada reconstruída. Esta tabela está agregada por "
        "dia da semana, então o filtro de dias não se aplica."
    )
    c1, c2 = st.columns([2, 3])
    nome_p = c1.selectbox("Métrica", list(METRICAS_PREV), key="met_prev")
    hz = c2.pills(
        "Antecedência da previsão", HORIZONTES, selection_mode="multi",
        default=HORIZONTES, key="hz",
    )
    campo, fmt = METRICAS_PREV[nome_p]
    res = resumo_previsao(RAIZ, tuple(linhas), tuple(sentidos), tuple(faixas),
                          tuple(tipos), tuple(hz))
    if res is None:
        st.warning("Nenhuma previsão com esses filtros.")
        return
    (n_tot, mae, a5, ad), por_cp, ph = res

    k = st.columns(4)
    k[0].metric("Pares previsão × chegada", milhar(n_tot))
    k[1].metric("Erro absoluto médio", f"{mae:.0f} s")
    k[2].metric("Chegou ≥ 5 min depois", pct(a5))
    k[3].metric("Chegou ≥ 1 min antes", pct(ad))

    min_np = st.slider(
        "Mínimo de pares por parada", 100, 20_000, 1_000, step=100, key="min_np"
    )
    por_cp = por_cp[por_cp["n"] >= min_np].reset_index(drop=True)
    x, lim = escala(por_cp[campo])
    por_cp["cor"] = cores(x)
    legenda(nome_p, lim, fmt)
    mapa([
        camada_tracados(tracados, linhas, sentidos),
        camada(
            "ScatterplotLayer", data=por_cp[["lon", "lat", "cor", "tip"]],
            get_position=["lon", "lat"], get_radius=110, radius_min_pixels=3,
            get_fill_color="cor", stroked=True, get_line_color=[255, 255, 255],
            line_width_min_pixels=1, pickable=True, auto_highlight=True,
        ),
    ])

    a, b = st.columns(2)
    with a:
        st.subheader("Erro por antecedência")
        st.altair_chart(
            alt.Chart(ph).mark_line(point=True).encode(
                x=alt.X("horizonte:N", sort=HORIZONTES, title="antecedência"),
                y=alt.Y(f"{campo}:Q", title=nome_p),
                color=alt.Color("letreiro:N", title="linha"),
                tooltip=["letreiro", "horizonte", alt.Tooltip("n", title="pares"),
                         alt.Tooltip(f"{campo}:Q", title=nome_p, format=".2f")],
            ).properties(height=320),
            width="stretch",
        )
        st.caption("Quanto mais longe a chegada, maior o erro — por isso toda comparação "
                   "deve ser feita dentro da mesma antecedência.")
    with b:
        st.subheader("Paradas com pior previsão")
        st.dataframe(
            por_cp.sort_values(campo, ascending=False).head(15)[
                ["nome", "linhas", "n", "mae_s", "p_atraso5", "p_adiantado1"]
            ],
            hide_index=True,
            column_config={
                "nome": st.column_config.TextColumn("parada"),
                "n": st.column_config.NumberColumn("pares", format="localized"),
                "mae_s": st.column_config.NumberColumn("erro médio (s)", format="%.0f"),
                "p_atraso5": st.column_config.NumberColumn("≥ 5 min depois", format="percent"),
                "p_adiantado1": st.column_config.NumberColumn("≥ 1 min antes", format="percent"),
            },
        )


# --------------------------------------------------------------------------
# página
# --------------------------------------------------------------------------
st.set_page_config(page_title="Atraso dos ônibus em SP", page_icon="🚌", layout="wide")

raiz = pasta_dados()
if not (raiz / "derivado" / "percursos.parquet").exists():
    st.error(
        f"Não encontrei `{raiz / 'derivado' / 'percursos.parquet'}`. Rode "
        "`python -m olhovivo atraso` antes, ou aponte `OLHOVIVO_DADOS` para a pasta "
        "que tem `derivado/` e `catalogo/`."
    )
    st.stop()

RAIZ = str(raiz)
perc = carregar_percursos(RAIZ)
paradas = carregar_paradas(RAIZ)
tracados = carregar_tracados(RAIZ)
rotulos_linha = rotulos_das_linhas(RAIZ)
todas_linhas = sorted(perc["letreiro"].cat.categories)
datas_todas = list(perc["data"].cat.categories)

with st.sidebar:
    st.header("Filtros")
    linhas = st.pills(
        "Linhas", todas_linhas, selection_mode="multi", default=todas_linhas,
        key="f_linhas",
    )
    with st.expander("Itinerário de cada linha"):
        st.markdown("\n".join(
            f"- **{lt}** · {rotulos_linha.get(lt, '')}" for lt in todas_linhas
        ))
    sentidos = st.pills(
        "Sentido", [1, 2], selection_mode="multi", default=[1, 2],
        format_func=lambda s: f"sentido {s}", key="f_sentidos",
        help="1 = terminal principal → secundário; 2 = o inverso.",
    )
    faixas = st.pills(
        "Faixa horária", list(FAIXAS), selection_mode="multi", default=list(FAIXAS),
        format_func=FAIXAS.get, key="f_faixas",
    )
    tipos = st.pills(
        "Tipo de dia", TIPOS_DIA, selection_mode="multi", default=TIPOS_DIA,
        key="f_tipos",
    )
    datas = st.pills(
        "Dias", datas_todas, selection_mode="multi", default=datas_todas,
        format_func=rotulo_data, key="f_datas",
    )
    min_n = st.slider(
        "Mínimo de trechos por parada", 5, 300, 30, step=5,
        help="Paradas com menos observações ficam fora do mapa e dos rankings.",
    )
    st.divider()
    st.caption(f"Dados em `{raiz}`")

if not (linhas and sentidos and faixas and tipos and datas):
    st.warning("Selecione ao menos uma opção em cada filtro.")
    st.stop()

# a ordem da seleção não muda o recorte — ordena para o cache reconhecer
filtros = (
    tuple(sorted(linhas)), tuple(sorted(sentidos)), tuple(sorted(faixas)),
    tuple(sorted(tipos)), tuple(sorted(datas)),
)
sel = perc[
    perc["letreiro"].isin(linhas)
    & perc["sentido"].isin(sentidos)
    & perc["faixa"].isin(faixas)
    & perc["tipo_dia"].isin(tipos)
    & perc["data"].isin(datas)
]
if sel.empty:
    st.warning("Nenhum trecho com essa combinação de filtros.")
    st.stop()

ini, fim = perc.attrs["periodo"]
st.title("Atraso dos ônibus em São Paulo")
st.caption(
    f"{milhar(len(perc))} trechos observados entre {ini:%d/%m %Hh%M} e "
    f"{fim:%d/%m %Hh%M}, em {len(todas_linhas)} linhas nos dois sentidos · "
    "API Olho Vivo (SPTrans)"
)
n_trechos, n_viagens, taxa, p90, vel = indicadores(RAIZ, filtros, sel)
k = st.columns(5)
k[0].metric("Trechos", milhar(n_trechos))
k[1].metric("Viagens", milhar(n_viagens))
k[2].metric("Taxa de atraso", pct(taxa))
k[3].metric("Atraso p90", f"{p90:.0f} s")
k[4].metric("Velocidade mediana", f"{vel:.1f} km/h".replace(".", ","))

t_mapa, t_reg, t_od, t_tempo, t_prev = st.tabs([
    "Mapa do atraso", "Regiões (agrupamentos)", "Origem → destino",
    "Padrões no tempo", "Previsão da API",
])
with t_mapa:
    aba_mapa(sel, filtros, linhas, sentidos, min_n)
with t_reg:
    aba_regioes(todas_linhas)
with t_od:
    aba_fluxos(sel, filtros, linhas, sentidos)
with t_tempo:
    aba_tempo(sel, filtros)
with t_prev:
    aba_previsao(linhas, sentidos, faixas, tipos)
