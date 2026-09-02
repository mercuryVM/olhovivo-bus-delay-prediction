"""
Variavel de atraso conforme a definicao do estudo.

    "O atraso sera definido como o desvio entre o tempo de percurso observado
     em um trecho e um tempo de referencia estimado para o mesmo trecho,
     sentido e faixa horaria."

Duas formulacoes, como o metodo pede:

* **continua** — `atraso_s`, para os modelos de regressao;
* **binaria** — `atrasado`, por limiar PERCENTUAL sobre o tempo de referencia,
  para as metricas de classificacao e para estimar a probabilidade de atraso.

Nao confundir com `erro_s` de `casamento.py`. Aquele mede a qualidade da
PREVISAO publicada pela SPTrans; este mede a irregularidade da OPERACAO. Sao
grandezas distintas, e a do estudo e esta.

Tempo de referencia
-------------------
A programacao de ciclos da Prefeitura e a fonte preferencial, mas nao esta
disponivel nesta base; usa-se entao a alternativa prevista no metodo — a
mediana observada na propria semana de coleta, por (linha, trecho, faixa
horaria) — robusta a valores extremos. Trechos com menos de
`min_amostras_referencia` observacoes ficam sem referencia e sao excluidos, para
nao comparar uma viagem contra ela mesma.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import geo
from .casamento import faixa_horaria
from .coleta import TZ_SP
from .storage import Armazenamento

log = logging.getLogger("olhovivo.atraso")

FAIXAS_PICO = {"pico_manha", "pico_tarde"}


def _regioes_das_linhas(catalogo: pd.DataFrame) -> pd.DataFrame:
    """
    Terminais de origem e destino de cada linha/sentido.

    No sentido 1 a linha vai do terminal principal ao secundario; no sentido 2,
    o inverso. Os campos `tp`/`ts` da API sao fixos no par e NAO se invertem
    sozinhos, entao a troca precisa ser feita aqui.
    """
    if catalogo.empty:
        return pd.DataFrame()
    c = catalogo.copy()
    sentido2 = c["sentido"] == 2
    c["regiao_origem"] = np.where(
        sentido2, c["terminal_secundario"], c["terminal_principal"]
    )
    c["regiao_destino"] = np.where(
        sentido2, c["terminal_principal"], c["terminal_secundario"]
    )
    return c[["cl", "regiao_origem", "regiao_destino"]]


def construir(
    cfg,
    arm: Armazenamento,
    min_confianca: float = 0.4,
) -> dict:
    """Monta a tabela de trechos percorridos com a variavel de atraso."""
    limiar_pct = float(cfg.get("atraso.limiar_percentual", 0.20))
    min_ref = int(cfg.get("atraso.min_amostras_referencia", 5))
    resolucao_h3 = int(cfg.get("geo.h3_resolucao", 9))

    ch = arm.ler_derivado("chegadas")
    if ch.empty:
        raise RuntimeError("rode `python -m olhovivo chegadas` antes")

    ch = ch[ch["confianca"] >= min_confianca].copy()
    ch["t_chegada"] = pd.to_datetime(ch["t_chegada"], utc=True)
    ch = ch.sort_values(["viagem_id", "ordem"])

    # -- trechos entre paradas CONSECUTIVAS da mesma viagem -----------------
    g = ch.groupby("viagem_id", sort=False)
    ch["t_origem"] = g["t_chegada"].shift(1)
    ch["ordem_origem"] = g["ordem"].shift(1)
    ch["cp_origem"] = g["cp"].shift(1)
    ch["lat_origem"] = g["lat"].shift(1)
    ch["lon_origem"] = g["lon"].shift(1)

    t = ch.dropna(subset=["t_origem"]).copy()
    t = t[t["ordem"] - t["ordem_origem"] == 1]
    t = t.rename(columns={"cp": "cp_destino", "ordem": "ordem_destino"})
    t["tempo_percurso_s"] = (t["t_chegada"] - t["t_origem"]).dt.total_seconds()
    t = t[(t["tempo_percurso_s"] > 5) & (t["tempo_percurso_s"] < 3600)]
    if t.empty:
        raise RuntimeError("nenhum trecho valido — periodo curto demais?")

    t["trecho"] = (
        t["cl"].astype(str) + ":" + t["ordem_origem"].astype("int64").astype(str)
    )

    # -- extensao do trecho, pela abscissa curvilinea -----------------------
    vinculos = arm.ler_catalogo("linha_parada")
    if len(vinculos):
        s = vinculos[["cl", "cp", "s_m"]]
        t = t.merge(
            s.rename(columns={"cp": "cp_destino", "s_m": "s_destino"}),
            on=["cl", "cp_destino"], how="left",
        ).merge(
            s.rename(columns={"cp": "cp_origem", "s_m": "s_origem"}),
            on=["cl", "cp_origem"], how="left",
        )
        t["extensao_m"] = t["s_destino"] - t["s_origem"]
        t["velocidade_kmh"] = (
            t["extensao_m"] / t["tempo_percurso_s"].replace(0, np.nan) * 3.6
        )

    # -- recortes temporais -------------------------------------------------
    local = t["t_origem"].dt.tz_convert(TZ_SP)
    t["hora"] = local.dt.hour.astype("int8")
    t["dia_semana"] = local.dt.dayofweek.astype("int8")
    t["faixa"] = [faixa_horaria(h) for h in t["hora"]]
    t["pico"] = t["faixa"].isin(FAIXAS_PICO).astype("int8")
    t["fim_de_semana"] = t["dia_semana"].isin([5, 6]).astype("int8")

    # -- tempo de REFERENCIA: mediana por (linha, trecho, faixa) ------------
    ref = (
        t.groupby(["cl", "ordem_origem", "faixa"])["tempo_percurso_s"]
        .agg(percurso_tipico_s="median", n_referencia="size")
        .reset_index()
    )
    t = t.merge(ref, on=["cl", "ordem_origem", "faixa"], how="left")

    antes = len(t)
    t = t[t["n_referencia"] >= min_ref]
    log.info(
        "referencia: %d de %d trechos com ao menos %d observacoes",
        len(t), antes, min_ref,
    )
    if t.empty:
        raise RuntimeError(
            f"nenhum trecho atingiu {min_ref} observacoes — colete por mais tempo "
            "ou reduza atraso.min_amostras_referencia"
        )

    # -- a variavel de atraso, nas duas formulacoes -------------------------
    t["atraso_s"] = t["tempo_percurso_s"] - t["percurso_tipico_s"]
    t["atraso_rel"] = t["atraso_s"] / t["percurso_tipico_s"].replace(0, np.nan)
    t["atrasado"] = (t["atraso_rel"] >= limiar_pct).astype("int8")

    # -- regularidade: intervalo ate o veiculo anterior no destino ----------
    t = t.sort_values(["cl", "cp_destino", "t_chegada"])
    t["headway_s"] = (
        t.groupby(["cl", "cp_destino"], sort=False)["t_chegada"]
        .diff().dt.total_seconds()
    )

    # -- regioes de origem e destino da LINHA (terminais) -------------------
    regioes = _regioes_das_linhas(arm.ler_catalogo("linhas"))
    if len(regioes):
        t = t.merge(regioes, on="cl", how="left")

    # -- localizacao do trecho ---------------------------------------------
    t["lat"] = (t["lat_origem"] + t["lat"]) / 2
    t["lon"] = (t["lon_origem"] + t["lon"]) / 2
    t["celula"] = [
        geo.celula(float(a), float(b), resolucao_h3)
        for a, b in zip(t["lat"], t["lon"])
    ]

    colunas = [
        "cl", "letreiro", "sentido", "prefixo", "viagem_id", "trecho",
        "cp_origem", "cp_destino", "ordem_origem", "ordem_destino",
        "t_origem", "t_chegada", "tempo_percurso_s", "percurso_tipico_s",
        "n_referencia", "atraso_s", "atraso_rel", "atrasado",
        "headway_s", "extensao_m", "velocidade_kmh",
        "hora", "dia_semana", "faixa", "pico", "fim_de_semana",
        "regiao_origem", "regiao_destino", "lat", "lon", "celula",
        "confianca",
    ]
    saida = t[[c for c in colunas if c in t.columns]].copy()
    for c in ("cp_origem", "cp_destino", "ordem_origem", "ordem_destino"):
        if c in saida:
            saida[c] = saida[c].astype("int64")

    arm.salvar_tabela("percursos", saida.to_dict("records"))

    if arm.mongo:
        docs = [
            {
                **r,
                "loc": geo.ponto_geojson(r["lat"], r["lon"]),
                "t_origem": r["t_origem"].to_pydatetime(),
                "t_chegada": r["t_chegada"].to_pydatetime(),
            }
            for r in saida.to_dict("records")
        ]
        arm.mongo.substituir_colecao("percursos", docs)

    resumo = {
        "trechos": int(len(saida)),
        "viagens": int(saida["viagem_id"].nunique()),
        "linhas": int(saida["cl"].nunique()),
        "limiar_percentual": limiar_pct,
        "taxa_atraso": round(float(saida["atrasado"].mean()), 4),
        "atraso_mediano_s": round(float(saida["atraso_s"].median()), 1),
        "atraso_p90_s": round(float(np.percentile(saida["atraso_s"], 90)), 1),
        "percurso_tipico_mediano_s": round(float(saida["percurso_tipico_s"].median()), 1),
        "velocidade_mediana_kmh": (
            round(float(saida["velocidade_kmh"].median()), 1)
            if "velocidade_kmh" in saida else None
        ),
    }
    arm.eventos.registrar("atraso_construido", **resumo)
    log.info("atraso: %s", resumo)
    return resumo


COLUNAS_MODELO = [
    "hora", "dia_semana", "pico", "fim_de_semana",
    "ordem_origem", "extensao_m", "percurso_tipico_s",
    "headway_s", "velocidade_kmh", "sentido", "lat", "lon",
]


def por_regiao(arm: Armazenamento, min_amostras: int = 30) -> pd.DataFrame:
    """
    Atraso agregado por par (regiao de origem, regiao de destino).

    E o cruzamento que o estudo pede: quais FLUXOS concentram atraso, e nao
    apenas quais pontos do mapa.
    """
    df = arm.ler_derivado("percursos")
    if df.empty:
        return df
    agrupado = (
        df.groupby(["regiao_origem", "regiao_destino"], dropna=False)
        .agg(
            n=("atraso_s", "size"),
            linhas=("cl", "nunique"),
            atraso_medio_s=("atraso_s", "mean"),
            atraso_mediano_s=("atraso_s", "median"),
            atraso_p90_s=("atraso_s", lambda x: float(np.percentile(x, 90))),
            prob_atraso=("atrasado", "mean"),
            velocidade_mediana_kmh=("velocidade_kmh", "median"),
        )
        .reset_index()
    )
    agrupado = agrupado[agrupado["n"] >= min_amostras]
    return agrupado.sort_values("prob_atraso", ascending=False)
