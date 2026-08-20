"""
Engenharia de atributos para os modelos preditivos.

Produz tres artefatos, um para cada familia de modelo pedida:

* **tabela tabular** (`modelagem.parquet`) — uma linha por par previsao x
  chegada, com atributos temporais, espaciais, operacionais e defasados. E o
  que o XGBoost consome.
* **sequencias** (`sequencias_lstm.npz`) — series por (linha, parada) em bins
  de tempo regulares, no formato (amostras, passos, atributos). E o que a LSTM
  consome.
* **grafo** (`grafo.npz`) — paradas como nos, trechos consecutivos como arestas
  dirigidas, com tempo mediano de percurso como peso. E o que a GNN consome.

Uma coisa que costuma passar batido: a variavel `horizonte_s` (com quanta
antecedencia a previsao foi feita) e o atributo mais forte de todos, porque o
erro cresce com o horizonte. Modelar sem ela produz um numero bonito e sem
sentido. Ela esta aqui, e a avaliacao deve ser estratificada por faixa de
horizonte.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from .storage import Armazenamento

log = logging.getLogger("olhovivo.features")

# feriados nacionais + municipais de SP relevantes para a operacao
FERIADOS = {
    "01-01", "01-25", "04-21", "05-01", "07-09", "09-07",
    "10-12", "11-02", "11-15", "11-20", "12-25",
}


def _marcar_feriado(serie: pd.Series) -> pd.Series:
    return serie.dt.strftime("%m-%d").isin(FERIADOS)


# ---------------------------------------------------------------- percursos
def tabela_percursos(arm: Armazenamento) -> pd.DataFrame:
    """
    Tempo de percurso entre paradas consecutivas, por viagem.

    E aqui que aparece o "atraso" no sentido operacional: o trecho levou quanto
    tempo a mais do que costuma levar naquela faixa horaria?
    """
    ch = arm.ler_derivado("chegadas")
    if ch.empty:
        return ch
    ch["t_chegada"] = pd.to_datetime(ch["t_chegada"], utc=True)
    ch = ch.sort_values(["viagem_id", "ordem"])

    ch["t_anterior"] = ch.groupby("viagem_id", sort=False)["t_chegada"].shift(1)
    ch["ordem_anterior"] = ch.groupby("viagem_id", sort=False)["ordem"].shift(1)
    ch["cp_anterior"] = ch.groupby("viagem_id", sort=False)["cp"].shift(1)

    trechos = ch.dropna(subset=["t_anterior"]).copy()
    trechos = trechos[trechos["ordem"] - trechos["ordem_anterior"] == 1]
    trechos["tempo_percurso_s"] = (
        trechos["t_chegada"] - trechos["t_anterior"]
    ).dt.total_seconds()
    trechos = trechos[
        (trechos["tempo_percurso_s"] > 5) & (trechos["tempo_percurso_s"] < 3600)
    ]

    local = trechos["t_chegada"].dt.tz_convert("America/Sao_Paulo")
    trechos["hora"] = local.dt.hour
    trechos["dia_semana"] = local.dt.dayofweek

    base = (
        trechos.groupby(["cl", "ordem", "hora"])["tempo_percurso_s"]
        .median()
        .rename("percurso_tipico_s")
        .reset_index()
    )
    trechos = trechos.merge(base, on=["cl", "ordem", "hora"], how="left")
    trechos["desvio_percurso_s"] = (
        trechos["tempo_percurso_s"] - trechos["percurso_tipico_s"]
    )
    trechos["razao_percurso"] = (
        trechos["tempo_percurso_s"] / trechos["percurso_tipico_s"].replace(0, np.nan)
    )
    return trechos


# ------------------------------------------------------------------ tabular
def montar_tabela(arm: Armazenamento, min_confianca: float = 0.4) -> pd.DataFrame:
    """Tabela de modelagem para o XGBoost."""
    df = arm.ler_derivado("previsao_realizado")
    if df.empty:
        raise RuntimeError("rode `python -m olhovivo casar` antes de gerar atributos")

    df = df[df["confianca"] >= min_confianca].copy()
    for coluna in ("ts_coleta", "t_previsto", "t_chegada"):
        df[coluna] = pd.to_datetime(df[coluna], utc=True)

    local = df["t_previsto"].dt.tz_convert("America/Sao_Paulo")
    minuto_dia = local.dt.hour * 60 + local.dt.minute

    df["minuto_dia"] = minuto_dia
    df["sin_dia"] = np.sin(2 * np.pi * minuto_dia / 1440)
    df["cos_dia"] = np.cos(2 * np.pi * minuto_dia / 1440)
    df["sin_semana"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["cos_semana"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
    df["fim_de_semana"] = df["dia_semana"].isin([5, 6]).astype("int8")
    df["feriado"] = _marcar_feriado(local).astype("int8")

    # posicao da parada no itinerario
    vinculos = arm.ler_catalogo("linha_parada")
    if not vinculos.empty:
        extensao = vinculos.groupby("cl")["s_m"].transform("max")
        vinculos = vinculos.assign(
            fracao_percurso=vinculos["s_m"] / extensao.replace(0, np.nan)
        )
        df = df.merge(
            vinculos[["cl", "cp", "ordem", "s_m", "fracao_percurso"]],
            on=["cl", "cp"],
            how="left",
            suffixes=("", "_cat"),
        )

    # carga da linha no instante: quantos veiculos distintos a API listou
    df["bin5"] = df["ts_coleta"].dt.floor("5min")
    carga = (
        df.groupby(["cl", "bin5"])["prefixo"]
        .nunique()
        .rename("veiculos_ativos")
        .reset_index()
    )
    df = df.merge(carga, on=["cl", "bin5"], how="left")

    # atributos defasados: como estava o atraso ANTES desta chegada
    df = df.sort_values("t_chegada")
    df["erro_anterior_parada_s"] = df.groupby(["cl", "cp"], sort=False)["erro_s"].shift(1)
    df["erro_media3_parada_s"] = df.groupby(["cl", "cp"], sort=False)["erro_s"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["erro_anterior_linha_s"] = df.groupby("cl", sort=False)["erro_s"].shift(1)

    # desvio de percurso do trecho anterior
    percursos = tabela_percursos(arm)
    if not percursos.empty:
        chave = percursos[["viagem_id", "cp", "desvio_percurso_s", "razao_percurso"]]
        df = df.merge(chave, on=["viagem_id", "cp"], how="left")

    df["alvo_erro_s"] = df["erro_s"]
    df["alvo_atrasado"] = (df["erro_s"] >= 300).astype("int8")

    destino = arm.raiz / "derivado" / "modelagem.parquet"
    df.to_parquet(destino, index=False)
    log.info("tabela de modelagem: %d linhas -> %s", len(df), destino)
    return df


COLUNAS_MODELO = [
    "horizonte_s",
    "minuto_dia",
    "sin_dia",
    "cos_dia",
    "sin_semana",
    "cos_semana",
    "fim_de_semana",
    "feriado",
    "dia_semana",
    "hora",
    "sentido",
    "ordem",
    "s_m",
    "fracao_percurso",
    "lat",
    "lon",
    "headway_obs_s",
    "veiculos_ativos",
    "erro_anterior_parada_s",
    "erro_media3_parada_s",
    "erro_anterior_linha_s",
    "desvio_percurso_s",
    "razao_percurso",
]


# --------------------------------------------------------------- sequencias
def montar_sequencias(
    arm: Armazenamento,
    bin_min: int = 15,
    passos: int = 8,
    min_pontos: int = 200,
) -> dict:
    """
    Series regulares por (linha, parada) para a LSTM.

    Cada amostra: os `passos` bins anteriores de erro medio, headway e volume,
    para prever o erro medio do bin seguinte.
    """
    df = arm.ler_derivado("previsao_realizado")
    if df.empty:
        raise RuntimeError("sem dataset casado")

    df["t_chegada"] = pd.to_datetime(df["t_chegada"], utc=True)
    df["bin"] = df["t_chegada"].dt.floor(f"{bin_min}min")

    agregado = (
        df.groupby(["cl", "cp", "bin"])
        .agg(
            erro_medio=("erro_s", "mean"),
            erro_max=("erro_s", "max"),
            headway=("headway_obs_s", "median"),
            volume=("prefixo", "nunique"),
        )
        .reset_index()
    )

    X: list[np.ndarray] = []
    y: list[float] = []
    chaves: list[tuple[int, int, pd.Timestamp]] = []

    for (cl, cp), grupo in agregado.groupby(["cl", "cp"], sort=False):
        if len(grupo) < passos + 1:
            continue
        g = grupo.sort_values("bin")
        # reindexa em grade regular para nao pular buracos silenciosamente
        g = g.set_index("bin").asfreq(f"{bin_min}min")
        g[["erro_medio", "erro_max"]] = g[["erro_medio", "erro_max"]].ffill(limit=2)
        g = g.fillna({"volume": 0, "headway": g["headway"].median()})
        g = g.dropna(subset=["erro_medio"])
        if len(g) < passos + 1:
            continue

        valores = g[["erro_medio", "erro_max", "headway", "volume"]].to_numpy("float32")
        hora = g.index.tz_convert("America/Sao_Paulo").hour.to_numpy("float32") / 23.0
        dia = g.index.tz_convert("America/Sao_Paulo").dayofweek.to_numpy("float32") / 6.0
        valores = np.column_stack([valores, hora, dia])

        for i in range(len(valores) - passos):
            X.append(valores[i : i + passos])
            y.append(float(valores[i + passos, 0]))
            chaves.append((int(cl), int(cp), g.index[i + passos]))

    if len(X) < min_pontos:
        log.warning(
            "apenas %d sequencias montadas — pouco para treinar LSTM. "
            "Com bins de %d min, cada (linha, parada) precisa de %d bins seguidos "
            "com chegada; aumente o periodo coletado ou o tamanho do bin.",
            len(X),
            bin_min,
            passos + 1,
        )

    n_atributos = 6
    Xa = (
        np.asarray(X, dtype="float32")
        if X
        else np.zeros((0, passos, n_atributos), dtype="float32")
    )
    ya = np.asarray(y, dtype="float32")
    destino = arm.raiz / "derivado" / "sequencias_lstm.npz"
    np.savez_compressed(
        destino,
        X=Xa,
        y=ya,
        cl=np.array([c[0] for c in chaves]),
        cp=np.array([c[1] for c in chaves]),
        bin=np.array([c[2].value for c in chaves]),
        atributos=np.array(["erro_medio", "erro_max", "headway", "volume", "hora", "dia"]),
    )
    log.info("sequencias LSTM: %s -> %s", Xa.shape, destino)
    return {"amostras": int(Xa.shape[0]), "passos": passos, "atributos": int(Xa.shape[-1])}


# -------------------------------------------------------------------- grafo
def montar_grafo(arm: Armazenamento, raio_vizinhanca_m: float = 400.0) -> dict:
    """
    Grafo da rede para a GNN.

    * nos: paradas que aparecem no dataset casado
    * arestas de itinerario: parada k -> parada k+1 de cada linha, peso = tempo
      mediano de percurso observado
    * arestas de vizinhanca: paradas a menos de `raio_vizinhanca_m` uma da
      outra, mesmo sem linha em comum — e assim que o congestionamento se
      propaga entre corredores paralelos
    * atributos do no: estatisticas de atraso, headway, volume, posicao
    * alvo do no: erro mediano
    """
    from . import geo

    df = arm.ler_derivado("previsao_realizado")
    if df.empty:
        raise RuntimeError("sem dataset casado")

    nos = (
        df.groupby("cp")
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            n=("erro_s", "size"),
            erro_medio=("erro_s", "mean"),
            erro_mediano=("erro_s", "median"),
            erro_desvio=("erro_s", "std"),
            headway=("headway_obs_s", "median"),
            linhas=("cl", "nunique"),
        )
        .reset_index()
        .fillna(0.0)
    )
    indice = {int(cp): i for i, cp in enumerate(nos["cp"])}

    arestas: list[tuple[int, int]] = []
    pesos: list[float] = []

    percursos = tabela_percursos(arm)
    if not percursos.empty:
        med = (
            percursos.groupby(["cp_anterior", "cp"])["tempo_percurso_s"]
            .median()
            .reset_index()
        )
        for r in med.itertuples():
            a, b = indice.get(int(r.cp_anterior)), indice.get(int(r.cp))
            if a is not None and b is not None:
                arestas.append((a, b))
                pesos.append(float(r.tempo_percurso_s))

    # arestas de vizinhanca espacial
    proj = geo.obter_projecao()
    x, y = proj.para_xy(nos["lat"].to_numpy(), nos["lon"].to_numpy())
    coords = np.column_stack([np.asarray(x), np.asarray(y)])
    try:
        from scipy.spatial import cKDTree

        arvore = cKDTree(coords)
        for a, b in arvore.query_pairs(r=raio_vizinhanca_m):
            arestas.append((a, b))
            pesos.append(0.0)
            arestas.append((b, a))
            pesos.append(0.0)
    except ImportError:
        log.warning("scipy ausente: grafo sem arestas de vizinhanca espacial")

    atributos = nos[
        ["lat", "lon", "n", "erro_medio", "erro_desvio", "headway", "linhas"]
    ].to_numpy("float32")
    alvo = nos["erro_mediano"].to_numpy("float32")
    edge_index = np.asarray(arestas, dtype="int64").T if arestas else np.zeros((2, 0), "int64")

    destino = arm.raiz / "derivado" / "grafo.npz"
    np.savez_compressed(
        destino,
        x=atributos,
        y=alvo,
        edge_index=edge_index,
        edge_weight=np.asarray(pesos, dtype="float32"),
        cp=nos["cp"].to_numpy("int64"),
        atributos=np.array(
            ["lat", "lon", "n", "erro_medio", "erro_desvio", "headway", "linhas"]
        ),
    )
    log.info("grafo: %d nos, %d arestas -> %s", len(nos), edge_index.shape[1], destino)
    return {"nos": int(len(nos)), "arestas": int(edge_index.shape[1])}
