"""
Atributos dos trechos a partir de camadas do GeoSampa (Prefeitura de Sao Paulo).

As camadas ficam em <raiz>/externo/geosampa/<camada>.geojson, baixadas do WFS
publico do GeoSampa em EPSG:4326:

* semaforo: cruzamentos semaforizados (carga de 2018; so a posicao, sem os
  tempos do sinal)
* corredor_onibus_v2: corredores de onibus (atualizados em 10/2025)
* faixa_onibus_v2: faixas exclusivas
* equipamento_feira_livre: feiras livres, com dia da semana e horario

Por par de paradas das linhas estudadas (a geometria de frota.geometria):

* geo_semaforos e geo_semaforos_km: cruzamentos a ate 20 m do tracado do par
* geo_corredor e geo_faixa: fracao do tracado a ate 15 m de um corredor ou de
  uma faixa. Sem o sentido: numa avenida de mao dupla, a faixa de um lado conta
  para os dois.

Por passagem, no dia e na hora local de `t_origem`:

* geo_faixa_ativa: a fracao do tracado em faixa que esta valendo naquela hora
  (muitas so valem no pico, como "2a a 6a das 06-09h")
* geo_feira_dia: feiras a ate 200 m do tracado naquele dia da semana
* geo_feira_agora: dessas, as que estao no horario de funcionamento. A camada
  nao traz o horario, entao vale a janela tipica de 6h as 15h

Sao atributos fixos ou de calendario: nada depende do que aconteceu depois de
`t_origem`.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import frota

log = logging.getLogger("olhovivo.geosampa")

COLUNAS_GEO = [
    "geo_semaforos", "geo_semaforos_km", "geo_corredor", "geo_faixa", "geo_faixa_ativa",
    "geo_feira_dia", "geo_feira_agora",
]
CAMADAS = ["semaforo", "corredor_onibus_v2", "faixa_onibus_v2", "equipamento_feira_livre"]
RAIO_SEMAFORO_M = 20.0
RAIO_FAIXA_M = 15.0
RAIO_FEIRA_M = 200.0
PASSO_LINHA_M = 5.0
# sem horario legivel: a janela tipica de uma feira, com montagem e desmontagem
HORARIO_FEIRA_PADRAO = (6.0, 15.0)
DIAS = {"SEG": 0, "TER": 1, "QUA": 2, "QUI": 3, "SEX": 4, "SAB": 5, "DOM": 6}
FUSO = "America/Sao_Paulo"
CHAVE = ["cp_origem", "cp_destino"]


def _ler(pasta: Path, camada: str) -> list[dict]:
    arq = pasta / f"{camada}.geojson"
    if not arq.exists():
        log.warning("camada ausente: %s", arq)
        return []
    return [f for f in json.loads(arq.read_text(encoding="utf-8"))["features"] if f.get("geometry")]


def _pontos(feicoes: list[dict], proj) -> np.ndarray:
    lonlat = np.array([f["geometry"]["coordinates"][:2] for f in feicoes], dtype="float64")
    if not len(lonlat):
        return np.empty((0, 2))
    x, y = proj.para_xy(lonlat[:, 1], lonlat[:, 0])
    return np.c_[np.asarray(x), np.asarray(y)]


def _linhas(feicoes: list[dict], proj) -> tuple[np.ndarray, np.ndarray]:
    """Pontos a cada PASSO_LINHA_M ao longo das linhas da camada e a feicao de cada um."""
    partes, ids = [], []
    for i, f in enumerate(feicoes):
        g = f["geometry"]
        trechos = [g["coordinates"]] if g["type"] == "LineString" else (
            g["coordinates"] if g["type"] == "MultiLineString" else [])
        for c in trechos:
            c = np.asarray(c, dtype="float64")
            if len(c) < 2:
                continue
            x, y = proj.para_xy(c[:, 1], c[:, 0])
            x, y = np.asarray(x), np.asarray(y)
            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
            sk = np.append(np.arange(0, s[-1], PASSO_LINHA_M), s[-1])
            partes.append(np.c_[np.interp(sk, s, x), np.interp(sk, s, y)])
            ids.append(np.full(len(sk), i))
    if not partes:
        return np.empty((0, 2)), np.empty(0, dtype="int64")
    return np.concatenate(partes), np.concatenate(ids)


def _sem_acento(texto) -> str:
    return unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode().upper()


def _dias(texto) -> set[int]:
    t = _sem_acento(texto)
    return {d for chave, d in DIAS.items() if re.search(rf"\b{chave}", t)}


def _horario(texto) -> tuple[float, float]:
    horas = [int(h) + int(m or 0) / 60
             for h, m in re.findall(r"(\d{1,2})\s*(?:[:H]\s*(\d{2}))?", _sem_acento(texto)) if int(h) < 24]
    if len(horas) >= 2 and min(horas) < max(horas):
        return min(horas), max(horas)
    return HORARIO_FEIRA_PADRAO


def _janelas_faixa(texto) -> list[tuple[int, float, float]]:
    """(dia, hora inicial, hora final) em que a faixa vale, de textos como
    "2a a 6a das 06-09h e das 17-20h; Sab. das 06-14h" ou "INTEGRAL"."""
    t = _sem_acento(texto)
    sempre = [(d, 0.0, 24.0) for d in range(7)]
    if not t.strip() or "INTEGRAL" in t:
        return sempre
    janelas = []
    for parte in t.split(";"):
        dias = set(range(5)) if re.search(r"2\S*\s*A\s*6", parte) else set()
        if "SAB" in parte:
            dias.add(5)
        if "DOM" in parte:
            dias.add(6)
        for h0, m0, h1, m1 in re.findall(r"(\d{1,2})(?::(\d{2}))?\s*-\s*(\d{1,2})(?::(\d{2}))?\s*H", parte):
            a, b = int(h0) + int(m0 or 0) / 60, int(h1) + int(m1 or 0) / 60
            janelas += [(d, a, b) for d in (dias or set(range(5)))]
    return janelas or sempre


def trechos(arm, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atributos fixos por par de paradas e a agenda das feiras perto de cada par."""
    pasta = arm.raiz / "externo" / "geosampa"
    proj = frota._projecao(cfg)
    pares, pontos = frota.geometria(arm, cfg)
    xy = np.c_[pontos["x"], pontos["y"]]
    par = pontos["par"]
    arvore = cKDTree(xy)
    saida = pares[["par", *CHAVE, "extensao_m"]].copy()

    sem = _pontos(_ler(pasta, "semaforo"), proj)
    contagem = np.zeros(len(pares))
    for idx in arvore.query_ball_point(sem, r=RAIO_SEMAFORO_M):
        for p in np.unique(par[idx]):
            contagem[p] += 1
    saida["geo_semaforos"] = contagem[saida["par"].to_numpy()]
    saida["geo_semaforos_km"] = saida["geo_semaforos"] / (saida["extensao_m"] / 1000).clip(lower=0.05)

    pontos_por_par = pd.Series(par).value_counts()
    faixas = pd.DataFrame(columns=["par", "dia", "h_ini", "h_fim", "fracao"])
    for nome, camada in (("geo_corredor", "corredor_onibus_v2"), ("geo_faixa", "faixa_onibus_v2")):
        feicoes = _ler(pasta, camada)
        lin, fid = _linhas(feicoes, proj)
        if not len(lin):
            saida[nome] = np.nan
            continue
        dist, j = cKDTree(lin).query(xy, k=1, distance_upper_bound=RAIO_FAIXA_M)
        perto = np.isfinite(dist)
        fracao = pd.Series(perto.astype("float64")).groupby(par).mean()
        saida[nome] = fracao.reindex(saida["par"]).to_numpy()
        if nome == "geo_faixa":
            # fracao de cada par sobre cada faixa, para somar so as que valem na hora
            pf = pd.DataFrame({"par": par[perto], "faixa": fid[j[perto]]})
            pf = pf.groupby(["par", "faixa"]).size().rename("n").reset_index()
            pf["fracao"] = pf["n"] / pontos_por_par.reindex(pf["par"]).to_numpy()
            janelas = {i: _janelas_faixa(f["properties"].get("tx_horario_funcionamento"))
                       for i, f in enumerate(feicoes)}
            faixas = pd.DataFrame([{"par": int(r.par), "dia": d, "h_ini": a, "h_fim": b, "fracao": r.fracao}
                                   for r in pf.itertuples() for d, a, b in janelas[int(r.faixa)]],
                                  columns=["par", "dia", "h_ini", "h_fim", "fracao"])

    feiras = _ler(pasta, "equipamento_feira_livre")
    agenda = []
    if feiras:
        fxy = _pontos(feiras, proj)
        for f, idx in zip(feiras, arvore.query_ball_point(fxy, r=RAIO_FEIRA_M)):
            if not idx:
                continue
            p = f["properties"]
            h_ini, h_fim = _horario(p.get("tx_horario_funcionamento"))
            for dia in _dias(p.get("tx_dia_funcionamento")):
                for q in np.unique(par[idx]):
                    agenda.append({"par": int(q), "dia": dia, "h_ini": h_ini, "h_fim": h_fim})
    agenda = pd.DataFrame(agenda, columns=["par", "dia", "h_ini", "h_fim"])
    agenda = agenda.merge(pares[["par", *CHAVE]], on="par").drop(columns="par")
    faixas = faixas.merge(pares[["par", *CHAVE]], on="par").drop(columns="par")
    log.info("geosampa: %d pares, %d semaforos, %d feiras, %d pares com feira, %d com faixa",
             len(saida), len(sem), len(feiras), agenda[CHAVE].drop_duplicates().shape[0],
             faixas[CHAVE].drop_duplicates().shape[0])
    return saida.drop(columns="par"), agenda, faixas


def gerar(arm, cfg) -> Path:
    """Grava derivado/geosampa_{trechos,feiras,faixas}.parquet."""
    fixos, agenda, faixas = trechos(arm, cfg)
    destino = arm.raiz / "derivado"
    fixos.to_parquet(destino / "geosampa_trechos.parquet", index=False)
    agenda.to_parquet(destino / "geosampa_feiras.parquet", index=False)
    faixas.to_parquet(destino / "geosampa_faixas.parquet", index=False)
    return destino


def atributos(df: pd.DataFrame, raiz: Path) -> pd.DataFrame:
    """Colunas de `COLUNAS_GEO` no indice de `df` (percursos com cp e t_origem)."""
    deriv = Path(raiz) / "derivado"
    saida = pd.DataFrame(np.nan, index=df.index, columns=COLUNAS_GEO)
    arq = deriv / "geosampa_trechos.parquet"
    if not arq.exists():
        return saida
    n = len(df)
    base = pd.DataFrame({"i": np.arange(n),
                         "cp_origem": df["cp_origem"].astype("int64").to_numpy(),
                         "cp_destino": df["cp_destino"].astype("int64").to_numpy()})
    fixos = pd.read_parquet(arq).astype({"cp_origem": "int64", "cp_destino": "int64"})
    m = base.merge(fixos, on=CHAVE, how="left").sort_values("i")
    for c in ("geo_semaforos", "geo_semaforos_km", "geo_corredor", "geo_faixa"):
        saida[c] = m[c].to_numpy()

    local = pd.to_datetime(df["t_origem"], utc=True).dt.tz_convert(FUSO)
    base["dia"] = local.dt.dayofweek.to_numpy()
    base["hora"] = (local.dt.hour + local.dt.minute / 60).to_numpy()
    arq_feiras = deriv / "geosampa_feiras.parquet"
    agenda = pd.read_parquet(arq_feiras) if arq_feiras.exists() else pd.DataFrame(
        columns=[*CHAVE, "dia", "h_ini", "h_fim"])
    j = base.merge(agenda.astype({"cp_origem": "int64", "cp_destino": "int64", "dia": "int64"}),
                   on=[*CHAVE, "dia"], how="inner")
    no_dia = j.groupby("i").size()
    agora = j[(j["hora"] >= j["h_ini"]) & (j["hora"] <= j["h_fim"])].groupby("i").size()
    saida["geo_feira_dia"] = no_dia.reindex(range(n), fill_value=0).to_numpy()
    saida["geo_feira_agora"] = agora.reindex(range(n), fill_value=0).to_numpy()

    arq_faixas = deriv / "geosampa_faixas.parquet"
    if arq_faixas.exists():
        fx = pd.read_parquet(arq_faixas).astype({"cp_origem": "int64", "cp_destino": "int64", "dia": "int64"})
        j = base.merge(fx, on=[*CHAVE, "dia"], how="inner")
        ativa = j[(j["hora"] >= j["h_ini"]) & (j["hora"] < j["h_fim"])].groupby("i")["fracao"].sum()
        saida["geo_faixa_ativa"] = ativa.reindex(range(n), fill_value=0).clip(upper=1).to_numpy()
    # par fora da geometria: sem atributo fixo, os de calendario tambem ficam desconhecidos
    fora = saida["geo_semaforos"].isna().to_numpy()
    saida.loc[fora, ["geo_faixa_ativa", "geo_feira_dia", "geo_feira_agora"]] = np.nan
    return saida.astype("float64")
