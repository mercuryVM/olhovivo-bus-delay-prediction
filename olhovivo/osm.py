"""
Atributos fixos dos trechos a partir do OpenStreetMap.

A consulta (Overpass) fica em <raiz>/externo/osm/osm_area.json: semaforos
(highway ou crossing = traffic_signals), vias e faixas de onibus (highway=busway,
busway*, lanes:bus, bus:lanes) e as vias principais com maxspeed, na area das
linhas estudadas. Mais atual que a camada de semaforos do GeoSampa (2018), mas
com cobertura que varia de bairro para bairro.

Por par de paradas (a geometria de frota.geometria):

* osm_semaforos e osm_semaforos_km: cruzamentos com semaforo a ate 20 m do
  tracado. Os nos a menos de 30 m uns dos outros contam como um cruzamento so,
  porque o OSM costuma marcar cada aproximacao.
* osm_faixa: fracao do tracado a ate 15 m de uma via ou faixa de onibus
* osm_vel_max_kmh: mediana do limite de velocidade das vias a ate 15 m do tracado
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import frota

log = logging.getLogger("olhovivo.osm")

COLUNAS_OSM = ["osm_semaforos", "osm_semaforos_km", "osm_faixa", "osm_vel_max_kmh"]
RAIO_SEMAFORO_M = 20.0
JUNTAR_SEMAFORO_M = 30.0
RAIO_VIA_M = 15.0
PASSO_M = 5.0
CHAVE = ["cp_origem", "cp_destino"]
TAGS_ONIBUS = ("busway", "busway:right", "busway:left", "busway:both", "lanes:bus", "bus:lanes")


def _xy(lat, lon, proj) -> np.ndarray:
    x, y = proj.para_xy(np.asarray(lat, dtype="float64"), np.asarray(lon, dtype="float64"))
    return np.c_[np.asarray(x), np.asarray(y)]


def _densificar(geometria: list[dict], proj) -> np.ndarray:
    p = _xy([g["lat"] for g in geometria], [g["lon"] for g in geometria], proj)
    if len(p) < 2:
        return p
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    sk = np.append(np.arange(0, s[-1], PASSO_M), s[-1])
    return np.c_[np.interp(sk, s, p[:, 0]), np.interp(sk, s, p[:, 1])]


def _velocidade(texto) -> float:
    m = re.match(r"\s*(\d+)", str(texto or ""))
    return float(m.group(1)) if m else np.nan


def trechos(arm, cfg) -> pd.DataFrame:
    """Atributos de `COLUNAS_OSM` por par de paradas."""
    elementos = json.loads((arm.raiz / "externo" / "osm" / "osm_area.json").read_text(encoding="utf-8"))["elements"]
    proj = frota._projecao(cfg)
    pares, pontos = frota.geometria(arm, cfg)
    xy = np.c_[pontos["x"], pontos["y"]]
    par = pontos["par"]
    saida = pares[["par", *CHAVE, "extensao_m"]].copy()

    nos = [e for e in elementos if e["type"] == "node"
           and "traffic_signals" in (e.get("tags", {}).get("highway"), e.get("tags", {}).get("crossing"))]
    sem = _xy([n["lat"] for n in nos], [n["lon"] for n in nos], proj) if nos else np.empty((0, 2))
    contagem = np.zeros(len(pares))
    if len(sem):
        # um cruzamento por grupo de nos proximos
        grafo = cKDTree(sem).sparse_distance_matrix(cKDTree(sem), JUNTAR_SEMAFORO_M)
        _, grupo = connected_components(grafo, directed=False)
        centros = pd.DataFrame(sem, columns=["x", "y"]).groupby(grupo).mean().to_numpy()
        for idx in cKDTree(xy).query_ball_point(centros, r=RAIO_SEMAFORO_M):
            for p in np.unique(par[idx]):
                contagem[p] += 1
    saida["osm_semaforos"] = contagem[saida["par"].to_numpy()]
    saida["osm_semaforos_km"] = saida["osm_semaforos"] / (saida["extensao_m"] / 1000).clip(lower=0.05)

    vias = [e for e in elementos if e["type"] == "way" and e.get("geometry")]
    onibus = [v for v in vias if v.get("tags", {}).get("highway") == "busway"
              or any(t in v.get("tags", {}) for t in TAGS_ONIBUS)]
    if onibus:
        lin = np.concatenate([_densificar(v["geometry"], proj) for v in onibus])
        dist, _ = cKDTree(lin).query(xy, k=1, distance_upper_bound=RAIO_VIA_M)
        fr = pd.Series(np.isfinite(dist).astype("float64")).groupby(par).mean()
        saida["osm_faixa"] = fr.reindex(saida["par"]).to_numpy()
    else:
        saida["osm_faixa"] = np.nan

    com_vel = [(v, _velocidade(v.get("tags", {}).get("maxspeed"))) for v in vias if "maxspeed" in v.get("tags", {})]
    com_vel = [(v, s) for v, s in com_vel if np.isfinite(s)]
    if com_vel:
        partes = [_densificar(v["geometry"], proj) for v, _ in com_vel]
        lin = np.concatenate(partes)
        vel = np.concatenate([np.full(len(p), s) for p, (_, s) in zip(partes, com_vel)])
        dist, j = cKDTree(lin).query(xy, k=1, distance_upper_bound=RAIO_VIA_M)
        ok = np.isfinite(dist)
        v_par = pd.Series(vel[j[ok]]).groupby(par[ok]).median()
        saida["osm_vel_max_kmh"] = v_par.reindex(saida["par"]).to_numpy()
    else:
        saida["osm_vel_max_kmh"] = np.nan
    log.info("osm: %d pares, %d nos de semaforo, %d vias de onibus, %d vias com limite", len(saida),
             len(nos), len(onibus), len(com_vel))
    return saida.drop(columns="par")


def gerar(arm, cfg) -> Path:
    """Grava derivado/osm_trechos.parquet."""
    destino = arm.raiz / "derivado" / "osm_trechos.parquet"
    trechos(arm, cfg).to_parquet(destino, index=False)
    return destino


def atributos(df: pd.DataFrame, raiz: Path) -> pd.DataFrame:
    """Colunas de `COLUNAS_OSM` no indice de `df` (percursos com cp_origem e cp_destino)."""
    saida = pd.DataFrame(np.nan, index=df.index, columns=COLUNAS_OSM)
    arq = Path(raiz) / "derivado" / "osm_trechos.parquet"
    if not arq.exists():
        return saida
    base = pd.DataFrame({"i": np.arange(len(df)),
                         "cp_origem": df["cp_origem"].astype("int64").to_numpy(),
                         "cp_destino": df["cp_destino"].astype("int64").to_numpy()})
    t = pd.read_parquet(arq).astype({"cp_origem": "int64", "cp_destino": "int64"})
    m = base.merge(t, on=CHAVE, how="left").sort_values("i")
    for c in COLUNAS_OSM:
        saida[c] = m[c].to_numpy()
    return saida.astype("float64")
