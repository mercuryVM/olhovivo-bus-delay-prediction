"""
Analise das regioes mais impactadas por atraso.

Quatro algoritmos, cada um respondendo uma pergunta diferente:

* **K-means** — particiona a cidade em K zonas de atraso. Usa o atraso como
  peso da amostra (`sample_weight`), entao os centroides sao puxados para onde
  o problema e maior, nao para onde ha mais onibus. Serve para dividir a
  operacao em regioes de gestao; exige escolher K e assume grupos convexos.

* **DBSCAN** — encontra aglomerados de densidade arbitraria e marca ruido. Bom
  para achar "manchas" de atraso sem supor formato. Sensivel ao par (eps,
  minPts), e um so eps nao serve para o centro e a periferia ao mesmo tempo.

* **HDBSCAN** — a versao hierarquica do DBSCAN: dispensa o eps e aceita
  densidades diferentes na mesma cidade, que e exatamente o caso de Sao Paulo
  (centro denso, extremos esparsos). E o mais indicado para o mapa final.

* **ST-DBSCAN** (Birant & Kut, 2007) — o unico que enxerga TEMPO. Dois pontos
  so sao vizinhos se estiverem a menos de `eps1` metros E a menos de `eps2`
  segundos um do outro. E o que separa "o Ibirapuera engarrafa toda terca as
  18h" de "o Ibirapuera engarrafou uma vez". Sem ele, um cluster espacial
  mistura o pico da manha com a madrugada.

Todos rodam sobre coordenadas **projetadas em metros**, nunca sobre graus:
1 grau de longitude em Sao Paulo vale ~102 km, 1 grau de latitude ~111 km, e
usar graus direto distorce o eps.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import geo
from .storage import Armazenamento

log = logging.getLogger("olhovivo.regioes")


# --------------------------------------------------------------- preparacao
def preparar_amostras(
    df: pd.DataFrame,
    proj,
    modo: str = "eventos",
    limiar_atraso_s: int = 300,
    min_amostras_parada: int = 20,
) -> pd.DataFrame:
    """
    `modo="eventos"`  -> uma amostra por chegada atrasada (mantem o tempo, e o
                         que o ST-DBSCAN precisa).
    `modo="agregado"` -> uma amostra por parada, com a estatistica do periodo
                         (mais estavel para K-means/HDBSCAN espacial).
    """
    base = df.dropna(subset=["lat", "lon", "erro_s"]).copy()
    if base.empty:
        return base

    if modo == "eventos":
        amostras = base[base["erro_s"] >= limiar_atraso_s].copy()
        amostras["peso"] = amostras["erro_s"].clip(lower=1)
        amostras["t"] = pd.to_datetime(amostras["t_chegada"], utc=True)
    else:
        agrupado = (
            base.groupby(["cp", "parada_nome"], dropna=False)
            .agg(
                n=("erro_s", "size"),
                lat=("lat", "mean"),
                lon=("lon", "mean"),
                erro_medio_s=("erro_s", "mean"),
                erro_mediano_s=("erro_s", "median"),
                p90_erro_s=("erro_s", lambda x: float(np.percentile(x, 90))),
                prob_atraso=("erro_s", lambda x: float((x >= limiar_atraso_s).mean())),
                headway_mediano_s=("headway_obs_s", "median"),
            )
            .reset_index()
        )
        amostras = agrupado[agrupado["n"] >= min_amostras_parada].copy()
        amostras["peso"] = amostras["prob_atraso"] * amostras["n"]
        amostras["t"] = pd.NaT

    if amostras.empty:
        return amostras

    x, y = proj.para_xy(amostras["lat"].to_numpy(), amostras["lon"].to_numpy())
    amostras["x_m"] = np.asarray(x)
    amostras["y_m"] = np.asarray(y)
    return amostras.reset_index(drop=True)


# ------------------------------------------------------------------ K-means
def rodar_kmeans(amostras: pd.DataFrame, k: int = 12, semente: int = 42) -> np.ndarray:
    from sklearn.cluster import KMeans

    X = amostras[["x_m", "y_m"]].to_numpy()
    peso = amostras["peso"].to_numpy(dtype="float64")
    modelo = KMeans(n_clusters=min(k, len(X)), n_init=10, random_state=semente)
    return modelo.fit_predict(X, sample_weight=peso)


# ------------------------------------------------------------------- DBSCAN
def rodar_dbscan(amostras: pd.DataFrame, eps_m: float = 300.0, min_amostras: int = 20) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    X = amostras[["x_m", "y_m"]].to_numpy()
    return DBSCAN(eps=eps_m, min_samples=min_amostras, n_jobs=-1).fit_predict(X)


# ------------------------------------------------------------------ HDBSCAN
def rodar_hdbscan(amostras: pd.DataFrame, min_cluster: int = 25) -> np.ndarray:
    X = amostras[["x_m", "y_m"]].to_numpy()
    try:
        from sklearn.cluster import HDBSCAN  # scikit-learn >= 1.3

        return HDBSCAN(min_cluster_size=min_cluster, n_jobs=-1).fit_predict(X)
    except ImportError:
        pass
    try:
        import hdbscan as _hdbscan  # pacote separado

        return _hdbscan.HDBSCAN(min_cluster_size=min_cluster).fit_predict(X)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HDBSCAN indisponivel: atualize o scikit-learn (>=1.3) ou "
            "instale o pacote `hdbscan`"
        ) from exc


# ---------------------------------------------------------------- ST-DBSCAN
RUIDO = -1
NAO_VISITADO = -2


def rodar_st_dbscan(
    amostras: pd.DataFrame,
    eps_espacial_m: float = 400.0,
    eps_temporal_s: float = 1800.0,
    min_pontos: int = 15,
    delta_erro_s: float | None = None,
) -> np.ndarray:
    """
    ST-DBSCAN (Birant & Kut, 2007), implementado aqui porque nao existe versao
    mantida e confiavel no PyPI.

    Vizinhanca = cilindro: raio `eps_espacial_m` no espaco E `eps_temporal_s`
    no tempo. `delta_erro_s`, se informado, aplica o criterio Delta-E do artigo
    original: um ponto so entra no cluster se o atraso dele nao destoar mais
    que esse limite da media do cluster — evita fundir uma mancha de 30 s com
    outra de 20 min so porque sao vizinhas.

    Complexidade: KD-tree espacial + filtro temporal. Para muitas amostras use
    `modo="agregado"` ou recorte a janela de analise.
    """
    n = len(amostras)
    if n == 0:
        return np.array([], dtype="int64")

    x = amostras["x_m"].to_numpy(dtype="float64")
    y = amostras["y_m"].to_numpy(dtype="float64")
    tempos = pd.to_datetime(amostras["t"], utc=True)
    if tempos.isna().all():
        raise ValueError(
            "ST-DBSCAN precisa da dimensao temporal; use modo='eventos'"
        )
    # NAO usar astype("int64")/1e9: o pandas 2.x guarda datetime em ns, us ou
    # ms conforme a origem do dado, e a conta silenciosamente erraria por 1000x.
    epoca = pd.Timestamp("1970-01-01", tz="UTC")
    t = (tempos - epoca).dt.total_seconds().to_numpy(dtype="float64")
    valores = amostras["erro_s"].to_numpy(dtype="float64") if "erro_s" in amostras else None

    vizinhos = _vizinhanca_espaco_tempo(x, y, t, eps_espacial_m, eps_temporal_s)

    rotulos = np.full(n, NAO_VISITADO, dtype="int64")
    cluster = 0

    for i in range(n):
        if rotulos[i] != NAO_VISITADO:
            continue
        if len(vizinhos[i]) < min_pontos:
            rotulos[i] = RUIDO
            continue

        rotulos[i] = cluster
        fila = deque(vizinhos[i])
        soma = float(valores[i]) if valores is not None else 0.0
        contagem = 1

        while fila:
            j = fila.popleft()
            if rotulos[j] == RUIDO:
                rotulos[j] = cluster
                continue
            if rotulos[j] != NAO_VISITADO:
                continue

            if delta_erro_s is not None and valores is not None and contagem:
                media = soma / contagem
                if abs(float(valores[j]) - media) > delta_erro_s:
                    continue

            rotulos[j] = cluster
            if valores is not None:
                soma += float(valores[j])
                contagem += 1

            if len(vizinhos[j]) >= min_pontos:
                fila.extend(k for k in vizinhos[j] if rotulos[k] == NAO_VISITADO)

        cluster += 1

    rotulos[rotulos == NAO_VISITADO] = RUIDO
    return rotulos


def _vizinhanca_espaco_tempo(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, eps_s: float, eps_t: float
) -> list[np.ndarray]:
    """Lista de vizinhos dentro do cilindro (raio espacial x janela temporal)."""
    n = len(x)
    try:
        from scipy.spatial import cKDTree

        arvore = cKDTree(np.column_stack([x, y]))
        candidatos = arvore.query_ball_point(np.column_stack([x, y]), r=eps_s)
        saida = []
        for i, cand in enumerate(candidatos):
            arr = np.fromiter(cand, dtype="int64", count=len(cand))
            saida.append(arr[np.abs(t[arr] - t[i]) <= eps_t])
        return saida
    except ImportError:
        log.warning("scipy ausente: usando busca em blocos (mais lenta)")

    saida = []
    bloco = 2000
    for i0 in range(0, n, bloco):
        i1 = min(i0 + bloco, n)
        dx = x[i0:i1, None] - x[None, :]
        dy = y[i0:i1, None] - y[None, :]
        dt = np.abs(t[i0:i1, None] - t[None, :])
        perto = (dx * dx + dy * dy <= eps_s * eps_s) & (dt <= eps_t)
        for linha in perto:
            saida.append(np.flatnonzero(linha))
    return saida


# ---------------------------------------------------------------- resultados
def resumir_clusters(
    amostras: pd.DataFrame, rotulos: np.ndarray, algoritmo: str, proj
) -> list[dict]:
    """Estatisticas + geometria de cada cluster, prontas para o mapa e o Mongo."""
    df = amostras.copy()
    df["cluster"] = rotulos
    saida: list[dict] = []

    for rotulo, grupo in df.groupby("cluster", sort=True):
        if rotulo == RUIDO:
            continue
        lat_c = float(grupo["lat"].mean())
        lon_c = float(grupo["lon"].mean())
        registro: dict[str, Any] = {
            "algoritmo": algoritmo,
            "rotulo": int(rotulo),
            "n_amostras": int(len(grupo)),
            "lat": lat_c,
            "lon": lon_c,
            "geometry": geo.ponto_geojson(lat_c, lon_c),
            "envoltoria": _envoltoria(grupo, proj),
            "raio_m": float(
                np.percentile(
                    geo.haversine_m(grupo["lat"], grupo["lon"], lat_c, lon_c), 90
                )
            ),
            "paradas": sorted({int(c) for c in grupo["cp"]})[:200]
            if "cp" in grupo
            else [],
            "celulas": sorted({str(c) for c in grupo["celula"]})[:200]
            if "celula" in grupo
            else [],
        }

        if "erro_s" in grupo:
            registro |= {
                "erro_medio_s": float(grupo["erro_s"].mean()),
                "erro_mediano_s": float(grupo["erro_s"].median()),
                "erro_p90_s": float(np.percentile(grupo["erro_s"], 90)),
            }
        if "erro_medio_s" in grupo and "erro_medio_s" not in registro:
            registro["erro_medio_s"] = float(grupo["erro_medio_s"].mean())
        if grupo["t"].notna().any():
            local = pd.to_datetime(grupo["t"], utc=True)
            registro |= {
                "inicio": local.min().to_pydatetime(),
                "fim": local.max().to_pydatetime(),
                "horas_predominantes": (
                    local.dt.tz_convert("America/Sao_Paulo")
                    .dt.hour.value_counts()
                    .head(3)
                    .index.tolist()
                ),
            }
        if "letreiro" in grupo:
            registro["linhas"] = (
                grupo["letreiro"].value_counts().head(15).index.astype(str).tolist()
            )
        saida.append(registro)

    saida.sort(key=lambda c: c.get("erro_medio_s", 0) * c["n_amostras"], reverse=True)
    return saida


def _envoltoria(grupo: pd.DataFrame, proj) -> dict | None:
    """Poligono GeoJSON do cluster (casco convexo; retangulo se faltar scipy)."""
    pontos = grupo[["lon", "lat"]].to_numpy()
    if len(pontos) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull

        casco = ConvexHull(pontos)
        anel = pontos[casco.vertices].tolist()
    except Exception:
        lon0, lat0 = pontos.min(axis=0)
        lon1, lat1 = pontos.max(axis=0)
        anel = [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]]
    anel.append(anel[0])
    return {"type": "Polygon", "coordinates": [[[float(a), float(b)] for a, b in anel]]}


def exportar_geojson(clusters: Sequence[dict], caminho) -> None:
    import json

    feicoes = []
    for c in clusters:
        propriedades = {
            k: v
            for k, v in c.items()
            if k not in ("geometry", "envoltoria") and not isinstance(v, (dict,))
        }
        feicoes.append(
            {
                "type": "Feature",
                "geometry": c.get("envoltoria") or c["geometry"],
                "properties": {
                    k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in propriedades.items()
                },
            }
        )
    with open(caminho, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feicoes}, fh, ensure_ascii=False)


# ---------------------------------------------------------------- orquestra
def executar(
    cfg,
    arm: Armazenamento,
    algoritmos: Sequence[str] = ("kmeans", "dbscan", "hdbscan", "st-dbscan"),
    modo: str = "eventos",
    limiar_atraso_s: int = 300,
) -> dict:
    proj = geo.obter_projecao(
        int(cfg.get("geo.epsg_metrico", 31983)),
        float(cfg.get("geo.ancora_lat", -23.5505)),
        float(cfg.get("geo.ancora_lon", -46.6333)),
    )
    df = arm.ler_derivado("previsao_realizado")
    if df.empty:
        raise RuntimeError("rode `python -m olhovivo casar` antes de agrupar regioes")

    amostras = preparar_amostras(df, proj, modo=modo, limiar_atraso_s=limiar_atraso_s)
    if amostras.empty:
        raise RuntimeError("nenhuma amostra apos o filtro de atraso")

    log.info("agrupando %d amostras (modo=%s)", len(amostras), modo)
    resultados: dict[str, Any] = {}
    todos: list[dict] = []

    for algoritmo in algoritmos:
        try:
            if algoritmo == "kmeans":
                rotulos = rodar_kmeans(amostras, int(cfg.get("regioes.kmeans_k", 12)))
            elif algoritmo == "dbscan":
                rotulos = rodar_dbscan(
                    amostras,
                    float(cfg.get("regioes.dbscan_eps_m", 300)),
                    int(cfg.get("regioes.dbscan_min_amostras", 20)),
                )
            elif algoritmo == "hdbscan":
                rotulos = rodar_hdbscan(
                    amostras, int(cfg.get("regioes.hdbscan_min_cluster", 25))
                )
            elif algoritmo in ("st-dbscan", "stdbscan"):
                if modo != "eventos":
                    log.warning("ST-DBSCAN exige modo='eventos'; pulando")
                    continue
                rotulos = rodar_st_dbscan(
                    amostras,
                    float(cfg.get("regioes.eps_espacial_m", 400)),
                    float(cfg.get("regioes.eps_temporal_s", 1800)),
                    int(cfg.get("regioes.min_pontos", 15)),
                )
            else:
                log.warning("algoritmo desconhecido: %s", algoritmo)
                continue
        except Exception as exc:
            log.error("%s falhou: %s", algoritmo, exc)
            resultados[algoritmo] = {"erro": str(exc)}
            continue

        clusters = resumir_clusters(amostras, rotulos, algoritmo, proj)
        ruido = int((rotulos == RUIDO).sum())
        resultados[algoritmo] = {
            "clusters": len(clusters),
            "ruido": ruido,
            "fracao_ruido": round(ruido / len(rotulos), 3),
            "top": [
                {
                    "rotulo": c["rotulo"],
                    "n": c["n_amostras"],
                    "erro_medio_s": round(c.get("erro_medio_s", 0), 1),
                    "lat": round(c["lat"], 5),
                    "lon": round(c["lon"], 5),
                }
                for c in clusters[:5]
            ],
        }
        todos.extend(clusters)
        amostras[f"cluster_{algoritmo}"] = rotulos
        log.info("%s: %d clusters, %d ruido", algoritmo, len(clusters), ruido)

    destino = arm.raiz / "derivado"
    destino.mkdir(parents=True, exist_ok=True)
    amostras.to_parquet(destino / "amostras_regioes.parquet", index=False)
    exportar_geojson(todos, destino / "regioes.geojson")

    if arm.mongo and todos:
        docs = [{**c} for c in todos]
        arm.mongo.substituir_colecao("regioes", docs, geo_campo="geometry")

    arm.eventos.registrar("regioes_agrupadas", **{k: v for k, v in resultados.items()})
    return resultados
