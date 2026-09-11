"""
Analise das regioes mais impactadas por atraso.

Fonte dos dados
---------------
Por padrao (`fonte="atraso"`) agrupa as ocorrencias da VARIAVEL DO ESTUDO: os
trechos de `percursos` marcados como `atrasado` — tempo de percurso acima do
limiar percentual sobre a referencia do trecho. `fonte="previsao"` agrupa, em
vez disso, o erro da previsao publicada pela API (`previsao_realizado`), que e
medida complementar.

Quatro algoritmos, cada um respondendo uma pergunta diferente:

* **K-means** — particiona a cidade em K zonas. Pondera cada parada pela
  quantidade de atraso (`sample_weight`), entao os centroides sao puxados para
  onde o problema e maior, nao para onde ha mais onibus.

* **DBSCAN** — aglomerados de densidade arbitraria, com ruido. Um so eps nao
  serve igualmente ao centro e a periferia.

* **HDBSCAN** — a versao hierarquica: dispensa o eps e aceita densidades
  diferentes na mesma cidade.

* **DBSCAN com restricao temporal** (variante de ST-DBSCAN) — o unico que
  enxerga TEMPO: dois eventos so sao vizinhos se estiverem a menos de `eps1`
  metros E a menos de `eps2` segundos um do outro.

Unidade de agrupamento (`modo`)
-------------------------------
* `agregado` — uma amostra por parada, com a estatistica da semana. Estavel e
  usa todos os dados. K-means roda sobre todas as paradas, ponderadas;
  DBSCAN/HDBSCAN rodam so sobre as paradas IMPACTADAS (taxa de atraso acima da
  taxa geral) — senao o que se agrupa e a densidade de paradas, nao de atraso.
* `eventos` — uma amostra por ocorrencia de atraso, com o horario. E o que o
  agrupamento temporal precisa. Acima de `regioes.max_amostras`, sorteia — e
  escala os limiares de densidade pela fracao sorteada, para o criterio
  "quantos vizinhos bastam" continuar significando a mesma densidade.
* `auto` (padrao) — cada algoritmo no regime em que faz sentido: os tres
  espaciais no agregado, o temporal nos eventos.

Tudo roda sobre coordenadas projetadas em metros, nunca sobre graus.
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

RUIDO = -1
NAO_VISITADO = -2
FORA_DO_FILTRO = -3

TEMPORAIS = {"st-dbscan", "stdbscan"}


# ------------------------------------------------------------------- fontes
def carregar_base(arm: Armazenamento, fonte: str, limiar_atraso_s: int) -> pd.DataFrame:
    """
    Le a tabela de entrada e padroniza as colunas.

    Saida: `valor_s` (a medida de atraso), `evento` (se e ocorrencia de atraso),
    `t`, `cp`, `lat`, `lon` e, quando existirem, `headway_s`, `letreiro`,
    `celula`, `regiao_origem`, `regiao_destino`.
    """
    if fonte == "atraso":
        df = arm.ler_derivado("percursos")
        if df.empty:
            raise RuntimeError("rode `python -m olhovivo atraso` antes de agrupar regioes")
        colunas = {
            "valor_s": df["atraso_s"].astype("float64"),
            "evento": df["atrasado"].astype(bool),
            "t": pd.to_datetime(df["t_chegada"], utc=True),
            "cp": df["cp_destino"].astype("int64"),
            "lat": df["lat"],
            "lon": df["lon"],
        }
        opcionais = {
            "headway_s": "headway_s",
            "letreiro": "letreiro",
            "celula": "celula",
            "regiao_origem": "regiao_origem",
            "regiao_destino": "regiao_destino",
        }
    elif fonte == "previsao":
        df = arm.ler_derivado("previsao_realizado")
        if df.empty:
            raise RuntimeError("rode `python -m olhovivo casar` antes de agrupar regioes")
        erro = df["erro_s"].astype("float64")
        colunas = {
            "valor_s": erro,
            "evento": erro >= limiar_atraso_s,
            "t": pd.to_datetime(df["t_chegada"], utc=True),
            "cp": df["cp"].astype("int64"),
            "lat": df["lat"],
            "lon": df["lon"],
        }
        opcionais = {
            "headway_s": "headway_obs_s",
            "letreiro": "letreiro",
            "celula": "celula",
            "parada_nome": "parada_nome",
        }
    else:
        raise ValueError(f"fonte desconhecida: {fonte!r} (use 'atraso' ou 'previsao')")

    for destino, origem in opcionais.items():
        if origem in df.columns:
            colunas[destino] = df[origem]
    base = pd.DataFrame(colunas)
    return base.dropna(subset=["lat", "lon", "valor_s"]).reset_index(drop=True)


# --------------------------------------------------------------- preparacao
def _projetar(amostras: pd.DataFrame, proj) -> pd.DataFrame:
    x, y = proj.para_xy(amostras["lat"].to_numpy(), amostras["lon"].to_numpy())
    amostras["x_m"] = np.asarray(x)
    amostras["y_m"] = np.asarray(y)
    return amostras


def preparar_amostras(
    base: pd.DataFrame, proj, modo: str = "eventos", min_amostras_parada: int = 20
) -> pd.DataFrame:
    """Uma amostra por ocorrencia (`eventos`) ou por parada (`agregado`)."""
    if base.empty:
        return base

    if modo == "eventos":
        amostras = base[base["evento"]].copy()
        amostras["peso"] = amostras["valor_s"].clip(lower=1)
    else:
        g = base.groupby("cp", sort=False)
        agrupado = g.agg(
            n=("valor_s", "size"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            valor_medio_s=("valor_s", "mean"),
            valor_mediano_s=("valor_s", "median"),
            prob_atraso=("evento", "mean"),
        )
        agrupado["valor_p90_s"] = g["valor_s"].quantile(0.9)
        if "headway_s" in base:
            agrupado["headway_mediano_s"] = g["headway_s"].median()
        amostras = agrupado.reset_index()
        amostras = amostras[amostras["n"] >= min_amostras_parada].copy()
        amostras["peso"] = amostras["prob_atraso"] * amostras["n"]
        amostras["t"] = pd.NaT

    if amostras.empty:
        return amostras
    return _projetar(amostras, proj)


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
    return DBSCAN(eps=eps_m, min_samples=min_amostras, n_jobs=4).fit_predict(X)


# ------------------------------------------------------------------ HDBSCAN
def rodar_hdbscan(amostras: pd.DataFrame, min_cluster: int = 25) -> np.ndarray:
    X = amostras[["x_m", "y_m"]].to_numpy()
    try:
        from sklearn.cluster import HDBSCAN  # scikit-learn >= 1.3

        return HDBSCAN(min_cluster_size=max(2, min_cluster), n_jobs=4).fit_predict(X)
    except ImportError:
        pass
    try:
        import hdbscan as _hdbscan  # pacote separado

        return _hdbscan.HDBSCAN(min_cluster_size=max(2, min_cluster)).fit_predict(X)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HDBSCAN indisponivel: atualize o scikit-learn (>=1.3) ou "
            "instale o pacote `hdbscan`"
        ) from exc


# ------------------------------------------------ DBSCAN com restricao temporal
def rodar_st_dbscan(
    amostras: pd.DataFrame,
    eps_espacial_m: float = 400.0,
    eps_temporal_s: float = 1800.0,
    min_pontos: int = 15,
    delta_erro_s: float | None = None,
) -> np.ndarray:
    """
    DBSCAN com vizinhanca em cilindro: raio `eps_espacial_m` no espaco E janela
    `eps_temporal_s` no tempo. Estrutura de dupla restricao de Birant e Kut
    (2007), com o segundo limiar reinterpretado como janela temporal; o density
    factor e o Delta-eps do artigo original nao sao implementados (ver o
    apendice metodologico). `delta_erro_s`, se informado, impede anexar ao
    cluster um ponto cujo valor destoe da media do cluster.
    """
    n = len(amostras)
    if n == 0:
        return np.array([], dtype="int64")

    x = amostras["x_m"].to_numpy(dtype="float64")
    y = amostras["y_m"].to_numpy(dtype="float64")
    tempos = pd.to_datetime(amostras["t"], utc=True)
    if tempos.isna().all():
        raise ValueError("o agrupamento temporal precisa da coluna `t`; use modo='eventos'")
    # NAO usar astype("int64")/1e9: o pandas 2.x guarda datetime em ns, us ou
    # ms conforme a origem do dado, e a conta silenciosamente erraria por 1000x.
    epoca = pd.Timestamp("1970-01-01", tz="UTC")
    t = (tempos - epoca).dt.total_seconds().to_numpy(dtype="float64")
    coluna_valor = "valor_s" if "valor_s" in amostras else ("erro_s" if "erro_s" in amostras else None)
    valores = amostras[coluna_valor].to_numpy(dtype="float64") if coluna_valor else None

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
                if abs(float(valores[j]) - soma / contagem) > delta_erro_s:
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
    """
    Vizinhos dentro do cilindro (raio espacial x janela temporal).

    A consulta espacial devolve TODOS os vizinhos no raio, antes do filtro de
    tempo — e como lista de listas de inteiros Python, que custa dezenas de
    bytes por entrada. Por isso a consulta e feita em blocos: so um bloco de
    candidatos existe por vez, e o que fica guardado ja e o resultado filtrado
    pelo tempo, em int32. Uma versao sem essa guarda, com 867 mil eventos,
    passou de 24 GiB.
    """
    n = len(x)
    if n > 120_000:
        raise MemoryError(
            f"{n:,} pontos e demais para o agrupamento temporal. "
            "Reduza regioes.max_amostras."
        )

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None

    saida: list[np.ndarray] = []
    pontos = np.column_stack([x, y])
    bloco = 4000

    if cKDTree is not None:
        arvore = cKDTree(pontos)
        for i0 in range(0, n, bloco):
            i1 = min(i0 + bloco, n)
            candidatos = arvore.query_ball_point(pontos[i0:i1], r=eps_s)
            for k, cand in enumerate(candidatos):
                arr = np.fromiter(cand, dtype="int32", count=len(cand))
                saida.append(arr[np.abs(t[arr] - t[i0 + k]) <= eps_t])
            del candidatos
        return saida

    log.warning("scipy ausente: usando busca em blocos (mais lenta)")
    for i0 in range(0, n, 2000):
        i1 = min(i0 + 2000, n)
        dx = x[i0:i1, None] - x[None, :]
        dy = y[i0:i1, None] - y[None, :]
        dt = np.abs(t[i0:i1, None] - t[None, :])
        perto = (dx * dx + dy * dy <= eps_s * eps_s) & (dt <= eps_t)
        for linha in perto:
            saida.append(np.flatnonzero(linha).astype("int32"))
    return saida


# ---------------------------------------------------------------- resultados
def _fluxos(tabela: pd.DataFrame, limite: int = 5) -> list[dict]:
    """Pares (regiao de origem, regiao de destino) mais frequentes."""
    if not {"regiao_origem", "regiao_destino"} <= set(tabela.columns):
        return []
    contagem = (
        tabela.dropna(subset=["regiao_origem", "regiao_destino"])
        .groupby(["regiao_origem", "regiao_destino"])
        .size()
        .sort_values(ascending=False)
        .head(limite)
    )
    return [
        {"origem": str(o), "destino": str(d), "n": int(n)}
        for (o, d), n in contagem.items()
    ]


def resumir_clusters(
    amostras: pd.DataFrame,
    rotulos: np.ndarray,
    algoritmo: str,
    proj,
    base: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Estatisticas + geometria de cada cluster, prontas para o mapa e o Mongo.

    Quando as amostras sao paradas (agregado), a composicao em linhas e em
    fluxos origem-destino vem da `base` de eventos das paradas do cluster —
    e assim que o agrupamento e cruzado com as regioes de origem e destino.
    """
    df = amostras.copy()
    df["cluster"] = rotulos
    saida: list[dict] = []
    coluna_valor = "valor_s" if "valor_s" in df else ("erro_s" if "erro_s" in df else None)
    eventos_base = base[base["evento"]] if base is not None and "evento" in base else None

    for rotulo, grupo in df.groupby("cluster", sort=True):
        if rotulo < 0:
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
            "envoltoria": _envoltoria(grupo),
            "raio_m": float(
                np.percentile(geo.haversine_m(grupo["lat"], grupo["lon"], lat_c, lon_c), 90)
            ),
            "paradas": sorted({int(c) for c in grupo["cp"]})[:200] if "cp" in grupo else [],
        }

        if coluna_valor:
            registro |= {
                "atraso_medio_s": float(grupo[coluna_valor].mean()),
                "atraso_mediano_s": float(grupo[coluna_valor].median()),
                "atraso_p90_s": float(np.percentile(grupo[coluna_valor], 90)),
            }
        elif "valor_medio_s" in grupo:
            peso = grupo["n"].to_numpy(dtype="float64")
            registro |= {
                "atraso_medio_s": float(np.average(grupo["valor_medio_s"], weights=peso)),
                "prob_atraso": float(np.average(grupo["prob_atraso"], weights=peso)),
                "n_observacoes": int(peso.sum()),
            }

        composicao = grupo
        if eventos_base is not None and "cp" in grupo and "regiao_origem" not in grupo:
            composicao = eventos_base[eventos_base["cp"].isin(set(grupo["cp"]))]
        registro["fluxos"] = _fluxos(composicao)
        if "letreiro" in composicao:
            registro["linhas"] = (
                composicao["letreiro"].value_counts().head(15).index.astype(str).tolist()
            )

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
        saida.append(registro)

    saida.sort(
        key=lambda c: c.get("atraso_medio_s", 0) * c.get("n_observacoes", c["n_amostras"]),
        reverse=True,
    )
    return saida


def _envoltoria(grupo: pd.DataFrame) -> dict | None:
    """Poligono GeoJSON do cluster (casco convexo; retangulo se faltar scipy)."""
    pontos = grupo[["lon", "lat"]].to_numpy()
    if len(pontos) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull

        anel = pontos[ConvexHull(pontos).vertices].tolist()
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
            k: v for k, v in c.items() if k not in ("geometry", "envoltoria")
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
        json.dump({"type": "FeatureCollection", "features": feicoes}, fh, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- orquestra
def executar(
    cfg,
    arm: Armazenamento,
    algoritmos: Sequence[str] = ("kmeans", "dbscan", "hdbscan", "st-dbscan"),
    modo: str = "auto",
    limiar_atraso_s: int = 300,
    fonte: str = "atraso",
) -> dict:
    proj = geo.obter_projecao(
        int(cfg.get("geo.epsg_metrico", 31983)),
        float(cfg.get("geo.ancora_lat", -23.5505)),
        float(cfg.get("geo.ancora_lon", -46.6333)),
    )
    base = carregar_base(arm, fonte, limiar_atraso_s)
    if base.empty:
        raise RuntimeError(f"tabela de {fonte} vazia")
    taxa_geral = float(base["evento"].mean())
    log.info(
        "fonte=%s: %d registros, taxa de atraso %.1f%%", fonte, len(base), taxa_geral * 100
    )

    max_amostras = int(cfg.get("regioes.max_amostras", 50_000))
    min_parada = int(cfg.get("regioes.min_amostras_parada", 20))
    cache: dict[str, tuple[pd.DataFrame, float]] = {}

    def amostras_de(regime: str) -> tuple[pd.DataFrame, float]:
        """(amostras, fracao sorteada) do regime, calculadas uma vez so."""
        if regime not in cache:
            a = preparar_amostras(base, proj, modo=regime, min_amostras_parada=min_parada)
            fracao = 1.0
            if regime == "eventos" and len(a) > max_amostras:
                fracao = max_amostras / len(a)
                log.warning(
                    "%d eventos excedem o teto de %d; sorteando %.1f%% e escalando "
                    "os limiares de densidade na mesma proporcao",
                    len(a), max_amostras, fracao * 100,
                )
                a = a.sample(n=max_amostras, random_state=42)
            elif regime == "agregado" and len(a) > max_amostras:
                raise RuntimeError(
                    f"{len(a)} paradas excedem o teto de {max_amostras}; "
                    "aumente regioes.max_amostras"
                )
            cache[regime] = (a, fracao)
            log.info("regime %s: %d amostras", regime, len(a))
        return cache[regime]

    def escala(valor: int, fracao: float) -> int:
        return max(3, int(round(valor * fracao)))

    resultados: dict[str, Any] = {}
    todos: list[dict] = []

    for algoritmo in algoritmos:
        temporal = algoritmo in TEMPORAIS
        regime = ("eventos" if temporal else "agregado") if modo == "auto" else modo
        if temporal and regime != "eventos":
            log.warning("%s precisa de modo 'eventos'; pulando", algoritmo)
            resultados[algoritmo] = {"pulado": "precisa de modo eventos"}
            continue

        amostras, fracao = amostras_de(regime)
        if amostras.empty:
            resultados[algoritmo] = {"erro": f"sem amostras no regime {regime}"}
            continue

        # no agregado, os metodos de densidade olham so as paradas impactadas
        alvo = amostras
        if regime == "agregado" and algoritmo in ("dbscan", "hdbscan"):
            alvo = amostras[amostras["prob_atraso"] >= taxa_geral]

        try:
            if algoritmo == "kmeans":
                rotulos = rodar_kmeans(alvo, int(cfg.get("regioes.kmeans_k", 12)))
            elif algoritmo == "dbscan":
                if regime == "agregado":
                    rotulos = rodar_dbscan(
                        alvo,
                        float(cfg.get("regioes.agregado_dbscan_eps_m", 800)),
                        int(cfg.get("regioes.agregado_dbscan_min", 3)),
                    )
                else:
                    rotulos = rodar_dbscan(
                        alvo,
                        float(cfg.get("regioes.dbscan_eps_m", 300)),
                        escala(int(cfg.get("regioes.dbscan_min_amostras", 20)), fracao),
                    )
            elif algoritmo == "hdbscan":
                minimo = (
                    int(cfg.get("regioes.agregado_hdbscan_min_cluster", 5))
                    if regime == "agregado"
                    else escala(int(cfg.get("regioes.hdbscan_min_cluster", 25)), fracao)
                )
                rotulos = rodar_hdbscan(alvo, minimo)
            elif temporal:
                rotulos = rodar_st_dbscan(
                    alvo,
                    float(cfg.get("regioes.eps_espacial_m", 400)),
                    float(cfg.get("regioes.eps_temporal_s", 1800)),
                    escala(int(cfg.get("regioes.min_pontos", 15)), fracao),
                )
            else:
                log.warning("algoritmo desconhecido: %s", algoritmo)
                continue
        except Exception as exc:
            log.error("%s falhou: %s", algoritmo, exc)
            resultados[algoritmo] = {"erro": str(exc)}
            continue

        clusters = resumir_clusters(
            alvo, rotulos, algoritmo, proj, base=base if regime == "agregado" else None
        )
        for c in clusters:
            c["modo"] = regime
            c["fonte"] = fonte
        ruido = int((rotulos == RUIDO).sum())
        resultados[algoritmo] = {
            "modo": regime,
            "amostras": int(len(alvo)),
            "clusters": len(clusters),
            "ruido": ruido,
            "fracao_ruido": round(ruido / max(len(rotulos), 1), 3),
            "top": [
                {
                    "rotulo": c["rotulo"],
                    "n": c.get("n_observacoes", c["n_amostras"]),
                    "atraso_medio_s": round(c.get("atraso_medio_s", 0), 1),
                    "fluxo": (c["fluxos"][0]["origem"] + " -> " + c["fluxos"][0]["destino"])
                    if c.get("fluxos")
                    else None,
                    "lat": round(c["lat"], 5),
                    "lon": round(c["lon"], 5),
                }
                for c in clusters[:5]
            ],
        }
        todos.extend(clusters)
        rot = pd.Series(rotulos, index=alvo.index)
        amostras[f"cluster_{algoritmo}"] = rot.reindex(amostras.index).fillna(FORA_DO_FILTRO).astype("int64")
        log.info("%s (%s): %d clusters, %d ruido", algoritmo, regime, len(clusters), ruido)

    destino = arm.raiz / "derivado"
    destino.mkdir(parents=True, exist_ok=True)
    for regime, (amostras, _) in cache.items():
        amostras.to_parquet(destino / f"amostras_regioes_{regime}.parquet", index=False)
    exportar_geojson(todos, destino / "regioes.geojson")

    if arm.mongo and todos:
        # substitui so os algoritmos que rodaram agora, sem apagar os outros
        col = arm.mongo.db["regioes"]
        col.delete_many({"algoritmo": {"$in": list(algoritmos)}})
        for i in range(0, len(todos), 1000):
            col.insert_many([dict(c) for c in todos[i : i + 1000]], ordered=False)

    arm.eventos.registrar("regioes_agrupadas", fonte=fonte, **resultados)
    return resultados
