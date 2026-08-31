"""
GTFS estatico da SPTrans — a camada de referencia do itinerario.

Por que isso existe
-------------------
A API Olho Vivo NAO devolve o tracado da linha nem a ordem das paradas:

* `/Parada/BuscarParadasPorLinha` devolve `{cp, np, ed, py, px}` — sem nenhum
  campo de sequencia, e a documentacao nunca promete que o array venha
  ordenado. Depender dessa ordem e a falha mais provavel de um projeto assim.
* `/KMZ` **nao** e o tracado das linhas: e o mapa de fluidez do transito
  (velocidade media e tempo de percurso por trecho viario). Util como variavel
  explicativa do atraso, inutil como itinerario.

O GTFS estatico resolve os dois: `shapes.txt` traz a polilinha de cada rota com
`shape_dist_traveled` (metros acumulados, ja pronto para referenciamento
linear) e `stop_times.txt` traz `stop_sequence`.

Chaves entre os dois mundos
---------------------------
    route_id = route_short_name = lt + "-" + tl   (ex.: "8000-10")
                                = campo `c` do Olho Vivo
    trip_id  = route_id + "-" + direction_id
    sl (Olho Vivo)              = direction_id + 1

O `cl` (codigoLinha) da API **nao existe no GTFS** — e um id interno opaco que
muda entre versoes e precisa ser resolvido em runtime via `/Linha/Buscar`.
Nunca persista `cl` como chave estavel entre execucoes.

Os codigos de parada tambem sao renumerados entre versoes do feed, entao o
casamento `cp` <-> `stop_id` e feito por id **e**, quando falha, por
proximidade espacial.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("olhovivo.gtfs")

URL_GTFS = "https://www.sptrans.com.br/umbraco/Surface/PerfilDesenvolvedor/BaixarGTFS"
ARQUIVOS = ("routes.txt", "trips.txt", "stops.txt", "stop_times.txt", "shapes.txt")


# ------------------------------------------------------------------ download
def baixar(destino_dir: Path, url: str = URL_GTFS, forcar: bool = False) -> Path:
    """
    Baixa o ZIP do GTFS e guarda com metadados de proveniencia.

    O feed e regerado diariamente. Guardar SHA-256 e `Last-Modified` a cada
    baixada e o que torna a analise reproduzivel: daqui a seis meses da para
    dizer exatamente qual versao do itinerario foi usada.
    """
    import requests

    destino_dir = Path(destino_dir)
    destino_dir.mkdir(parents=True, exist_ok=True)
    zip_path = destino_dir / "sptrans-gtfs.zip"
    meta_path = destino_dir / "sptrans-gtfs.json"

    if zip_path.exists() and not forcar:
        log.info("GTFS ja baixado em %s (use --forcar para atualizar)", zip_path)
        return zip_path

    log.info("baixando GTFS da SPTrans...")
    r = requests.get(url, timeout=180, stream=True)
    r.raise_for_status()

    temp = zip_path.with_suffix(".zip.part")
    sha = hashlib.sha256()
    tamanho = 0
    with temp.open("wb") as fh:
        for pedaco in r.iter_content(chunk_size=1 << 20):
            fh.write(pedaco)
            sha.update(pedaco)
            tamanho += len(pedaco)
    temp.replace(zip_path)

    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            "o download nao e um ZIP valido — a SPTrans pode ter mudado a URL "
            f"ou exigido login. Verifique {url}"
        )

    meta = {
        "url": url,
        "baixado_em": datetime.now(timezone.utc).isoformat(),
        "bytes": tamanho,
        "sha256": sha.hexdigest(),
        "last_modified": r.headers.get("Last-Modified"),
        "arquivos": sorted(zipfile.ZipFile(zip_path).namelist()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("GTFS: %.1f MB, sha256=%s", tamanho / 1e6, meta["sha256"][:12])
    return zip_path


# ------------------------------------------------------------------- leitura
class GTFS:
    """Tabelas do feed carregadas sob demanda."""

    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"GTFS nao encontrado: {zip_path}")
        self._cache: dict[str, pd.DataFrame] = {}

    def tabela(self, nome: str, **kwargs) -> pd.DataFrame:
        if nome not in self._cache:
            with zipfile.ZipFile(self.zip_path) as z:
                candidatos = [n for n in z.namelist() if n.endswith(nome)]
                if not candidatos:
                    raise KeyError(f"{nome} nao esta no feed")
                with z.open(candidatos[0]) as fh:
                    self._cache[nome] = pd.read_csv(fh, low_memory=False, **kwargs)
        return self._cache[nome]

    # -- atalhos -------------------------------------------------------------
    @property
    def rotas(self) -> pd.DataFrame:
        r = self.tabela("routes.txt")
        # route_type 3 = onibus; o feed mistura metro e CPTM
        if "route_type" in r:
            r = r[r["route_type"] == 3]
        return r

    @property
    def viagens(self) -> pd.DataFrame:
        return self.tabela("trips.txt")

    @property
    def paradas(self) -> pd.DataFrame:
        return self.tabela("stops.txt")

    @property
    def horarios(self) -> pd.DataFrame:
        return self.tabela("stop_times.txt")

    @property
    def formas(self) -> pd.DataFrame:
        return self.tabela("shapes.txt")

    # -- consultas -----------------------------------------------------------
    def viagem_de(self, letreiro_completo: str, sentido: int) -> pd.Series | None:
        """
        A viagem representativa de (route_id, sentido).

        `sentido` e o `sl` do Olho Vivo: 1 ou 2. No GTFS vira direction_id 0/1.
        Entre varias viagens do mesmo sentido, pega a de itinerario mais longo
        (mais paradas) — e a que cobre o percurso completo, sem os atendimentos
        parciais que a SPTrans tambem cadastra.
        """
        v = self.viagens
        direcao = int(sentido) - 1
        alvo = v[(v["route_id"] == letreiro_completo)]
        if "direction_id" in alvo:
            alvo = alvo[alvo["direction_id"] == direcao]
        if alvo.empty:
            return None
        if len(alvo) == 1:
            return alvo.iloc[0]

        contagem = (
            self.horarios[self.horarios["trip_id"].isin(alvo["trip_id"])]
            .groupby("trip_id")
            .size()
        )
        if contagem.empty:
            return alvo.iloc[0]
        return alvo[alvo["trip_id"] == contagem.idxmax()].iloc[0]

    def _indice_formas(self) -> dict:
        """
        Agrupa shapes.txt por shape_id UMA vez.

        O arquivo tem ~1,1 milhao de pontos. Filtrar linearmente a cada consulta
        custa segundos por linha; com 2.900 linhas isso vira horas.
        """
        if not hasattr(self, "_formas_idx"):
            f = self.formas.sort_values(["shape_id", "shape_pt_sequence"])
            self._formas_idx = {k: v for k, v in f.groupby("shape_id", sort=False)}
            log.info("shapes.txt indexado: %d formas", len(self._formas_idx))
        return self._formas_idx

    def polilinha(self, shape_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(lat, lon, dist_acumulada_m) da forma, ja ordenada."""
        s = self._indice_formas().get(shape_id)
        if s is None or len(s) < 2:
            return None
        dist = (
            s["shape_dist_traveled"].to_numpy("float64")
            if "shape_dist_traveled" in s
            else np.full(len(s), np.nan)
        )
        return (
            s["shape_pt_lat"].to_numpy("float64"),
            s["shape_pt_lon"].to_numpy("float64"),
            dist,
        )

    def sequencia_paradas(self, trip_id: str) -> pd.DataFrame:
        """Paradas da viagem na ordem do itinerario, com coordenadas."""
        h = self.horarios
        seq = h[h["trip_id"] == trip_id].sort_values("stop_sequence")
        return seq.merge(self.paradas, on="stop_id", how="left")

    def tracado_de(
        self, letreiro_completo: str, sentido: int
    ) -> dict[str, Any] | None:
        """Traçado + paradas ordenadas de uma linha/sentido do Olho Vivo."""
        viagem = self.viagem_de(letreiro_completo, sentido)
        if viagem is None:
            return None
        shape_id = viagem.get("shape_id")
        forma = self.polilinha(shape_id) if pd.notna(shape_id) else None
        if forma is None:
            return None
        lat, lon, dist = forma
        return {
            "route_id": letreiro_completo,
            "trip_id": viagem["trip_id"],
            "shape_id": shape_id,
            "lat": lat,
            "lon": lon,
            "dist_m": dist,
            "paradas": self.sequencia_paradas(viagem["trip_id"]),
        }


# ------------------------------------------------------------------- geojson
def geojson_da_linha(gtfs: GTFS, route_id: str, sentido: int) -> dict | None:
    """
    Traçado + paradas ordenadas de uma linha, em GeoJSON.

    Nao depende da API nem de token: sai tudo do feed estatico. Serve para
    conferir no mapa (QGIS, geojson.io, kepler.gl) se o itinerario e a ordem
    das paradas estao corretos ANTES de gastar uma semana coletando.
    """
    from . import geo

    dados = gtfs.tracado_de(route_id, sentido)
    if dados is None:
        return None

    proj = geo.obter_projecao()
    xs, ys = proj.para_xy(dados["lat"], dados["lon"])
    ref = geo.ReferenciadorLinear(np.asarray(xs), np.asarray(ys))

    paradas = dados["paradas"].dropna(subset=["stop_lat", "stop_lon"])
    px, py = proj.para_xy(paradas["stop_lat"].to_numpy(), paradas["stop_lon"].to_numpy())
    projecoes = [ref.projetar(float(a), float(b)) for a, b in zip(px, py)]

    feicoes: list[dict] = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(lo), float(la)] for la, lo in zip(dados["lat"], dados["lon"])
                ],
            },
            "properties": {
                "tipo": "tracado",
                "route_id": route_id,
                "sentido": sentido,
                "trip_id": str(dados["trip_id"]),
                "shape_id": str(dados["shape_id"]),
                "comprimento_m": round(ref.comprimento, 1),
                "n_paradas": int(len(paradas)),
            },
        }
    ]

    anterior = None
    for (r, p) in zip(paradas.itertuples(), projecoes):
        espacamento = None if anterior is None else round(p.s - anterior, 1)
        anterior = p.s
        feicoes.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(r.stop_lon), float(r.stop_lat)],
                },
                "properties": {
                    "tipo": "parada",
                    "stop_sequence": int(r.stop_sequence),
                    "stop_id": int(r.stop_id),
                    "stop_name": str(r.stop_name),
                    "dist_no_tracado_m": round(p.s, 1),
                    "espacamento_m": espacamento,
                    "afastamento_do_tracado_m": round(p.distancia, 1),
                },
            }
        )

    return {"type": "FeatureCollection", "features": feicoes}


# --------------------------------------------------------------- ponte de ids
def tabela_ponte(gtfs: GTFS, catalogo_linhas: pd.DataFrame) -> pd.DataFrame:
    """
    Liga o `cl` do Olho Vivo ao `route_id`/`trip_id`/`shape_id` do GTFS.

    Recarregue a cada execucao: o `cl` e opaco e pode mudar quando a SPTrans
    altera a linha.
    """
    if catalogo_linhas.empty:
        return pd.DataFrame()

    cat = catalogo_linhas.copy()
    cat["route_id"] = cat["letreiro_completo"]
    faltando = cat["route_id"].isna()
    if faltando.any() and "tl" in cat:
        cat.loc[faltando, "route_id"] = (
            cat.loc[faltando, "letreiro"].astype(str)
            + "-"
            + cat.loc[faltando, "tl"].astype("Int64").astype(str)
        )

    linhas: list[dict] = []
    for r in cat.itertuples():
        route_id = getattr(r, "route_id", None)
        if not isinstance(route_id, str) or not route_id:
            continue
        viagem = gtfs.viagem_de(route_id, int(r.sentido or 1))
        if viagem is None:
            continue
        linhas.append(
            {
                "cl": int(r.cl),
                "letreiro": r.letreiro,
                "route_id": route_id,
                "sentido": int(r.sentido or 1),
                "direction_id": int(r.sentido or 1) - 1,
                "trip_id": viagem["trip_id"],
                "shape_id": viagem.get("shape_id"),
            }
        )

    ponte = pd.DataFrame(linhas)
    if not ponte.empty:
        log.info(
            "ponte GTFS: %d de %d codigos de linha casados (%.0f%%)",
            len(ponte),
            len(cat),
            100 * len(ponte) / len(cat),
        )
    else:
        log.warning("nenhuma linha do catalogo casou com o GTFS")
    return ponte


def casar_paradas(
    paradas_api: pd.DataFrame, paradas_gtfs: pd.DataFrame, raio_m: float = 80.0
) -> pd.DataFrame:
    """
    Casa `cp` (Olho Vivo) com `stop_id` (GTFS).

    Tenta primeiro por igualdade de codigo; para o que sobrar, casa por
    proximidade — os codigos sao renumerados entre versoes do feed, entao
    igualdade sozinha nao basta.
    """
    from . import geo

    if paradas_api.empty or paradas_gtfs.empty:
        return pd.DataFrame()

    gtfs = paradas_gtfs.dropna(subset=["stop_lat", "stop_lon"]).copy()
    gtfs["stop_id_str"] = gtfs["stop_id"].astype(str)

    api = paradas_api.copy()
    api["cp_str"] = api["cp"].astype(str)

    direto = api.merge(
        gtfs[["stop_id", "stop_id_str", "stop_lat", "stop_lon", "stop_name"]],
        left_on="cp_str",
        right_on="stop_id_str",
        how="left",
    )
    casou_id = direto["stop_id"].notna()

    resultado = direto.assign(metodo=np.where(casou_id, "id", None))

    pendentes = resultado[~casou_id]
    if len(pendentes):
        proj = geo.obter_projecao()
        gx, gy = proj.para_xy(gtfs["stop_lat"].to_numpy(), gtfs["stop_lon"].to_numpy())
        px, py = proj.para_xy(
            pendentes["lat"].to_numpy(), pendentes["lon"].to_numpy()
        )
        try:
            from scipy.spatial import cKDTree

            arvore = cKDTree(np.column_stack([np.asarray(gx), np.asarray(gy)]))
            dist, idx = arvore.query(np.column_stack([np.asarray(px), np.asarray(py)]))
        except ImportError:
            dist = np.full(len(pendentes), np.inf)
            idx = np.zeros(len(pendentes), dtype=int)

        dentro = dist <= raio_m
        posicoes = resultado.index[~casou_id]
        resultado.loc[posicoes[dentro], "stop_id"] = (
            gtfs["stop_id"].to_numpy()[idx[dentro]]
        )
        resultado.loc[posicoes[dentro], "metodo"] = "proximidade"
        resultado.loc[posicoes[dentro], "dist_casamento_m"] = dist[dentro]

    taxa = float(resultado["stop_id"].notna().mean())
    log.info(
        "casamento de paradas cp<->stop_id: %.0f%% (%d por id, %d por proximidade)",
        taxa * 100,
        int((resultado["metodo"] == "id").sum()),
        int((resultado["metodo"] == "proximidade").sum()),
    )
    return resultado


def cobertura_de_paradas(
    gtfs: GTFS, catalogo_linhas, vinculos
) -> "pd.DataFrame":
    """
    Compara quantas paradas a API conhece por linha contra o GTFS.

    Constatacao empirica: `/Parada/BuscarParadasPorLinha` e
    `/Previsao/Linha` devolvem **o mesmo subconjunto** de paradas, e ele e menor
    que o itinerario do GTFS — as vezes muito menor (702U-10: 9 de 40).

    Isso NAO e bug: a API so preve chegada nas paradas que ela cadastra. Mas e
    decisivo na escolha das linhas do estudo, porque so da para medir erro de
    previsao onde existe previsao. Linha com cobertura baixa rende um estudo
    ralo, por mais dias que se colete.
    """
    if catalogo_linhas.empty or vinculos.empty:
        return pd.DataFrame()

    por_linha = vinculos.groupby("cl").size().rename("paradas_api")
    linhas: list[dict] = []

    for r in catalogo_linhas.itertuples():
        cl = int(r.cl)
        if cl not in por_linha.index:
            continue
        route_id = getattr(r, "letreiro_completo", None)
        sentido = int(getattr(r, "sentido", 0) or 1)
        n_gtfs = 0
        dados = gtfs.tracado_de(str(route_id), sentido) if route_id else None
        if dados is not None:
            n_gtfs = int(len(dados["paradas"]))
        n_api = int(por_linha.loc[cl])
        linhas.append(
            {
                "cl": cl,
                "route_id": route_id,
                "sentido": sentido,
                "paradas_api": n_api,
                "paradas_gtfs": n_gtfs,
                "cobertura": round(n_api / n_gtfs, 3) if n_gtfs else None,
            }
        )

    df = pd.DataFrame(linhas)
    return df.sort_values("cobertura", ascending=False) if len(df) else df


# --------------------------------------------------------------- sanidade
def verificar(gtfs: GTFS) -> dict:
    """Checagens baratas que pegam feed corrompido antes de custar uma semana."""
    problemas: list[str] = []
    rotas = gtfs.rotas
    viagens = gtfs.viagens

    if rotas.empty:
        problemas.append("routes.txt sem rotas de onibus (route_type=3)")

    padrao = rotas["route_id"].astype(str).str.match(r"^[0-9A-Z]+-[0-9]+$")
    if not padrao.all():
        problemas.append(
            f"{int((~padrao).sum())} route_id fora do padrao letreiro-digito"
        )

    if "shape_id" in viagens:
        sem_forma = int(viagens["shape_id"].isna().sum())
        if sem_forma:
            problemas.append(f"{sem_forma} viagens sem shape_id")

    resumo = {
        "rotas_onibus": int(len(rotas)),
        "viagens": int(len(viagens)),
        "paradas": int(len(gtfs.paradas)),
        "formas": int(gtfs.formas["shape_id"].nunique()),
        "tem_shape_dist": bool("shape_dist_traveled" in gtfs.formas.columns),
        "problemas": problemas,
    }
    log.info("GTFS verificado: %s", resumo)
    return resumo
