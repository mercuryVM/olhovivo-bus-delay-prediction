"""
Store geoespacial em MongoDB.

Modelagem
---------
| colecao              | tipo          | geometria                | indice          |
|----------------------|---------------|--------------------------|-----------------|
| `linhas`             | normal        | -                        | letreiro, sentido |
| `paradas`            | normal        | `loc` Point              | 2dsphere        |
| `tracados`           | normal        | `geometry` LineString    | 2dsphere        |
| `linha_parada`       | normal        | -                        | (cl, ordem)     |
| `posicoes`           | serie temporal| `loc` Point              | meta + 2dsphere |
| `previsoes`          | serie temporal| -                        | meta            |
| `chegadas`           | normal        | `loc` Point              | 2dsphere        |
| `previsao_realizado` | normal        | `loc` Point              | 2dsphere + erro |
| `regioes`            | normal        | `geometry` Point/Polygon | 2dsphere        |
| `coleta_eventos`     | normal        | -                        | ts              |

Todas as geometrias seguem GeoJSON, com **[longitude, latitude]** nessa ordem —
que e o inverso do que a API Olho Vivo devolve (`py` = lat, `px` = lon). A
conversao acontece num lugar so, em `geo.ponto_geojson`.

Colecoes de serie temporal (MongoDB 5.0+) comprimem muito bem dados de AVL:
mesma chave de meta (linha + prefixo) agrupada em buckets por tempo. Onde o
servidor for antigo demais, o codigo cai para colecao normal sem quebrar.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from .geo import linha_geojson, ponto_geojson

log = logging.getLogger("olhovivo.mongo")

COL_LINHAS = "linhas"
COL_PARADAS = "paradas"
COL_TRACADOS = "tracados"
COL_LINHA_PARADA = "linha_parada"
COL_POSICOES = "posicoes"
COL_PREVISOES = "previsoes"
COL_CHEGADAS = "chegadas"
COL_PREV_REAL = "previsao_realizado"
COL_REGIOES = "regioes"
COL_EVENTOS = "coleta_eventos"


class MongoStore:
    def __init__(self, cfg):
        from pymongo import MongoClient

        uri = cfg.mongo_uri
        if not uri:
            raise RuntimeError("MONGO_URI nao definido")

        self.cfg = cfg
        self.cliente = MongoClient(
            uri,
            serverSelectionTimeoutMS=8000,
            tz_aware=True,
            appname="olhovivo-each",
        )
        # falha rapido se o servidor nao responder
        self.cliente.admin.command("ping")
        self.db = self.cliente[cfg.mongo_db]

        self.usar_ts = bool(cfg.get("mongo.usar_timeseries", True))
        self.granularidade = cfg.get("mongo.granularidade", "seconds")
        self.lote = int(cfg.get("mongo.lote_insercao", 5000))
        self.ttl_dias = int(cfg.get("mongo.ttl_dias_brutos", 0))
        self.brutos = bool(cfg.get("mongo.brutos", False))

        self._buf_pos: list[dict] = []
        self._buf_prev: list[dict] = []
        self._trava = threading.Lock()
        self.stats = {"posicoes": 0, "previsoes": 0, "erros": 0}

        versao = self.cliente.server_info().get("version", "0")
        self.versao_maior = int(str(versao).split(".")[0] or 0)
        log.info("MongoDB %s conectado em %s", versao, cfg.mongo_db)

    # ------------------------------------------------------------- preparacao
    def preparar(self) -> None:
        """Cria colecoes e indices. Idempotente — pode rodar toda vez."""
        from pymongo import ASCENDING, DESCENDING, GEOSPHERE

        existentes = set(self.db.list_collection_names())

        if self.usar_ts and self.versao_maior >= 5:
            self._criar_timeseries(COL_POSICOES, "ta", "meta", existentes)
            self._criar_timeseries(COL_PREVISOES, "ts_coleta", "meta", existentes)

        # -- catalogo
        self.db[COL_LINHAS].create_index([("letreiro", ASCENDING), ("sentido", ASCENDING)])
        self.db[COL_LINHAS].create_index([("terminal_principal", ASCENDING)])

        self.db[COL_PARADAS].create_index([("loc", GEOSPHERE)], name="loc_2dsphere")
        self.db[COL_PARADAS].create_index([("nome", ASCENDING)])
        self.db[COL_PARADAS].create_index([("celula", ASCENDING)])

        self.db[COL_TRACADOS].create_index([("geometry", GEOSPHERE)], name="geom_2dsphere")

        self.db[COL_LINHA_PARADA].create_index(
            [("cl", ASCENDING), ("ordem", ASCENDING)], name="cl_ordem"
        )
        self.db[COL_LINHA_PARADA].create_index([("cp", ASCENDING)])

        # -- fatos brutos (so quando eles sao mesmo gravados aqui)
        if not self.brutos:
            log.info("mongo.brutos=false: posicoes/previsoes ficam so no Parquet")
        else:
            self.db[COL_POSICOES].create_index([("meta.cl", ASCENDING), ("ta", ASCENDING)])
            self.db[COL_POSICOES].create_index(
                [("meta.prefixo", ASCENDING), ("ta", ASCENDING)]
            )
            self._tentar_geo(COL_POSICOES, "loc")
            self.db[COL_PREVISOES].create_index(
                [("meta.cl", ASCENDING), ("ts_coleta", ASCENDING)]
            )
            self.db[COL_PREVISOES].create_index(
                [("meta.cp", ASCENDING), ("ts_coleta", ASCENDING)]
            )

        # -- derivados
        self.db[COL_CHEGADAS].create_index([("loc", GEOSPHERE)], name="loc_2dsphere")
        self.db[COL_CHEGADAS].create_index(
            [("cl", ASCENDING), ("cp", ASCENDING), ("t_chegada", ASCENDING)]
        )
        self.db[COL_CHEGADAS].create_index([("viagem_id", ASCENDING), ("ordem", ASCENDING)])

        self.db[COL_PREV_REAL].create_index([("loc", GEOSPHERE)], name="loc_2dsphere")
        self.db[COL_PREV_REAL].create_index(
            [("cl", ASCENDING), ("cp", ASCENDING), ("t_chegada", ASCENDING)]
        )
        self.db[COL_PREV_REAL].create_index([("celula", ASCENDING), ("faixa", ASCENDING)])
        self.db[COL_PREV_REAL].create_index([("erro_s", DESCENDING)])
        self.db[COL_PREV_REAL].create_index(
            [("horizonte_s", ASCENDING), ("erro_abs_s", DESCENDING)]
        )

        self.db[COL_REGIOES].create_index([("geometry", GEOSPHERE)], name="geom_2dsphere")
        self.db[COL_REGIOES].create_index([("algoritmo", ASCENDING), ("rotulo", ASCENDING)])

        self.db[COL_EVENTOS].create_index([("ts", DESCENDING)])

        if self.ttl_dias > 0:
            self._aplicar_ttl()

        log.info("colecoes e indices prontos em '%s'", self.cfg.mongo_db)

    def _criar_timeseries(
        self, nome: str, campo_tempo: str, campo_meta: str, existentes: set[str]
    ) -> None:
        if nome in existentes:
            return
        try:
            self.db.create_collection(
                nome,
                timeseries={
                    "timeField": campo_tempo,
                    "metaField": campo_meta,
                    "granularity": self.granularidade,
                },
            )
            log.info("colecao de serie temporal criada: %s", nome)
        except Exception as exc:
            log.warning(
                "nao consegui criar %s como serie temporal (%s); usando colecao normal",
                nome,
                exc,
            )

    def _tentar_geo(self, colecao: str, campo: str) -> None:
        """2dsphere em campo de medicao so existe em serie temporal do 6.0+."""
        from pymongo import GEOSPHERE

        try:
            self.db[colecao].create_index([(campo, GEOSPHERE)], name=f"{campo}_2dsphere")
        except Exception as exc:
            log.info("sem indice geo em %s.%s (%s) — normal em Mongo < 6.0", colecao, campo, exc)

    def _aplicar_ttl(self) -> None:
        segundos = self.ttl_dias * 86400
        for colecao, campo in ((COL_POSICOES, "ta"), (COL_PREVISOES, "ts_coleta")):
            try:
                self.db[colecao].create_index(
                    [(campo, 1)], expireAfterSeconds=segundos, name=f"{campo}_ttl"
                )
            except Exception as exc:
                log.warning("TTL nao aplicado em %s: %s", colecao, exc)

    # ------------------------------------------------------------- ingestao
    def inserir_posicoes(self, linhas: Sequence[dict]) -> None:
        docs = [self._doc_posicao(r) for r in linhas]
        with self._trava:
            self._buf_pos.extend(docs)
            cheio = len(self._buf_pos) >= self.lote
        if cheio:
            self._descarregar_posicoes()

    def inserir_previsoes(self, linhas: Sequence[dict]) -> None:
        docs = [self._doc_previsao(r) for r in linhas]
        with self._trava:
            self._buf_prev.extend(docs)
            cheio = len(self._buf_prev) >= self.lote
        if cheio:
            self._descarregar_previsoes()

    @staticmethod
    def _doc_posicao(r: dict) -> dict:
        return {
            "ta": r.get("ta") or r["ts_coleta"],
            "meta": {
                "cl": r["cl"],
                "letreiro": r.get("letreiro"),
                "sentido": r.get("sentido"),
                "prefixo": r.get("prefixo"),
            },
            "ts_coleta": r["ts_coleta"],
            "loc": ponto_geojson(r["lat"], r["lon"]),
            "lat": r["lat"],
            "lon": r["lon"],
            "acessivel": r.get("acessivel"),
            "destino": r.get("destino"),
        }

    @staticmethod
    def _doc_previsao(r: dict) -> dict:
        return {
            "ts_coleta": r["ts_coleta"],
            "meta": {
                "cl": r["cl"],
                "cp": r["cp"],
                "letreiro": r.get("letreiro"),
                "sentido": r.get("sentido"),
                "prefixo": r.get("prefixo"),
            },
            "t_previsto": r.get("t_previsto"),
            "t_previsto_str": r.get("t_previsto_str"),
            "horizonte_s": r.get("horizonte_s"),
            "ta": r.get("ta"),
            "loc_veiculo": ponto_geojson(r["lat"], r["lon"])
            if r.get("lat") is not None
            else None,
            "parada": {
                "nome": r.get("parada_nome"),
                "loc": ponto_geojson(r["parada_lat"], r["parada_lon"])
                if r.get("parada_lat") is not None
                else None,
            },
        }

    def _descarregar_posicoes(self) -> None:
        with self._trava:
            lote, self._buf_pos = self._buf_pos, []
        self._inserir(COL_POSICOES, lote, "posicoes")

    def _descarregar_previsoes(self) -> None:
        with self._trava:
            lote, self._buf_prev = self._buf_prev, []
        self._inserir(COL_PREVISOES, lote, "previsoes")

    def _inserir(self, colecao: str, docs: list[dict], chave_stat: str) -> None:
        if not docs:
            return
        try:
            self.db[colecao].insert_many(docs, ordered=False)
            self.stats[chave_stat] += len(docs)
        except Exception as exc:
            self.stats["erros"] += 1
            log.error("falha ao inserir %d docs em %s: %s", len(docs), colecao, exc)

    def flush(self) -> None:
        self._descarregar_posicoes()
        self._descarregar_previsoes()

    def fechar(self) -> None:
        self.flush()
        self.cliente.close()

    # ------------------------------------------------------------- catalogo
    def upsert_paradas(self, paradas: Iterable[dict]) -> int:
        from pymongo import ReplaceOne

        ops = [
            ReplaceOne(
                {"_id": p["cp"]},
                {
                    "_id": p["cp"],
                    "cp": p["cp"],
                    "nome": p.get("nome"),
                    "endereco": p.get("endereco"),
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "loc": ponto_geojson(p["lat"], p["lon"]),
                    "celula": p.get("celula"),
                    "atualizado_em": p.get("atualizado_em"),
                },
                upsert=True,
            )
            for p in paradas
            if p.get("lat") and p.get("lon")
        ]
        return self._bulk(COL_PARADAS, ops)

    def upsert_linhas(self, linhas: Iterable[dict]) -> int:
        from pymongo import ReplaceOne

        ops = [
            ReplaceOne({"_id": l["cl"]}, {"_id": l["cl"], **l}, upsert=True)
            for l in linhas
        ]
        return self._bulk(COL_LINHAS, ops)

    def upsert_linha_parada(self, vinculos: Iterable[dict]) -> int:
        from pymongo import ReplaceOne

        ops = [
            ReplaceOne(
                {"_id": f"{v['cl']}:{v['cp']}"},
                {"_id": f"{v['cl']}:{v['cp']}", **v},
                upsert=True,
            )
            for v in vinculos
        ]
        return self._bulk(COL_LINHA_PARADA, ops)

    def upsert_tracado(self, cl: int, lats: Sequence[float], lons: Sequence[float], meta: dict) -> None:
        self.db[COL_TRACADOS].replace_one(
            {"_id": cl},
            {
                "_id": cl,
                "cl": cl,
                "geometry": linha_geojson(lats, lons),
                "n_pontos": len(lats),
                **meta,
            },
            upsert=True,
        )

    def _bulk(self, colecao: str, ops: list) -> int:
        if not ops:
            return 0
        res = self.db[colecao].bulk_write(ops, ordered=False)
        return (res.upserted_count or 0) + (res.modified_count or 0)

    # ------------------------------------------------------------- derivados
    def substituir_colecao(
        self, nome: str, docs: Iterable[dict], geo_campo: str | None = "loc"
    ) -> int:
        """
        Regrava uma tabela derivada inteira (chegadas, previsao_realizado).

        Aceita ITERAVEL, nao lista: com milhoes de documentos, materializar tudo
        antes de inserir dobra o pico de memoria sem nenhum ganho.
        """
        import itertools

        col = self.db[nome]
        col.delete_many({})
        fluxo = iter(docs)
        total = 0
        while True:
            fatia = list(itertools.islice(fluxo, self.lote))
            if not fatia:
                break
            col.insert_many(fatia, ordered=False)
            total += len(fatia)
        return total

    # -------------------------------------------------------------- consultas
    def paradas_proximas(self, lat: float, lon: float, raio_m: float = 500.0) -> list[dict]:
        """$geoNear puro: paradas dentro do raio, ja ordenadas por distancia."""
        return list(
            self.db[COL_PARADAS].aggregate(
                [
                    {
                        "$geoNear": {
                            "near": ponto_geojson(lat, lon),
                            "distanceField": "distancia_m",
                            "maxDistance": raio_m,
                            "spherical": True,
                        }
                    }
                ]
            )
        )

    def hotspots_por_celula(self, min_amostras: int = 30, faixa: str | None = None) -> list[dict]:
        """Agrega o erro de previsao por celula H3 — a base do mapa de calor."""
        casar: dict[str, Any] = {"erro_s": {"$ne": None}}
        if faixa:
            casar["faixa"] = faixa

        # $percentile so existe no MongoDB 7.0+; abaixo disso usa desvio padrao
        percentis: dict[str, Any] = (
            {
                "erro_p50_s": {
                    "$percentile": {"input": "$erro_s", "p": [0.5], "method": "approximate"}
                },
                "erro_p90_s": {
                    "$percentile": {"input": "$erro_s", "p": [0.9], "method": "approximate"}
                },
            }
            if self.versao_maior >= 7
            else {"erro_desvio_s": {"$stdDevSamp": "$erro_s"}}
        )

        return list(
            self.db[COL_PREV_REAL].aggregate(
                [
                    {"$match": casar},
                    {
                        "$group": {
                            "_id": "$celula",
                            "n": {"$sum": 1},
                            "erro_medio_s": {"$avg": "$erro_s"},
                            **percentis,
                            "prob_atraso_5min": {
                                "$avg": {"$cond": [{"$gte": ["$erro_s", 300]}, 1, 0]}
                            },
                            "lat": {"$avg": "$lat"},
                            "lon": {"$avg": "$lon"},
                            "linhas": {"$addToSet": "$letreiro"},
                        }
                    },
                    {"$match": {"n": {"$gte": min_amostras}}},
                    {"$sort": {"prob_atraso_5min": -1}},
                ]
            )
        )

    def paradas_criticas(self, min_amostras: int = 30, limite: int = 50) -> list[dict]:
        return list(
            self.db[COL_PREV_REAL].aggregate(
                [
                    {"$match": {"erro_s": {"$ne": None}}},
                    {
                        "$group": {
                            "_id": {"cp": "$cp", "cl": "$cl"},
                            "parada": {"$first": "$parada_nome"},
                            "letreiro": {"$first": "$letreiro"},
                            "n": {"$sum": 1},
                            "erro_medio_s": {"$avg": "$erro_s"},
                            "desvio_s": {"$stdDevSamp": "$erro_s"},
                            "prob_atraso_5min": {
                                "$avg": {"$cond": [{"$gte": ["$erro_s", 300]}, 1, 0]}
                            },
                            "loc": {"$first": "$loc"},
                        }
                    },
                    {"$match": {"n": {"$gte": min_amostras}}},
                    {"$sort": {"prob_atraso_5min": -1, "n": -1}},
                    {"$limit": limite},
                ]
            )
        )

    def cobertura_coleta(self) -> dict:
        """
        Quanto de dado bruto ja entrou, e em que janela.

        NAO use `$group` sem `$match` aqui. A versao anterior fazia isso e virava
        COLLSCAN completo sobre ~100 M de posicoes e ~20 M de previsoes — em
        colecao de serie temporal, obrigando a descomprimir todos os buckets.
        Sao ~25-30 GB puxados pelo cache do WiredTiger a cada chamada, e este e
        justamente o comando que se roda varias vezes por dia acompanhando a
        coleta. Contagem vem dos metadados; extremos vem de dois IXSCAN de uma
        linha cada sobre o timeField.
        """
        def _resumo(colecao: str, campo: str) -> dict:
            col = self.db[colecao]
            try:
                primeiro = next(
                    iter(col.find({}, {campo: 1}).sort(campo, 1).limit(1)), {}
                )
                ultimo = next(
                    iter(col.find({}, {campo: 1}).sort(campo, -1).limit(1)), {}
                )
            except Exception as exc:
                log.warning("nao consegui ler a janela de %s: %s", colecao, exc)
                primeiro = ultimo = {}
            return {
                "n": col.estimated_document_count(),
                "inicio": primeiro.get(campo),
                "fim": ultimo.get(campo),
            }

        return {
            "posicoes": _resumo(COL_POSICOES, "ta"),
            "previsoes": _resumo(COL_PREVISOES, "ts_coleta"),
            "paradas": self.db[COL_PARADAS].estimated_document_count(),
            "linhas": self.db[COL_LINHAS].estimated_document_count(),
        }

    def registrar_evento(self, tipo: str, **dados) -> None:
        try:
            self.db[COL_EVENTOS].insert_one(
                {"ts": datetime.now(timezone.utc), "tipo": tipo, **dados}
            )
        except Exception:
            pass
