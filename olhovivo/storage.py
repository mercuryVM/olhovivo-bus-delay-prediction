"""
Armazenamento.

Dois backends, com papeis diferentes e complementares:

* **Parquet** (local, append-only, particionado por dia/hora) e o arquivo
  durAvel. Sobrevive a queda de energia, notebook dormindo e Ctrl+C: cada
  descarga vira um arquivo fechado, escrito em `.part` e renomeado atomicamente,
  e um manifesto registra a janela temporal de cada arquivo — e assim que a
  analise sabe quais minutos ficaram sem coleta.

* **MongoDB** e o store consultavel, com indice geoespacial 2dsphere e colecoes
  de serie temporal. E nele que se faz "quais paradas num raio de 500 m tiveram
  atraso mediano acima de X entre 17h e 19h".

Perder o Mongo nao perde dado: da para reconstruir tudo a partir do Parquet.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

log = logging.getLogger("olhovivo.storage")

TS = pa.timestamp("ms", tz="UTC")


# ------------------------------------------------------------------ esquemas
ESQUEMA_POSICOES = pa.schema(
    [
        ("ts_coleta", TS),                  # quando NOS consultamos a API
        ("hr_api", pa.string()),            # hora de referencia devolvida (HH:MM)
        ("cl", pa.int32()),                 # codigo da linha (por sentido)
        ("letreiro", pa.dictionary(pa.int32(), pa.string())),
        ("sentido", pa.int8()),
        ("destino", pa.dictionary(pa.int32(), pa.string())),
        ("prefixo", pa.dictionary(pa.int32(), pa.string())),
        ("acessivel", pa.bool_()),
        ("ta", TS),                         # timestamp do GPS do veiculo (UTC)
        ("lat", pa.float64()),
        ("lon", pa.float64()),
    ]
)

ESQUEMA_PREVISOES = pa.schema(
    [
        ("ts_coleta", TS),
        ("hr_api", pa.string()),
        ("cl", pa.int32()),
        ("letreiro", pa.dictionary(pa.int32(), pa.string())),
        ("sentido", pa.int8()),
        ("cp", pa.int32()),                 # codigo da parada
        ("parada_nome", pa.dictionary(pa.int32(), pa.string())),
        ("parada_lat", pa.float64()),
        ("parada_lon", pa.float64()),
        ("prefixo", pa.dictionary(pa.int32(), pa.string())),
        ("acessivel", pa.bool_()),
        ("t_previsto_str", pa.string()),    # "HH:MM" cru, como veio
        ("t_previsto", TS),                 # resolvido para instante absoluto
        ("horizonte_s", pa.int32()),        # t_previsto - ts_coleta
        ("ta", TS),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
    ]
)

ESQUEMA_LINHAS = pa.schema(
    [
        ("cl", pa.int32()),
        ("letreiro", pa.string()),
        ("letreiro_completo", pa.string()),
        ("tl", pa.int32()),
        ("sentido", pa.int8()),
        ("circular", pa.bool_()),
        ("terminal_principal", pa.string()),
        ("terminal_secundario", pa.string()),
        ("atualizado_em", TS),
    ]
)

ESQUEMA_PARADAS = pa.schema(
    [
        ("cp", pa.int32()),
        ("nome", pa.string()),
        ("endereco", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("celula", pa.string()),
        ("atualizado_em", TS),
    ]
)

ESQUEMA_LINHA_PARADA = pa.schema(
    [
        ("cl", pa.int32()),
        ("cp", pa.int32()),
        ("ordem_api", pa.int32()),          # ordem como veio do endpoint
        ("ordem", pa.int32()),              # ordem corrigida pelo tracado
        ("s_m", pa.float64()),              # abscissa da parada no tracado
        ("dist_tracado_m", pa.float64()),
        ("atualizado_em", TS),
    ]
)

ESQUEMA_CHEGADAS = pa.schema(
    [
        ("cl", pa.int32()),
        ("letreiro", pa.string()),
        ("sentido", pa.int8()),
        ("prefixo", pa.string()),
        ("viagem_id", pa.string()),
        ("cp", pa.int32()),
        ("ordem", pa.int32()),
        ("t_chegada", TS),
        ("metodo", pa.string()),            # "cruzamento" | "aproximacao"
        ("dist_min_m", pa.float64()),
        ("salto_m", pa.float64()),          # distancia entre as amostras usadas
        ("salto_s", pa.float64()),
        ("velocidade_kmh", pa.float64()),
        ("confianca", pa.float32()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
    ]
)

ESQUEMA_PREV_REAL = pa.schema(
    [
        ("cl", pa.int32()),
        ("letreiro", pa.string()),
        ("sentido", pa.int8()),
        ("cp", pa.int32()),
        ("parada_nome", pa.string()),
        ("prefixo", pa.string()),
        ("viagem_id", pa.string()),
        ("ts_coleta", TS),
        ("t_previsto", TS),
        ("t_chegada", TS),
        ("horizonte_s", pa.int32()),        # antecedencia da previsao
        ("erro_s", pa.int32()),             # chegada - previsto (+ = atrasou)
        ("erro_abs_s", pa.int32()),
        ("headway_obs_s", pa.int32()),      # intervalo ate o veiculo anterior
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("celula", pa.string()),
        ("dia_semana", pa.int8()),
        ("hora", pa.int8()),
        ("faixa", pa.string()),             # pico_manha | entrepico | ...
        ("confianca", pa.float32()),
    ]
)

ESQUEMAS = {
    "posicoes": ESQUEMA_POSICOES,
    "previsoes": ESQUEMA_PREVISOES,
    "chegadas": ESQUEMA_CHEGADAS,
    "previsao_realizado": ESQUEMA_PREV_REAL,
}


def agora_utc() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ escritor
class EscritorParquet:
    """Buffer em memoria que vira arquivo Parquet fechado a cada rotacao."""

    def __init__(
        self,
        raiz: Path,
        nome: str,
        esquema: pa.Schema,
        rotacao_s: float = 300.0,
        rotacao_linhas: int = 250_000,
    ):
        self.dir_base = Path(raiz) / "bruto" / nome
        self.dir_base.mkdir(parents=True, exist_ok=True)
        self.nome = nome
        self.esquema = esquema
        self.rotacao_s = rotacao_s
        self.rotacao_linhas = rotacao_linhas
        self.manifesto = self.dir_base / "_manifesto.jsonl"

        self._buffer: list[dict] = []
        self._trava = threading.Lock()
        self._ultimo_flush = agora_utc()
        self._seq = 0
        self.total_gravado = 0

    def adicionar(self, linhas: Sequence[dict]) -> None:
        if not linhas:
            return
        with self._trava:
            self._buffer.extend(linhas)
            precisa = (
                len(self._buffer) >= self.rotacao_linhas
                or (agora_utc() - self._ultimo_flush).total_seconds() >= self.rotacao_s
            )
        if precisa:
            self.flush()

    def flush(self) -> Path | None:
        with self._trava:
            if not self._buffer:
                self._ultimo_flush = agora_utc()
                return None
            lote = self._buffer
            self._buffer = []
            self._seq += 1
            seq = self._seq
            self._ultimo_flush = agora_utc()

        agora = agora_utc()
        destino_dir = self.dir_base / f"dt={agora:%Y-%m-%d}" / f"h={agora:%H}"
        destino_dir.mkdir(parents=True, exist_ok=True)
        nome_arq = f"{self.nome}-{agora:%Y%m%dT%H%M%S}-{seq:05d}.parquet"
        final = destino_dir / nome_arq
        temp = destino_dir / (nome_arq + ".part")

        tabela = pa.Table.from_pylist(lote, schema=self.esquema)

        # ordenar por (linha, veiculo, tempo) antes de gravar deixa valores
        # iguais adjacentes: o zstd cai bastante e a leitura por linha fica
        # mais rapida. Custa quase nada num lote de alguns minutos.
        chaves = [
            c for c in ("cl", "prefixo", "ta", "ts_coleta") if c in self.esquema.names
        ]
        if chaves:
            try:
                tabela = tabela.sort_by([(c, "ascending") for c in chaves])
            except Exception:
                pass

        pq.write_table(
            tabela,
            temp,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            version="2.6",
            row_group_size=1_000_000,
            write_page_index=True,
        )
        os.replace(temp, final)
        self.total_gravado += len(lote)

        campo_ts = "ts_coleta" if "ts_coleta" in self.esquema.names else None
        janela = {}
        if campo_ts:
            col = tabela.column(campo_ts)
            try:
                janela = {
                    "ts_min": str(pc.min(col).as_py()),
                    "ts_max": str(pc.max(col).as_py()),
                }
            except Exception:
                janela = {}

        self._anotar_manifesto(
            {
                "arquivo": str(final.relative_to(self.dir_base)),
                "linhas": len(lote),
                "gravado_em": agora.isoformat(),
                **janela,
            }
        )
        log.debug("%s: %d linhas -> %s", self.nome, len(lote), final.name)
        return final

    def _anotar_manifesto(self, registro: dict) -> None:
        with self.manifesto.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def fechar(self) -> None:
        self.flush()


class RegistroEventos:
    """Log estruturado da coleta: inicio, parada, erro, lacuna, reautenticacao."""

    def __init__(self, raiz: Path, nome: str = "coleta"):
        d = Path(raiz) / "logs"
        d.mkdir(parents=True, exist_ok=True)
        self.caminho = d / f"{nome}.jsonl"
        self._trava = threading.Lock()

    def registrar(self, tipo: str, **dados: Any) -> None:
        registro = {"ts": agora_utc().isoformat(), "tipo": tipo, **dados}
        with self._trava, self.caminho.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
            fh.flush()


# ------------------------------------------------------------------- fachada
class Armazenamento:
    """Escreve nos backends configurados (parquet, mongo) de uma vez so."""

    def __init__(self, cfg, backends: Iterable[str] | None = None):
        self.cfg = cfg
        self.raiz = cfg.dir_dados
        self.backends = list(backends if backends is not None else cfg.backends)
        self.eventos = RegistroEventos(self.raiz)

        self._escritores: dict[str, EscritorParquet] = {}
        self.mongo = None

        if "mongo" in self.backends:
            from .mongo import MongoStore

            try:
                self.mongo = MongoStore(cfg)
                self.mongo.preparar()
            except Exception as exc:
                log.error("MongoDB indisponivel (%s); seguindo so com Parquet", exc)
                self.eventos.registrar("mongo_indisponivel", erro=str(exc))
                self.mongo = None
                self.backends = [b for b in self.backends if b != "mongo"]

    # -- escritores parquet sob demanda -------------------------------------
    def _escritor(self, nome: str) -> EscritorParquet:
        if nome not in self._escritores:
            self._escritores[nome] = EscritorParquet(
                self.raiz,
                nome,
                ESQUEMAS[nome],
                rotacao_s=self.cfg.get("armazenamento.rotacao_s", 300),
                rotacao_linhas=self.cfg.get("armazenamento.rotacao_linhas", 250_000),
            )
        return self._escritores[nome]

    # -- fatos ---------------------------------------------------------------
    def escrever_posicoes(self, linhas: Sequence[dict]) -> None:
        if not linhas:
            return
        if "parquet" in self.backends:
            self._escritor("posicoes").adicionar(linhas)
        if self.mongo:
            self.mongo.inserir_posicoes(linhas)

    def escrever_previsoes(self, linhas: Sequence[dict]) -> None:
        if not linhas:
            return
        if "parquet" in self.backends:
            self._escritor("previsoes").adicionar(linhas)
        if self.mongo:
            self.mongo.inserir_previsoes(linhas)

    # -- derivados -----------------------------------------------------------
    def salvar_tabela(self, nome: str, linhas: Sequence[dict]) -> Path | None:
        """Grava uma tabela derivada inteira (chegadas, previsao_realizado)."""
        if not linhas:
            return None
        destino_dir = self.raiz / "derivado"
        destino_dir.mkdir(parents=True, exist_ok=True)
        destino = destino_dir / f"{nome}.parquet"
        temp = destino.with_suffix(".parquet.part")
        tabela = pa.Table.from_pylist(list(linhas), schema=ESQUEMAS.get(nome))
        pq.write_table(tabela, temp, compression="zstd")
        os.replace(temp, destino)
        return destino

    # -- catalogo ------------------------------------------------------------
    def salvar_catalogo(self, nome: str, linhas: Sequence[dict], esquema: pa.Schema) -> Path:
        destino_dir = self.raiz / "catalogo"
        destino_dir.mkdir(parents=True, exist_ok=True)
        destino = destino_dir / f"{nome}.parquet"
        temp = destino.with_suffix(".parquet.part")
        pq.write_table(
            pa.Table.from_pylist(list(linhas), schema=esquema), temp, compression="zstd"
        )
        os.replace(temp, destino)
        return destino

    def ler_catalogo(self, nome: str):
        import pandas as pd

        caminho = self.raiz / "catalogo" / f"{nome}.parquet"
        if not caminho.exists():
            return pd.DataFrame()
        return pq.read_table(caminho).to_pandas()

    def ler_derivado(self, nome: str):
        import pandas as pd

        caminho = self.raiz / "derivado" / f"{nome}.parquet"
        if not caminho.exists():
            return pd.DataFrame()
        return pq.read_table(caminho).to_pandas()

    # -- leitura dos brutos --------------------------------------------------
    def ler_bruto(
        self,
        nome: str,
        inicio: datetime | None = None,
        fim: datetime | None = None,
        colunas: list[str] | None = None,
        filtro_extra: str | None = None,
    ):
        """Le o dataset particionado com DuckDB (nao carrega tudo na RAM a toa)."""
        import duckdb
        import pandas as pd

        base = self.raiz / "bruto" / nome
        if not base.exists():
            return pd.DataFrame()

        padrao = str(base / "**" / "*.parquet").replace("\\", "/")
        cols = ", ".join(colunas) if colunas else "*"
        onde: list[str] = []
        if inicio:
            onde.append(f"ts_coleta >= TIMESTAMPTZ '{inicio.isoformat()}'")
        if fim:
            onde.append(f"ts_coleta < TIMESTAMPTZ '{fim.isoformat()}'")
        if filtro_extra:
            onde.append(f"({filtro_extra})")
        clausula = (" WHERE " + " AND ".join(onde)) if onde else ""

        con = duckdb.connect()
        try:
            return con.execute(
                f"SELECT {cols} FROM read_parquet('{padrao}', union_by_name=true)"
                f"{clausula}"
            ).df()
        finally:
            con.close()

    # -- ciclo de vida -------------------------------------------------------
    def flush(self) -> None:
        for esc in self._escritores.values():
            esc.flush()
        if self.mongo:
            self.mongo.flush()

    def fechar(self) -> None:
        for esc in self._escritores.values():
            esc.fechar()
        if self.mongo:
            self.mongo.fechar()

    def resumo(self) -> dict:
        return {
            nome: esc.total_gravado for nome, esc in self._escritores.items()
        } | ({"mongo": self.mongo.stats} if self.mongo else {})
