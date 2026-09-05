"""
Casamento previsao x realizado — o dataset central do estudo.

Cada linha do resultado responde: "as HH:MM a API prometeu que o veiculo P
chegaria na parada C as HH:MM; ele chegou as HH:MM; o erro foi de N segundos".

Como o par e formado
--------------------
A previsao e identificada por (linha, parada, prefixo do veiculo). A chegada
casada e a **primeira chegada daquele mesmo veiculo naquela parada em ou apos
o instante da consulta** (com uma folga de `tolerancia_passado_s`, porque o
veiculo pode ter cruzado a parada entre a geracao da previsao e o momento em
que ela foi lida). Isso e feito com `merge_asof(direction="forward")`, que e o
join temporal correto para esse caso — nao um join por igualdade nem por janela
fixa.

Pares com erro absurdo (acima de `erro_max_abs_s`) sao descartados: quase
sempre indicam que o veiculo trocou de linha, entrou em garagem, ou que a
deteccao de chegada perdeu o evento.

Sobre "atraso"
--------------
A SPTrans nao publica tabela de horarios pela API, entao nao existe "atraso
contra o programado". O que existe, e e mais util para o passageiro, sao tres
medidas — todas produzidas aqui ou em `features.py`:

* `erro_s` — erro da previsao (chegada real menos previsto). Positivo = a API
  prometeu antes do que aconteceu; e o atraso percebido por quem esta no ponto.
* `headway_obs_s` — intervalo real entre onibus consecutivos da linha naquela
  parada. Regularidade importa mais que pontualidade em servico de alta
  frequencia.
* desvio de tempo de percurso — comparacao do trecho contra a mediana daquele
  mesmo trecho na mesma faixa horaria (em `features.py`).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from . import geo
from .coleta import TZ_SP
from .storage import Armazenamento

log = logging.getLogger("olhovivo.casamento")

FAIXAS = [
    (0, 6, "madrugada"),
    (6, 9, "pico_manha"),
    (9, 16, "entrepico"),
    (16, 20, "pico_tarde"),
    (20, 24, "noite"),
]


def faixa_horaria(hora: int) -> str:
    for ini, fim, nome in FAIXAS:
        if ini <= hora < fim:
            return nome
    return "indefinida"


def _headway(chegadas: pd.DataFrame) -> pd.DataFrame:
    """Intervalo, em segundos, ate o onibus anterior da mesma linha na parada."""
    ch = chegadas.sort_values(["cl", "cp", "t_chegada"]).copy()
    ch["headway_obs_s"] = (
        ch.groupby(["cl", "cp"], sort=False)["t_chegada"].diff().dt.total_seconds()
    )
    return ch


def executar(
    cfg,
    arm: Armazenamento,
    inicio: datetime | None = None,
    fim: datetime | None = None,
    conf_minima: float = 0.35,
) -> dict:
    horizonte_max = int(cfg.get("casamento.horizonte_max_s", 3600))
    tol_passado = int(cfg.get("casamento.tolerancia_passado_s", 60))
    erro_max = int(cfg.get("casamento.erro_max_abs_s", 2700))
    resolucao_h3 = int(cfg.get("geo.h3_resolucao", 9))

    chegadas = arm.ler_derivado("chegadas")
    if chegadas.empty:
        raise RuntimeError(
            "nenhuma chegada detectada: rode `python -m olhovivo chegadas` antes"
        )
    chegadas["t_chegada"] = pd.to_datetime(chegadas["t_chegada"], utc=True)
    chegadas = chegadas[chegadas["confianca"] >= conf_minima]
    chegadas = _headway(chegadas)

    previsoes = arm.ler_bruto(
        "previsoes",
        inicio=inicio,
        fim=fim,
        colunas=[
            "ts_coleta",
            "cl",
            "letreiro",
            "sentido",
            "cp",
            "parada_nome",
            "prefixo",
            "t_previsto",
            "horizonte_s",
        ],
    )
    if previsoes.empty:
        raise RuntimeError("nenhuma previsao coletada no periodo informado")

    previsoes = previsoes.dropna(subset=["t_previsto", "prefixo"])
    previsoes["ts_coleta"] = pd.to_datetime(previsoes["ts_coleta"], utc=True)
    previsoes["t_previsto"] = pd.to_datetime(previsoes["t_previsto"], utc=True)
    previsoes = previsoes[
        (previsoes["horizonte_s"] >= 0) & (previsoes["horizonte_s"] <= horizonte_max)
    ]

    # uma previsao por (linha, parada, veiculo, instante de consulta)
    previsoes = previsoes.drop_duplicates(
        subset=["cl", "cp", "prefixo", "ts_coleta"], keep="last"
    )

    # merge_asof exige a MESMA resolucao de datetime nos dois lados; o Parquet
    # devolve ms e a aritmetica com Timedelta promove para us. Normaliza em ns.
    NS = "datetime64[ns, UTC]"

    esq = previsoes.assign(
        prefixo=previsoes["prefixo"].astype(str),
        cl=previsoes["cl"].astype("int64"),
        cp=previsoes["cp"].astype("int64"),
        chave_tempo=(
            previsoes["ts_coleta"] - pd.Timedelta(seconds=tol_passado)
        ).astype(NS),
    ).sort_values("chave_tempo")

    dir_ = chegadas.assign(
        prefixo=chegadas["prefixo"].astype(str),
        cl=chegadas["cl"].astype("int64"),
        cp=chegadas["cp"].astype("int64"),
        t_chegada=chegadas["t_chegada"].astype(NS),
    ).sort_values("t_chegada")

    # A celula depende so de (lat, lon), que vem da CHEGADA. Calcular aqui, nas
    # ~176 mil chegadas, em vez de depois do join nos ~5 milhoes de pares: sao
    # 28x menos chamadas para exatamente o mesmo resultado.
    dir_["celula"] = [
        geo.celula(float(a), float(b), resolucao_h3)
        for a, b in zip(dir_["lat"], dir_["lon"])
    ]

    log.info(
        "casando %d previsoes com %d chegadas (confianca >= %.2f)",
        len(esq),
        len(dir_),
        conf_minima,
    )

    pares = pd.merge_asof(
        esq,
        dir_[
            [
                "cl",
                "cp",
                "prefixo",
                "t_chegada",
                "viagem_id",
                "ordem",
                "confianca",
                "headway_obs_s",
                "lat",
                "lon",
                "celula",
            ]
        ],
        left_on="chave_tempo",
        right_on="t_chegada",
        by=["cl", "cp", "prefixo"],
        direction="forward",
        tolerance=pd.Timedelta(seconds=horizonte_max),
    )

    total_bruto = len(pares)
    pares = pares.dropna(subset=["t_chegada"])
    casadas = len(pares)

    pares["erro_s"] = (pares["t_chegada"] - pares["t_previsto"]).dt.total_seconds()
    pares = pares[pares["erro_s"].abs() <= erro_max]

    # /Previsao/Linha devolve `np` VAZIO (confirmado nos dados coletados):
    # o nome da parada so vem em /Parada/*. Preenche pelo catalogo, senao o
    # dataset final sai com a coluna em branco.
    catalogo_paradas = arm.ler_catalogo("paradas")
    if len(catalogo_paradas):
        nomes = dict(
            zip(catalogo_paradas["cp"].astype("int64"), catalogo_paradas["nome"])
        )
        vazio = pares["parada_nome"].isna() | (
            pares["parada_nome"].astype(str).str.strip() == ""
        )
        if vazio.any():
            pares.loc[vazio, "parada_nome"] = pares.loc[vazio, "cp"].map(nomes)
            log.info(
                "nome da parada preenchido pelo catalogo em %d linhas", int(vazio.sum())
            )

    local = pares["t_chegada"].dt.tz_convert(TZ_SP)
    pares["dia_semana"] = local.dt.dayofweek.astype("int8")
    pares["hora"] = local.dt.hour.astype("int8")
    pares["faixa"] = pd.cut(
        pares["hora"],
        bins=[f[0] for f in FAIXAS] + [24],
        right=False,
        labels=[f[2] for f in FAIXAS],
    ).astype(str)
    pares["erro_abs_s"] = pares["erro_s"].abs()

    saida = pares[
        [
            "cl",
            "letreiro",
            "sentido",
            "cp",
            "parada_nome",
            "prefixo",
            "viagem_id",
            "ts_coleta",
            "t_previsto",
            "t_chegada",
            "horizonte_s",
            "erro_s",
            "erro_abs_s",
            "headway_obs_s",
            "lat",
            "lon",
            "celula",
            "dia_semana",
            "hora",
            "faixa",
            "confianca",
        ]
    ].copy()

    for coluna in ("horizonte_s", "erro_s", "erro_abs_s", "headway_obs_s"):
        saida[coluna] = saida[coluna].fillna(0).round().astype("int32")
    saida["sentido"] = saida["sentido"].fillna(0).astype("int8")
    saida["confianca"] = saida["confianca"].astype("float32")

    # grava direto do DataFrame: `to_dict("records")` em 5 M de linhas custa
    # ~6 GB de RAM so para o pyarrow reconverter tudo de volta
    arm.salvar_tabela_df("previsao_realizado", saida)

    if arm.mongo:
        # gerador, nao lista: os documentos sao consumidos em lotes e nao
        # coexistem todos em memoria com o DataFrame
        def _docs():
            for r in saida.itertuples(index=False):
                d = r._asdict()
                d["loc"] = geo.ponto_geojson(d["lat"], d["lon"])
                for campo in ("ts_coleta", "t_previsto", "t_chegada"):
                    d[campo] = d[campo].to_pydatetime()
                yield d

        arm.mongo.substituir_colecao("previsao_realizado", _docs())

    resumo = {
        "previsoes_avaliadas": total_bruto,
        "casadas": casadas,
        "taxa_casamento": round(casadas / total_bruto, 3) if total_bruto else 0.0,
        "validas": len(saida),
        "erro_medio_s": round(float(saida["erro_s"].mean()), 1) if len(saida) else 0.0,
        "erro_mediano_s": round(float(saida["erro_s"].median()), 1) if len(saida) else 0.0,
        "mae_s": round(float(saida["erro_abs_s"].mean()), 1) if len(saida) else 0.0,
        "rmse_s": round(float(np.sqrt((saida["erro_s"] ** 2).mean())), 1) if len(saida) else 0.0,
        "prob_atraso_acima_5min": round(float((saida["erro_s"] >= 300).mean()), 4)
        if len(saida)
        else 0.0,
        "prob_adiantado_1min": round(float((saida["erro_s"] <= -60).mean()), 4)
        if len(saida)
        else 0.0,
    }
    arm.eventos.registrar("casamento_concluido", **resumo)
    log.info("casamento: %s", resumo)
    return resumo


def resumo_por_linha(arm: Armazenamento, min_amostras: int = 50) -> pd.DataFrame:
    """Ranking das linhas por probabilidade de atraso."""
    df = arm.ler_derivado("previsao_realizado")
    if df.empty:
        return df
    agrupado = (
        df.groupby(["cl", "letreiro", "sentido"], dropna=False)
        .agg(
            n=("erro_s", "size"),
            erro_medio_s=("erro_s", "mean"),
            erro_mediano_s=("erro_s", "median"),
            mae_s=("erro_abs_s", "mean"),
            p90_erro_s=("erro_s", lambda x: float(np.percentile(x, 90))),
            prob_atraso_5min=("erro_s", lambda x: float((x >= 300).mean())),
            headway_mediano_s=("headway_obs_s", "median"),
        )
        .reset_index()
    )
    agrupado = agrupado[agrupado["n"] >= min_amostras]
    return agrupado.sort_values("prob_atraso_5min", ascending=False)
