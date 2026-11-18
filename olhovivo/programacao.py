"""
Intervalo programado entre onibus, do frequencies.txt do GTFS da SPTrans.

O GTFS da SPTrans descreve as viagens por frequencia: uma viagem-modelo por
linha e sentido (trip_id = route_id + "-" + direction_id) e, para cada faixa de
horario, o intervalo em segundos. Quase todas as viagens valem para todos os
dias (service_id USD), entao o intervalo nao muda entre dia util e fim de semana.

Por passagem, na hora local de `t_origem`:

* prog_intervalo_s: intervalo programado da linha e sentido naquela hora
* prog_razao: tempo desde o ultimo onibus no mesmo trecho
  (desde_ultimo_veiculo_s) sobre o intervalo programado. Perto de 0 e comboio;
  bem acima de 1, o onibus anterior passou ha muito mais tempo que o previsto.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("olhovivo.programacao")

COLUNAS_PROG = ["prog_intervalo_s", "prog_razao"]
FUSO = "America/Sao_Paulo"


def _hora(texto: str) -> float:
    h, m, s = (int(x) for x in str(texto).split(":"))
    return h + m / 60 + s / 3600


def intervalos(raiz: Path) -> pd.DataFrame:
    """(cl, hora 0-23, intervalo_s) para as linhas do catalogo."""
    raiz = Path(raiz)
    with zipfile.ZipFile(raiz / "gtfs" / "sptrans-gtfs.zip") as z:
        freq = pd.read_csv(io.TextIOWrapper(z.open("frequencies.txt"), encoding="utf-8-sig"),
                           dtype={"trip_id": str})
    linhas = pd.read_parquet(raiz / "catalogo" / "linhas.parquet", columns=["cl", "letreiro_completo", "sentido"])
    linhas = linhas.dropna()
    linhas["trip_id"] = linhas["letreiro_completo"].astype(str) + "-" + (linhas["sentido"].astype(int) - 1).astype(str)

    partes = []
    for r in freq.itertuples():
        a, b = _hora(r.start_time), _hora(r.end_time)
        # faixa de 00:00 a 00:59 cobre a hora 0; uma de 00:00 a 02:59, as horas 0 a 2
        for h in range(int(a), int(np.floor(b)) + 1):
            partes.append((r.trip_id, h % 24, float(r.headway_secs)))
    tab = pd.DataFrame(partes, columns=["trip_id", "hora", "intervalo_s"])
    tab = tab.groupby(["trip_id", "hora"], as_index=False)["intervalo_s"].min()
    tab = linhas[["cl", "trip_id"]].merge(tab, on="trip_id")
    log.info("programacao: %d linhas-sentido com intervalo, de %d no catalogo",
             tab["cl"].nunique(), linhas["cl"].nunique())
    return tab[["cl", "hora", "intervalo_s"]]


def atributos(df: pd.DataFrame, raiz: Path) -> pd.DataFrame:
    """Colunas de `COLUNAS_PROG` no indice de `df` (percursos com cl, t_origem e desde_ultimo_veiculo_s)."""
    saida = pd.DataFrame(np.nan, index=df.index, columns=COLUNAS_PROG)
    try:
        tab = intervalos(raiz)
    except (FileNotFoundError, KeyError) as exc:
        log.warning("sem programacao do GTFS: %s", exc)
        return saida
    hora = pd.to_datetime(df["t_origem"], utc=True).dt.tz_convert(FUSO).dt.hour
    base = pd.DataFrame({"i": np.arange(len(df)), "cl": df["cl"].astype("int64").to_numpy(),
                         "hora": hora.to_numpy()})
    m = base.merge(tab.astype({"cl": "int64", "hora": "int64"}), on=["cl", "hora"], how="left").sort_values("i")
    saida["prog_intervalo_s"] = m["intervalo_s"].to_numpy()
    if "desde_ultimo_veiculo_s" in df:
        desde = pd.to_numeric(df["desde_ultimo_veiculo_s"], errors="coerce").to_numpy("float64")
        saida["prog_razao"] = desde / saida["prog_intervalo_s"].to_numpy()
    return saida.astype("float64")
