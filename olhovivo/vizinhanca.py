"""
Atributos de vizinhanca para o atraso por trecho, todos conhecidos em `t_origem`.

* **Vizinhos na linha:** mediana movel (30 min) do desvio no trecho de cima e
  no de baixo da mesma linha.
* **Trecho fisico:** a mesma mediana no par de paradas, juntando todas as
  linhas que passam por ele.
* **Previsao da API:** a previsao de chegada deste onibus na parada de destino,
  feita ate 5 min antes de `t_origem`, convertida em desvio contra o tempo
  tipico do trecho.

Cada grupo traz tambem quantas observacoes entraram e a idade da mais recente.
Nada usa informacao posterior a `t_origem`: as janelas so veem chegadas
anteriores a ela e a previsao da API precisa ter sido coletada antes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import NS

JANELA = pd.Timedelta("30min")
VALIDADE_PREVISAO = pd.Timedelta("5min")

COLUNAS_VIZINHANCA = [
    "acima_med", "acima_n", "acima_idade",
    "abaixo_med", "abaixo_n", "abaixo_idade",
    "fisico_med", "fisico_n", "fisico_idade",
    "api_desvio_s", "api_idade_s",
]


def _recente(ev: pd.DataFrame, chave: list[str], consulta: pd.DataFrame, nome: str) -> pd.DataFrame:
    """Mediana movel de 30 min do desvio por `chave`, vista estritamente antes de t."""
    ev = ev.sort_values(chave + ["t"]).reset_index(drop=True)
    janela = ev.set_index("t").groupby(chave)["desvio"].rolling(JANELA)
    ev[f"{nome}_med"] = janela.median().to_numpy()
    ev[f"{nome}_n"] = janela.count().to_numpy()
    m = pd.merge_asof(
        consulta.sort_values("t"), ev.rename(columns={"t": "t_ev"}).sort_values("t_ev"),
        left_on="t", right_on="t_ev", by=chave, allow_exact_matches=False, tolerance=JANELA,
    )
    m[f"{nome}_idade"] = (m["t"] - m["t_ev"]).dt.total_seconds()
    return m.set_index("i").sort_index()[[f"{nome}_med", f"{nome}_n", f"{nome}_idade"]]


def atributos(df: pd.DataFrame, previsoes: pd.DataFrame | str | Path | None = None) -> pd.DataFrame:
    """
    Colunas de `COLUNAS_VIZINHANCA` no indice de `df` (a base de
    `modelos.preparar_atraso`, com `alvo_s`). Sem `previsoes`, as colunas da
    API ficam vazias.
    """
    t_origem = pd.to_datetime(df["t_origem"].astype(NS).to_numpy(), utc=True)
    ev = pd.DataFrame({
        "cl": df["cl"].to_numpy(), "ordem": df["ordem_origem"].to_numpy(),
        "cp_o": df["cp_origem"].to_numpy(), "cp_d": df["cp_destino"].to_numpy(),
        "t": pd.to_datetime(df["t_chegada"].astype(NS).to_numpy(), utc=True),
        "desvio": df["alvo_s"].to_numpy(),
    }).dropna(subset=["desvio"])
    i = np.arange(len(df))
    partes = []

    for nome, passo in (("acima", -1), ("abaixo", +1)):
        q = pd.DataFrame({"i": i, "cl": df["cl"].to_numpy(),
                          "ordem": df["ordem_origem"].to_numpy() + passo, "t": t_origem})
        partes.append(_recente(ev[["cl", "ordem", "t", "desvio"]], ["cl", "ordem"], q, nome))

    q = pd.DataFrame({"i": i, "cp_o": df["cp_origem"].to_numpy(),
                      "cp_d": df["cp_destino"].to_numpy(), "t": t_origem})
    partes.append(_recente(ev[["cp_o", "cp_d", "t", "desvio"]], ["cp_o", "cp_d"], q, "fisico"))

    api = pd.DataFrame(np.nan, index=i, columns=["api_desvio_s", "api_idade_s"])
    if previsoes is not None:
        prev = previsoes if isinstance(previsoes, pd.DataFrame) else pd.read_parquet(
            previsoes, columns=["prefixo", "cp", "ts_coleta", "t_previsto"])
        prev = prev.assign(cp=prev["cp"].astype("int64"), prefixo=prev["prefixo"].astype(str),
                           ts_coleta=prev["ts_coleta"].astype(NS),
                           t_previsto=prev["t_previsto"].astype(NS))
        q = pd.DataFrame({"i": i, "prefixo": df["prefixo"].astype(str).to_numpy(),
                          "cp": df["cp_destino"].astype("int64").to_numpy(), "t": t_origem})
        m = pd.merge_asof(q.sort_values("t"), prev.sort_values("ts_coleta"), left_on="t",
                          right_on="ts_coleta", by=["prefixo", "cp"], allow_exact_matches=False,
                          tolerance=VALIDADE_PREVISAO).set_index("i").sort_index()
        api["api_desvio_s"] = ((m["t_previsto"] - m["t"]).dt.total_seconds().to_numpy()
                               - df["percurso_tipico_s"].to_numpy())
        api["api_idade_s"] = (m["t"] - m["ts_coleta"]).dt.total_seconds().to_numpy()
    partes.append(api)

    saida = pd.concat(partes, axis=1)[COLUNAS_VIZINHANCA]
    saida.index = df.index
    return saida.astype("float64")
