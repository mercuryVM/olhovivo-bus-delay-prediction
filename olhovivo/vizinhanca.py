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
    "api_desvio_s", "api_idade_s", "api_trechos",
]


def _parada_com_previsao(df: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada trecho, a primeira parada a partir do destino que a API cobre com
    previsao para a linha (`cp_api`), quantos trechos ha da origem ate ela e o
    tempo tipico desse estirao. Com as paradas do GTFS, parte dos destinos nao
    tem previsao da API; a previsao da proxima parada coberta ainda informa o
    atraso que vem pela frente.
    """
    cobertas = set(zip(prev["cl"].astype("int64"), prev["cp"].astype("int64"))) if "cl" in prev else None
    cl = df["cl"].astype("int64").to_numpy()
    o = df["ordem_origem"].astype("int64").to_numpy()
    d = df["ordem_destino"].astype("int64").to_numpy()
    tip = df["percurso_tipico_s"].to_numpy("float64")
    faixa = df["faixa"].astype(str).to_numpy()
    cp_api = np.full(len(df), np.nan)
    trechos = np.full(len(df), np.nan)
    estirao = np.full(len(df), np.nan)

    for linha in np.unique(cl):
        m = cl == linha
        paradas = (pd.DataFrame({"ordem": d[m], "cp": df["cp_destino"].to_numpy()[m]})
                   .drop_duplicates("ordem").set_index("ordem")["cp"].astype("int64"))
        if cobertas is None:
            ordens_api = paradas.index.to_numpy()
        else:
            ordens_api = np.array(sorted(k for k, c in paradas.items() if (linha, c) in cobertas))
        if len(ordens_api) == 0:
            continue
        # tempo tipico por trecho (ordem de origem) e faixa; sem a faixa, o do trecho
        t = pd.DataFrame({"o": o[m], "f": faixa[m], "tip": tip[m]})
        por_faixa = t.groupby(["f", "o"])["tip"].median()
        por_trecho = t.groupby("o")["tip"].median()
        n = int(max(o[m].max(), d[m].max())) + 1
        base = por_trecho.reindex(range(n)).fillna(por_trecho.median()).to_numpy()
        acumulado = {}
        for f in np.unique(faixa[m]):
            v = base.copy()
            serie = por_faixa.loc[f] if f in por_faixa.index.get_level_values(0) else None
            if serie is not None:
                v[serie.index.to_numpy()] = serie.to_numpy()
            acumulado[f] = np.concatenate([[0.0], np.cumsum(v)])

        pos = np.searchsorted(ordens_api, d[m])
        tem = pos < len(ordens_api)
        a = np.where(tem, ordens_api[np.minimum(pos, len(ordens_api) - 1)], -1)
        idx = np.flatnonzero(m)
        cp_api[idx[tem]] = paradas.reindex(a[tem]).to_numpy()
        trechos[idx[tem]] = a[tem] - o[m][tem]
        cum_o = np.array([acumulado[f][k] for f, k in zip(faixa[m], o[m])])
        cum_a = np.array([acumulado[f][k] if k >= 0 else np.nan for f, k in zip(faixa[m], a)])
        est = cum_a - cum_o
        # com previsao no proprio destino, o estirao e o trecho: tipico da linha
        est = np.where(a == d[m], tip[m], est)
        estirao[idx[tem]] = est[tem]

    return pd.DataFrame({"cp_api": cp_api, "trechos": trechos, "tipico_estirao_s": estirao},
                        index=df.index)


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

    api = pd.DataFrame(np.nan, index=i, columns=["api_desvio_s", "api_idade_s", "api_trechos"])
    if previsoes is not None:
        colunas = ["prefixo", "cp", "ts_coleta", "t_previsto"]
        prev = previsoes if isinstance(previsoes, pd.DataFrame) else pd.read_parquet(
            previsoes, columns=colunas + ["cl"])
        prev = prev.assign(cp=prev["cp"].astype("int64"), prefixo=prev["prefixo"].astype(str),
                           ts_coleta=prev["ts_coleta"].astype(NS),
                           t_previsto=prev["t_previsto"].astype(NS))
        alvo = _parada_com_previsao(df, prev)
        q = pd.DataFrame({"i": i, "prefixo": df["prefixo"].astype(str).to_numpy(),
                          "cp": alvo["cp_api"].to_numpy(), "t": t_origem}).dropna(subset=["cp"])
        q["cp"] = q["cp"].astype("int64")
        m = pd.merge_asof(q.sort_values("t"), prev[colunas].sort_values("ts_coleta"), left_on="t",
                          right_on="ts_coleta", by=["prefixo", "cp"], allow_exact_matches=False,
                          tolerance=VALIDADE_PREVISAO).set_index("i").reindex(i)
        # desvio previsto no estirao ate a parada com previsao, repartido pelo
        # peso do trecho no tempo tipico do estirao (com previsao no destino,
        # o estirao e o proprio trecho e o calculo e o de sempre)
        restante = (m["t_previsto"] - m["t"]).dt.total_seconds().to_numpy()
        tipico = df["percurso_tipico_s"].to_numpy("float64")
        estirao = alvo["tipico_estirao_s"].to_numpy("float64")
        api["api_desvio_s"] = (restante - estirao) * tipico / estirao
        api["api_idade_s"] = (m["t"] - m["ts_coleta"]).dt.total_seconds().to_numpy()
        api["api_trechos"] = alvo["trechos"].to_numpy("float64")
    partes.append(api)

    saida = pd.concat(partes, axis=1)[COLUNAS_VIZINHANCA]
    saida.index = df.index
    return saida.astype("float64")
