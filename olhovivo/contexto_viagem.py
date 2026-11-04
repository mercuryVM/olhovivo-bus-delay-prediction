"""
Atributos da viagem, da tendencia do trecho e do calendario, todos conhecidos
em `t_origem` (scripts/experimento_variaveis.py: juntos tiram 0,23 s do erro
do XGBoost na semana 1).

* **Viagem:** atraso acumulado da viagem ate o trecho anterior e a fracao da
  rota ja percorrida.
* **Tendencia:** desvio mediano do trecho na janela de 30 a 60 min antes de
  `t_origem` e a diferenca para o desvio recente (ultimos 30 min): o transito
  esta piorando ou melhorando.
* **Feriado:** feriados nacionais e de SP no periodo das coletas.
"""

from __future__ import annotations

import pandas as pd

COLUNAS_CONTEXTO = [
    "atraso_acumulado_s", "fracao_rota",
    "desvio_30_60min_s", "tendencia_s",
    "feriado",
]

FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02", "2026-11-15", "2026-11-20"}
JANELA = pd.Timedelta("30min")
NS = "datetime64[ns, UTC]"


def atributos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Colunas de `COLUNAS_CONTEXTO` no indice de `df` (a base de
    `modelos.preparar_atraso`, com `alvo_s` e `desvio_recente_trecho_s`).
    """
    t_origem = pd.to_datetime(df["t_origem"], utc=True).astype(NS)
    t_chegada = pd.to_datetime(df["t_chegada"], utc=True).astype(NS)
    out = pd.DataFrame(index=df.index)

    # trechos anteriores da viagem: todos terminaram ate a parada de origem
    ordem = df[["viagem_id", "ordem_origem", "alvo_s"]].sort_values(["viagem_id", "ordem_origem"])
    out["atraso_acumulado_s"] = (
        ordem.groupby("viagem_id")["alvo_s"].transform(lambda x: x.cumsum().shift(1)).fillna(0)
    )
    out["fracao_rota"] = df["ordem_origem"] / df.groupby("cl")["ordem_destino"].transform("max")

    # mediana movel de 30 min do desvio no trecho, lida em t_origem - 30 min
    ev = (
        pd.DataFrame({"cl": df["cl"], "ordem_origem": df["ordem_origem"],
                      "t": t_chegada, "alvo_s": df["alvo_s"]})
        .dropna().sort_values(["cl", "ordem_origem", "t"]).reset_index(drop=True)
    )
    ev["desvio_30_60min_s"] = (
        ev.set_index("t").groupby(["cl", "ordem_origem"])["alvo_s"].rolling(JANELA).median().to_numpy()
    )
    q = pd.DataFrame({"cl": df["cl"], "ordem_origem": df["ordem_origem"],
                      "t": t_origem - JANELA}).reset_index()
    m = pd.merge_asof(
        q.sort_values("t"),
        ev[["cl", "ordem_origem", "t", "desvio_30_60min_s"]].sort_values("t"),
        on="t", by=["cl", "ordem_origem"], allow_exact_matches=False, tolerance=JANELA,
    )
    out["desvio_30_60min_s"] = m.set_index("index")["desvio_30_60min_s"]
    out["tendencia_s"] = (
        pd.to_numeric(df["desvio_recente_trecho_s"], errors="coerce") - out["desvio_30_60min_s"]
    )

    dia = t_origem.dt.tz_convert("America/Sao_Paulo").dt.strftime("%Y-%m-%d")
    out["feriado"] = dia.isin(FERIADOS).astype("int8")
    return out[COLUNAS_CONTEXTO]
