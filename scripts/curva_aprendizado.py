"""Curva de aprendizado do XGBoost de atraso: o erro cai com mais dias de treino?

    python scripts/curva_aprendizado.py dados_semana [previsao_realizado.parquet]

O teste e sempre o mesmo de `comparar_atraso.py` (os 30% finais). O treino
usa so os ultimos k dias antes do teste, para k = 1 ate todos. Dois cenarios:

* **tudo com k dias:** o tempo tipico de cada trecho (a referencia do alvo) e o
  modelo saem so desses k dias. E o efeito real de ter mais coleta.
* **so o modelo:** a referencia vem do treino inteiro e so o modelo ve k dias.
  Separa quanto do ganho vem da referencia e quanto vem do modelo.

O MAE do alvo e o MAE do tempo de percurso (o alvo e tempo menos referencia,
a previsao e referencia mais o previsto), entao ele e comparavel entre
cenarios mesmo com referencias diferentes. Para o teste ser o mesmo em todos
os pontos, as metricas usam so os trechos de teste que tem referencia em
todas as rodadas.
"""
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olhovivo import modelos, vizinhanca  # noqa: E402
from olhovivo.atraso import COLUNAS_MODELO  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("curva")
raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "dados")
arq_previsoes = Path(sys.argv[2]) if len(sys.argv) > 2 else raiz / "derivado" / "previsao_realizado.parquet"
saida = raiz / "derivado" / "modelos"
cache = saida / "cache_curva"
cache.mkdir(parents=True, exist_ok=True)
FUSO = "America/Sao_Paulo"

base = pickle.loads((saida / "cache_atraso_v2" / "base.pkl").read_bytes())
viz = pickle.loads((saida / "cache_atraso_v2" / "vizinhanca.pkl").read_bytes())
base["df"] = base["df"].join(viz)
percursos = pd.read_parquet(raiz / "derivado" / "percursos.parquet")
previsoes = pd.read_parquet(arq_previsoes, columns=["prefixo", "cp", "ts_coleta", "t_previsto"])

df_cheio = percursos.dropna(subset=["tempo_percurso_s"]).reset_index(drop=True)
treino_cheio, teste = modelos.divisao_temporal(df_cheio, "t_origem")
dia = pd.to_datetime(treino_cheio["t_origem"], utc=True).dt.tz_convert(FUSO).dt.normalize()
dias = sorted(dia.unique())
log.info("treino: %d dias (%s a %s) | teste: %d trechos", len(dias),
         dias[0].date(), dias[-1].date(), len(teste))


def ajustar(d: pd.DataFrame, i_tr, i_va, i_te) -> np.ndarray:
    colunas = [c for c in COLUNAS_MODELO + vizinhanca.COLUNAS_VIZINHANCA if c in d.columns]
    X = d[colunas].apply(pd.to_numeric, errors="coerce")
    reg = xgb.XGBRegressor(
        objective="reg:absoluteerror", eval_metric="mae", n_estimators=1500, max_depth=6,
        learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        n_jobs=-1, early_stopping_rounds=50,
    )
    reg.fit(X.loc[i_tr], d.loc[i_tr, "alvo_s"], eval_set=[(X.loc[i_va], d.loc[i_va, "alvo_s"])],
            verbose=False)
    return reg.predict(X.loc[i_te])


def rodada(nome: str, calcular):
    arquivo = cache / f"{nome}.pkl"
    if arquivo.exists():
        return pickle.loads(arquivo.read_bytes())
    valor = calcular()
    arquivo.write_bytes(pickle.dumps(valor))
    return valor


resultados = {}
for k in range(1, len(dias) + 1):
    escolhidos = set(dias[-k:])
    sub = treino_cheio[dia.isin(escolhidos)]
    tr, va = modelos.separar_validacao(sub, "t_origem")

    # cenario 1: referencia e modelo so com os k dias
    def tudo():
        d = modelos._atributos_atraso(df_cheio, tr.index, 0.20)
        d = d.join(vizinhanca.atributos(d, previsoes))
        te = teste.index[d.loc[teste.index, "alvo_s"].notna().to_numpy()]
        trv, vav = (x.index[d.loc[x.index, "alvo_s"].notna().to_numpy()] for x in (tr, va))
        p = ajustar(d, trv, vav, te)
        return {"indice": te, "y": d.loc[te, "alvo_s"].to_numpy(), "p": p}

    # cenario 2: referencia do treino inteiro, modelo so com os k dias
    def so_modelo():
        d = base["df"]
        trv, vav = d.index.intersection(tr.index), d.index.intersection(va.index)
        te = base["teste"]
        return {"indice": te, "y": d.loc[te, "alvo_s"].to_numpy(), "p": ajustar(d, trv, vav, te)}

    resultados[k] = {
        "dias": [str(x.date()) for x in sorted(escolhidos)],
        "trechos_treino": len(sub),
        "tudo": rodada(f"tudo_{k}", tudo),
        "so_modelo": rodada(f"so_modelo_{k}", so_modelo),
    }
    log.info("k=%d dias pronto", k)

# mesmos trechos de teste em todos os pontos
comum = teste.index
for r in resultados.values():
    for c in ("tudo", "so_modelo"):
        comum = comum.intersection(r[c]["indice"])


def mae(r, prever_zero=False):
    s = pd.Series(np.abs(r["y"] - (0 if prever_zero else r["p"])), index=r["indice"])
    return float(s.loc[comum].mean())


tabela = []
for k, r in resultados.items():
    tabela.append({
        "k_dias": k, "dias": r["dias"], "trechos_treino": r["trechos_treino"],
        "trivial_mae_s": mae(r["tudo"], prever_zero=True),
        "xgb_tudo_mae_s": mae(r["tudo"]),
        "xgb_so_modelo_mae_s": mae(r["so_modelo"]),
    })

print(f"\nteste comum: {len(comum)} de {len(teste)} trechos")
print("| dias de treino | trechos | trivial (ref. k dias) | XGBoost, tudo com k dias | XGBoost, só o modelo |")
print("|---|---|---|---|---|")
for t in tabela:
    print(f"| {t['k_dias']} ({t['dias'][0][5:]} a {t['dias'][-1][5:]}) | {t['trechos_treino']} | "
          f"{t['trivial_mae_s']:.2f} | {t['xgb_tudo_mae_s']:.2f} | {t['xgb_so_modelo_mae_s']:.2f} |")
(saida / "curva_aprendizado.json").write_text(
    json.dumps({"teste_comum": len(comum), "pontos": tabela}, indent=1, ensure_ascii=False),
    encoding="utf-8")
