"""Experimento: atributos novos para o atraso por trecho, medidos no XGBoost.

    python scripts/experimento_atributos.py dados_semana [previsao_realizado.parquet]

Mesma base de `modelos.preparar_atraso` (cache de comparar_atraso.py). Cada
atributo novo so usa informacao anterior a `t_origem`.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olhovivo import modelos  # noqa: E402
from olhovivo.atraso import COLUNAS_MODELO  # noqa: E402
from olhovivo.features import NS  # noqa: E402

raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "dados")
arq_prev = Path(sys.argv[2]) if len(sys.argv) > 2 else raiz / "derivado" / "previsao_realizado.parquet"
base = pickle.loads((raiz / "derivado/modelos/cache_atraso/base.pkl").read_bytes())
d = base["df"].copy()
JANELA = pd.Timedelta("30min")


def recente(ev: pd.DataFrame, chave: list[str], consulta: pd.DataFrame, nome: str) -> pd.DataFrame:
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
    return m.set_index("i")[[f"{nome}_med", f"{nome}_n", f"{nome}_idade"]]


ev = pd.DataFrame({
    "cl": d["cl"].to_numpy(), "ordem": d["ordem_origem"].to_numpy(),
    "cp_o": d["cp_origem"].to_numpy(), "cp_d": d["cp_destino"].to_numpy(),
    "t": d["t_chegada"].astype(NS).to_numpy(), "desvio": d["alvo_s"].to_numpy(),
})
ev["t"] = pd.to_datetime(ev["t"], utc=True)
t_origem = pd.to_datetime(d["t_origem"].astype(NS).to_numpy(), utc=True)
inicio = time.perf_counter()

# 1) vizinhos da mesma linha: trecho de cima (k-1) e de baixo (k+1)
for nome, passo in (("acima", -1), ("abaixo", +1)):
    q = pd.DataFrame({"i": np.arange(len(d)), "cl": d["cl"].to_numpy(),
                      "ordem": d["ordem_origem"].to_numpy() + passo, "t": t_origem})
    r = recente(ev[["cl", "ordem", "t", "desvio"]], ["cl", "ordem"], q, nome)
    d[r.columns] = r.sort_index().to_numpy()

# 2) mesmo trecho fisico, todas as linhas
q = pd.DataFrame({"i": np.arange(len(d)), "cp_o": d["cp_origem"].to_numpy(),
                  "cp_d": d["cp_destino"].to_numpy(), "t": t_origem})
r = recente(ev[["cp_o", "cp_d", "t", "desvio"]], ["cp_o", "cp_d"], q, "fisico")
d[r.columns] = r.sort_index().to_numpy()

# 3) previsao da API para a chegada DESTE onibus na parada de destino, feita
# antes de t_origem (consulta mais recente, ate 5 min antes)
prev = pd.read_parquet(arq_prev, columns=["prefixo", "cp", "ts_coleta", "t_previsto"])
prev = prev.assign(cp=prev["cp"].astype("int64"), prefixo=prev["prefixo"].astype(str),
                   ts_coleta=prev["ts_coleta"].astype(NS), t_previsto=prev["t_previsto"].astype(NS))
q = pd.DataFrame({"i": np.arange(len(d)), "prefixo": d["prefixo"].astype(str).to_numpy(),
                  "cp": d["cp_destino"].astype("int64").to_numpy(), "t": t_origem})
m = pd.merge_asof(q.sort_values("t"), prev.sort_values("ts_coleta"), left_on="t",
                  right_on="ts_coleta", by=["prefixo", "cp"], allow_exact_matches=False,
                  tolerance=pd.Timedelta("5min")).set_index("i").sort_index()
api_tempo = (m["t_previsto"] - m["t"]).dt.total_seconds().to_numpy()
d["api_desvio_s"] = api_tempo - d["percurso_tipico_s"].to_numpy()
d["api_idade_s"] = (m["t"] - m["ts_coleta"]).dt.total_seconds().to_numpy()
print(f"atributos em {time.perf_counter() - inicio:.0f} s | cobertura: "
      f"acima {d['acima_med'].notna().mean():.0%}, abaixo {d['abaixo_med'].notna().mean():.0%}, "
      f"fisico {d['fisico_med'].notna().mean():.0%}, api {d['api_desvio_s'].notna().mean():.0%}")

GRUPOS = {
    "vizinhos": ["acima_med", "acima_n", "acima_idade", "abaixo_med", "abaixo_n", "abaixo_idade"],
    "fisico": ["fisico_med", "fisico_n", "fisico_idade"],
    "api": ["api_desvio_s", "api_idade_s"],
}
tr, va, te = (d.loc[base[k]] for k in ("treino", "validacao", "teste"))
y = {k: v["alvo_s"].to_numpy("float32") for k, v in (("tr", tr), ("va", va), ("te", te))}


def treinar(colunas, **param):
    comum = dict(n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=10, n_jobs=-1, early_stopping_rounds=50)
    comum.update(param)
    X = {k: v[colunas].apply(pd.to_numeric, errors="coerce") for k, v in (("tr", tr), ("va", va), ("te", te))}
    reg = xgb.XGBRegressor(objective="reg:absoluteerror", eval_metric="mae", **comum)
    reg.fit(X["tr"], y["tr"], eval_set=[(X["va"], y["va"])], verbose=False)
    p_va, p_te = reg.predict(X["va"]), reg.predict(X["te"])
    return float(np.mean(np.abs(p_va - y["va"]))), float(np.mean(np.abs(p_te - y["te"]))), reg, p_te


base_cols = [c for c in COLUNAS_MODELO if c in d.columns]
res = {}
for nome, cols in (
    ("atual", base_cols),
    ("+vizinhos", base_cols + GRUPOS["vizinhos"]),
    ("+fisico", base_cols + GRUPOS["fisico"]),
    ("+api", base_cols + GRUPOS["api"]),
    ("+tudo", base_cols + sum(GRUPOS.values(), [])),
):
    mva, mte, reg, _ = treinar(cols)
    res[nome] = {"mae_validacao": mva, "mae_teste": mte}
    print(f"{nome:12s} validacao {mva:6.2f} s | teste {mte:6.2f} s")

# ajuste de hiperparametros so na validacao, com todos os atributos
cols = base_cols + sum(GRUPOS.values(), [])
melhor = None
for md in (4, 6, 8, 10):
    for mcw in (5, 20, 50):
        for lr in (0.03, 0.08):
            mva, mte, reg, p = treinar(cols, max_depth=md, min_child_weight=mcw, learning_rate=lr)
            if melhor is None or mva < melhor[0]:
                melhor = (mva, mte, dict(max_depth=md, min_child_weight=mcw, learning_rate=lr), reg, p)
print(f"ajustado {melhor[2]}: validacao {melhor[0]:.2f} s | teste {melhor[1]:.2f} s")
res["ajustado"] = {"mae_validacao": melhor[0], "mae_teste": melhor[1], "param": melhor[2]}
imp = sorted(zip(cols, melhor[3].feature_importances_), key=lambda kv: -kv[1])[:10]
print("importancia:", [(c, round(float(v), 3)) for c, v in imp])
p = melhor[4]
cte = te["alvo_atrasado"].to_numpy()
print("r2 teste:", round(1 - np.sum((p - y["te"]) ** 2) / np.sum((y["te"] - y["te"].mean()) ** 2), 3))
(raiz / "derivado/modelos/experimento_atributos.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
d[sum(GRUPOS.values(), [])].to_parquet(raiz / "derivado/modelos/cache_atraso/atributos_novos.parquet")
