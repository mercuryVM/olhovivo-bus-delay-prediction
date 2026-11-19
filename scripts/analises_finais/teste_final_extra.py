"""Metricas extras do teste final a partir das previsoes guardadas (sem retreinar). So leitura.

- Cobertura do intervalo de 80% (quantis 10 e 90) da LSTM, geral e por dia, e largura mediana.
- F1 e AUC da classificacao atrasado/nao atrasado derivada da previsao de cada modelo.
- Bootstrap por viagem (1000 reamostragens) das diferencas de MAE em relacao ao XGBoost.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/dados/derivado/modelos/cache_diaria")
FUSO = "America/Sao_Paulo"
DIAS = [f"2026-11-0{d}" for d in range(2, 9)]
pd.set_option("display.width", 220)

pc = pd.read_parquet("/dados/derivado/percursos.parquet", columns=["viagem_id", "tempo_percurso_s", "t_origem"])
pc = pc.dropna(subset=["tempo_percurso_s"]).reset_index(drop=True)
print("arquivos no cache:", len(list(CACHE.glob("*.pkl"))))

partes = []
for dia in DIAS:
    r = pickle.loads((CACHE / f"todos_frota_{dia}_redes.pkl").read_bytes())
    extra = {}
    for nome in ("todos_frota_sem_api", "todos_frota_geo"):
        arq = CACHE / f"{nome}_{dia}.pkl"
        if arq.exists():
            extra[nome] = pickle.loads(arq.read_bytes())
    comum = pd.Index(r["indice"])
    for e in extra.values():
        comum = comum.intersection(e["indice"])
    pos = pd.Index(r["indice"]).get_indexer(comum)
    df = pd.DataFrame({"dia": dia, "alvo": r["alvo"][pos],
                       "q10": r["quantis_lstm"][pos, 0], "q90": r["quantis_lstm"][pos, 2]}, index=comum)
    for m, v in r["previsto_por_modelo"].items():
        df[m] = v[pos]
    for nome, e in extra.items():
        df[nome.replace("todos_frota_", "xgb_")] = e["previsto"][pd.Index(e["indice"]).get_indexer(comum)]
    partes.append(df)
t = pd.concat(partes)
t["viagem_id"] = pc.loc[t.index, "viagem_id"].to_numpy()
t["tempo"] = pc.loc[t.index, "tempo_percurso_s"].to_numpy()
t["tipico"] = t["tempo"] - t["alvo"]
t["trivial"] = 0.0
modelos = [c for c in ("xgboost", "lstm", "gnn", "conjunto", "xgb_sem_api", "xgb_geo", "trivial") if c in t.columns]
print(f"trechos de teste: {len(t):,} | viagens: {t['viagem_id'].nunique():,} | modelos: {modelos}")

print("\n== MAE (s) e acerto20")
for m in modelos:
    e = np.abs(t["alvo"] - t[m])
    print(f"   {m:12s} MAE {e.mean():6.2f} | acerto20 {100 * (e <= 0.2 * t['tempo']).mean():5.1f}%")

print("\n== Intervalo de 80% da LSTM (quantis 10 e 90)")
dentro = (t["alvo"] >= t["q10"]) & (t["alvo"] <= t["q90"])
print(f"   cobertura geral: {100 * dentro.mean():.1f}% (nominal 80%) | abaixo do q10: {100 * (t['alvo'] < t['q10']).mean():.1f}%"
      f" | acima do q90: {100 * (t['alvo'] > t['q90']).mean():.1f}% | largura mediana {np.median(t['q90'] - t['q10']):.1f} s")
print(t.assign(d=dentro).groupby("dia")["d"].mean().mul(100).round(1).to_string())

print("\n== Classificacao: atrasado = alvo >= 20% do tempo tipico (referencia do treino)")
y = (t["alvo"] >= 0.2 * t["tipico"]).to_numpy()
print(f"   taxa de atrasados no teste: {100 * y.mean():.1f}% | F1 de marcar tudo como atrasado: {2 * y.mean() / (1 + y.mean()):.3f}")


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


for m in [c for c in modelos if c != "trivial"]:
    score = (t[m] / t["tipico"]).to_numpy()
    p = score >= 0.2
    tp = (p & y).sum()
    f1 = 2 * tp / (p.sum() + y.sum())
    print(f"   {m:12s} F1 {f1:.3f} | AUC {auc(y, score):.3f} | precisao {tp / max(p.sum(), 1):.3f} | revocacao {tp / y.sum():.3f}")

print("\n== Bootstrap por viagem (1000 reamostragens): MAE do modelo menos MAE do XGBoost (s)")
g = t.groupby("viagem_id")
n = g.size().to_numpy()
soma = {m: g.apply(lambda d, m=m: np.abs(d["alvo"] - d[m]).sum(), include_groups=False).to_numpy() for m in modelos}
rng = np.random.default_rng(7)
k = len(n)
dif = {m: [] for m in modelos if m != "xgboost"}
for _ in range(1000):
    idx = rng.integers(0, k, k)
    w = np.bincount(idx, minlength=k)
    base = (soma["xgboost"] * w).sum() / (n * w).sum()
    for m in dif:
        dif[m].append((soma[m] * w).sum() / (n * w).sum() - base)
for m, v in dif.items():
    v = np.array(v)
    print(f"   {m:12s} diferenca {v.mean():+.3f} s | IC95% [{np.percentile(v, 2.5):+.3f}; {np.percentile(v, 97.5):+.3f}]"
          f" | {'significativa' if (np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0) else 'NAO significativa'}")
