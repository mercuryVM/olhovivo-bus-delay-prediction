"""Erro do modelo por linha na semana de teste (02-08/11), a partir das previsoes JA salvas do teste final.
Nao treina nada. So leitura; saida por print.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/dados/derivado/modelos/cache_diaria")
DIAS = [f"2026-11-0{d}" for d in range(2, 9)]
pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 200)

pc = pd.read_parquet("/dados/derivado/percursos.parquet",
                     columns=["viagem_id", "tempo_percurso_s", "cl", "letreiro", "sentido"])
pc = pc.dropna(subset=["tempo_percurso_s"]).reset_index(drop=True)
print("arquivos no cache:", len(list(CACHE.glob("*.pkl"))))

partes = []
for dia in DIAS:
    r = pickle.loads((CACHE / f"todos_frota_{dia}_redes.pkl").read_bytes())
    if dia == DIAS[0]:
        print("chaves do pkl:", list(r.keys()), "| modelos:", list(r["previsto_por_modelo"].keys()))
    idx = pd.Index(r["indice"])
    df = pd.DataFrame({"dia": dia, "alvo": np.asarray(r["alvo"])}, index=idx)
    for m, v in r["previsto_por_modelo"].items():
        df[m] = np.asarray(v)
    partes.append(df)
t = pd.concat(partes)
for c in ("viagem_id", "tempo_percurso_s", "cl", "letreiro", "sentido"):
    t[c] = pc.loc[t.index, c].to_numpy()
t["trivial"] = 0.0
modelos = [c for c in ("xgboost", "lstm", "gnn", "conjunto", "trivial") if c in t.columns]
for m in modelos:
    t["e_" + m] = np.abs(t["alvo"] - t[m])
    t["ok_" + m] = t["e_" + m] <= 0.2 * t["tempo_percurso_s"]
print(f"passagens (trechos) de teste: {len(t):,} | viagens: {t['viagem_id'].nunique():,} | modelos: {modelos}")
print("checagem geral (todos juntos): " + " | ".join(f"{m} {t['e_' + m].mean():.2f}" for m in modelos))
print("checagem media dos 7 dias: " + " | ".join(f"{m} {t.groupby('dia')['e_' + m].mean().mean():.2f}" for m in modelos))


def tabela(chave, nome):
    g = t.groupby(chave)
    o = pd.DataFrame({"passagens": g.size(), "viagens": g["viagem_id"].nunique()})
    for m in modelos:
        o["mae_" + m] = g["e_" + m].mean()
    o["mae7d_xgb"] = t.groupby(chave + ["dia"])["e_xgboost"].mean().groupby(level=list(range(len(chave)))).mean()
    o["mae7d_triv"] = t.groupby(chave + ["dia"])["e_trivial"].mean().groupby(level=list(range(len(chave)))).mean()
    o["ganho_s"] = o["mae_trivial"] - o["mae_xgboost"]
    o["ganho_%"] = 100 * o["ganho_s"] / o["mae_trivial"]
    o["acerto20_xgb_%"] = 100 * g["ok_xgboost"].mean()
    o["acerto20_triv_%"] = 100 * g["ok_trivial"].mean()
    o["dias_xgb_vence"] = (t.groupby(chave + ["dia"])[["e_xgboost", "e_trivial"]].mean()
                           .pipe(lambda d: d["e_xgboost"] < d["e_trivial"]).groupby(level=list(range(len(chave)))).sum())
    o = o.sort_values("ganho_%")
    print(f"\n== Por {nome} (MAE em s; mae_* = todas as passagens juntas; mae7d = media dos 7 dias)")
    print(o.round(2).to_string())
    v = o["ganho_s"] > 0
    print(f"   XGBoost ganha do trivial em {int(v.sum())} de {len(o)} | menor margem {o['ganho_s'].min():.2f} s "
          f"({o['ganho_%'].min():.1f}%) em {o['ganho_s'].idxmin()} | maior {o['ganho_s'].max():.2f} s ({o['ganho_%'].max():.1f}%)")
    for m in ("lstm", "gnn", "conjunto"):
        if m in modelos:
            print(f"   {m} melhor que XGBoost em {int((o['mae_' + m] < o['mae_xgboost']).sum())} de {len(o)}")
    return o


tabela(["letreiro"], "letreiro (dois sentidos juntos)")
tabela(["letreiro", "sentido", "cl"], "letreiro x sentido (codigo cl)")

print("\n== Bootstrap por viagem (1000x) da diferenca trivial - XGBoost por letreiro")
rng = np.random.default_rng(7)
for lt, d in t.groupby("letreiro"):
    g = d.groupby("viagem_id")
    n = g.size().to_numpy()
    sx = g["e_xgboost"].sum().to_numpy()
    st = g["e_trivial"].sum().to_numpy()
    k = len(n)
    dif = []
    for _ in range(1000):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        dif.append(((st - sx) * w).sum() / (n * w).sum())
    dif = np.array(dif)
    print(f"   {lt}: {dif.mean():+.2f} s IC95% [{np.percentile(dif, 2.5):+.2f}; {np.percentile(dif, 97.5):+.2f}] ({k} viagens)")
