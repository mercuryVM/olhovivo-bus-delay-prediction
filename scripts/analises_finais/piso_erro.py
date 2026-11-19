"""Conferencia do erro irredutivel no mesmo conjunto de passagens do teste final. So leitura.

O slide compara o erro do XGBoost (todas as passagens do teste, 02 a 08/11) com a diferenca entre pares de
onibus proximos (todos os 32 dias). Aqui tudo e medido no mesmo conjunto: passagens do teste que tem outro
onibus no mesmo trecho ate 2 min antes. Tambem sem os trechos de ponta (saida e chegada no terminal).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
CACHE = Path("/dados/derivado/modelos/cache_diaria")
DIAS = [f"2026-11-0{d}" for d in range(2, 9)]

pc = pd.read_parquet("/dados/derivado/percursos.parquet", columns=[
    "cl", "sentido", "viagem_id", "prefixo", "cp_origem", "cp_destino", "ordem_origem", "t_origem", "tempo_percurso_s"])
pc = pc.dropna(subset=["tempo_percurso_s"]).reset_index(drop=True)      # mesmo indice do cache (teste_final_extra.py)
pc["t"] = pd.to_datetime(pc["t_origem"], utc=True)

# trechos de ponta: o primeiro e o ultimo trecho bem registrados (cobertura >= 0,7) de cada linha e sentido, e o que
# fica fora deles (mesma regra do fluxos_episodios.py)
nv = pc.groupby("viagem_id")["ordem_origem"].transform("nunique")
nl = pc.groupby(["cl", "sentido"])["ordem_origem"].transform("nunique")
base = pc[nv >= 0.5 * nl]
tot = base.groupby(["cl", "sentido"])["viagem_id"].nunique().rename("tot").reset_index()
cov = base.groupby(["cl", "sentido", "ordem_origem"])["viagem_id"].nunique().rename("n").reset_index().merge(tot, on=["cl", "sentido"])
cov["cob"] = cov["n"] / cov["tot"]
lim = []
for (cl, se), g in cov.groupby(["cl", "sentido"]):
    boa = g[g["cob"] >= 0.7].sort_values("ordem_origem")
    if len(boa) >= 6:
        lim.append((cl, se, boa["ordem_origem"].iloc[0], boa["ordem_origem"].iloc[-1]))
lim = pd.DataFrame(lim, columns=["cl", "sentido", "o_a0", "o_b0"])
mm = pc.groupby(["cl", "sentido"])["ordem_origem"].agg(o_min="min", o_max="max").reset_index().merge(lim, on=["cl", "sentido"], how="left")
mm["o_a0"] = mm["o_a0"].fillna(mm["o_min"])
mm["o_b0"] = mm["o_b0"].fillna(mm["o_max"])
x = pc[["cl", "sentido", "ordem_origem"]].merge(mm[["cl", "sentido", "o_a0", "o_b0"]], on=["cl", "sentido"], how="left")
pc["ponta"] = ((x["ordem_origem"] <= x["o_a0"]) | (x["ordem_origem"] >= x["o_b0"])).to_numpy()
print(f"passagens {len(pc):,} | de ponta {100 * pc['ponta'].mean():.1f}%")

# pares: o onibus anterior no mesmo trecho fisico, outro veiculo, ate 2 min antes
s = pc.sort_values(["cp_origem", "cp_destino", "t"])
idx = s.index.to_numpy()
mesmo = (s["cp_origem"].to_numpy()[1:] == s["cp_origem"].to_numpy()[:-1]) & (s["cp_destino"].to_numpy()[1:] == s["cp_destino"].to_numpy()[:-1])
outro = s["prefixo"].astype(str).to_numpy()[1:] != s["prefixo"].astype(str).to_numpy()[:-1]
dt = (s["t"].iloc[1:].to_numpy() - s["t"].iloc[:-1].to_numpy()) / np.timedelta64(1, "s")
ok = mesmo & outro & (dt < 120)
seg = pd.DataFrame({"i": idx[1:][ok], "tempo_frente": s["tempo_percurso_s"].to_numpy()[:-1][ok], "dt": dt[ok]})
seg = seg.drop_duplicates("i").set_index("i")
pc["tempo_frente"] = seg["tempo_frente"].reindex(pc.index)
pc["dt_frente"] = seg["dt"].reindex(pc.index)
dif_todos = (pc["tempo_percurso_s"] - pc["tempo_frente"]).abs().dropna()
print(f"pares ate 2 min, 32 dias: {len(dif_todos):,} | diferenca media {dif_todos.mean():.2f} s | piso {dif_todos.mean() / np.sqrt(2):.2f} s")
sp = (pc["tempo_percurso_s"] - pc["tempo_frente"]).abs()[~pc["ponta"]].dropna()
print(f"   sem trechos de ponta: {len(sp):,} pares | diferenca media {sp.mean():.2f} s | piso {sp.mean() / np.sqrt(2):.2f} s")
pt = (pc["tempo_percurso_s"] - pc["tempo_frente"]).abs()[pc["ponta"]].dropna()
print(f"   so trechos de ponta: {len(pt):,} pares | diferenca media {pt.mean():.2f} s")

# previsoes guardadas do teste final
partes = []
for dia in DIAS:
    r = pickle.loads((CACHE / f"todos_frota_{dia}_redes.pkl").read_bytes())
    df = pd.DataFrame({"dia": dia, "alvo": r["alvo"], "xgb": r["previsto_por_modelo"]["xgboost"]}, index=pd.Index(r["indice"]))
    partes.append(df)
t = pd.concat(partes)
t = t.join(pc[["tempo_percurso_s", "tempo_frente", "dt_frente", "ponta"]])
t["tipico"] = t["tempo_percurso_s"] - t["alvo"]
t["e_xgb"] = (t["alvo"] - t["xgb"]).abs()
t["e_triv"] = t["alvo"].abs()
t["e_copia"] = (t["tempo_percurso_s"] - t["tempo_frente"]).abs()
print(f"\npassagens do teste: {len(t):,}")


def linha(nome, d):
    por_dia = d.groupby("dia")[["e_triv", "e_xgb"]].mean().mean()
    c = d["e_copia"].dropna()
    print(f"   {nome:48s} n {len(d):8,} | trivial {d['e_triv'].mean():6.2f} (media dos dias {por_dia['e_triv']:.2f}) | XGBoost {d['e_xgb'].mean():6.2f}"
          f" (media dos dias {por_dia['e_xgb']:.2f})" + (f" | copiar o da frente {c.mean():6.2f} | piso {c.mean() / np.sqrt(2):6.2f}" if len(c) else ""))


linha("todas", t)
linha("sem trechos de ponta", t[~t["ponta"]])
linha("so trechos de ponta", t[t["ponta"]])
par = t.dropna(subset=["tempo_frente"])
linha("com onibus ate 2 min antes (mesmo conjunto)", par)
linha("com onibus ate 2 min antes, sem ponta", par[~par["ponta"]])
linha("com onibus ate 1 min antes, sem ponta", par[(~par["ponta"]) & (par["dt_frente"] <= 60)])
print(f"   fracao das passagens do teste com onibus ate 2 min antes: {100 * len(par) / len(t):.1f}%")
