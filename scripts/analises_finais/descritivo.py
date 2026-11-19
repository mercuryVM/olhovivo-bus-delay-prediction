"""Parte descritiva com os 32 dias (rodada final). So leitura.

Atraso aqui usa a referencia da tabela percursos (mediana do periodo todo por trecho,
sentido e faixa): serve para descrever, nao para prever.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

D = Path("/dados/derivado")
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}
pd.set_option("display.width", 220)

pc = pd.read_parquet(D / "percursos.parquet")
print("colunas:", ", ".join(pc.columns))
pc = pc.dropna(subset=["tempo_percurso_s", "atraso_s"]).reset_index(drop=True)
t = pd.to_datetime(pc["t_origem"], utc=True).dt.tz_convert(FUSO)
pc["dia"] = t.dt.strftime("%Y-%m-%d")
pc["hora"] = t.dt.hour
dow = t.dt.dayofweek
pc["tipo"] = np.where(pc["dia"].isin(FERIADOS) | (dow == 6), "domingo/feriado",
                      np.where(dow == 5, "sabado", "util"))
pico = ((pc["hora"] >= 6) & (pc["hora"] < 9)) | ((pc["hora"] >= 16) & (pc["hora"] < 20))
pc["pico_util"] = pico & (pc["tipo"] == "util")
dist = next((c for c in ("extensao_m", "dist_m", "distancia_m") if c in pc.columns), None)
if dist:
    pc["vel_kmh"] = pc[dist] / pc["tempo_percurso_s"] * 3.6
    pc.loc[(pc["vel_kmh"] <= 0) | (pc["vel_kmh"] > 90), "vel_kmh"] = np.nan


def resumo(g):
    r = {"passagens": len(g), "atrasados_%": 100 * g["atrasado"].mean(),
         "tempo_mediano_s": g["tempo_percurso_s"].median(),
         "atraso_mediano_s": g["atraso_s"].median(),
         "p90_atraso_s": g["atraso_s"].quantile(0.9),
         "atraso_medio_dos_atrasados_s": g.loc[g["atrasado"] == 1, "atraso_s"].mean()}
    if dist:
        r["vel_mediana_kmh"] = g["vel_kmh"].median()
    return pd.Series(r)


print("\n== geral")
print(resumo(pc).round(2).to_string())
print(f"   um em cada {1 / pc['atrasado'].mean():.1f} passagens atrasa")

print("\n== por tipo de dia")
print(pc.groupby("tipo").apply(resumo, include_groups=False).round(1).to_string())

print("\n== dia util: pico (6-9h e 16-20h) x fora")
u = pc[pc["tipo"] == "util"]
print(u.groupby(u["pico_util"].map({True: "pico", False: "fora"})).apply(resumo, include_groups=False).round(1).to_string())
print(f"   pico de dia util = {100 * pc['pico_util'].mean():.1f}% das passagens e "
      f"{100 * pc.loc[pc['atrasado'] == 1, 'pico_util'].mean():.1f}% dos trechos atrasados")

print("\n== dia util por hora")
print(u.groupby("hora").apply(resumo, include_groups=False).round(1).to_string())

print("\n== domingo/feriado por hora (atrasados %, vel)")
dd = pc[pc["tipo"] == "domingo/feriado"].groupby("hora").apply(resumo, include_groups=False)
print(dd[[c for c in ("passagens", "atrasados_%", "vel_mediana_kmh") if c in dd.columns]].round(1).to_string())

print("\n== por linha (letreiro)")
print(pc.groupby("letreiro").apply(resumo, include_groups=False).sort_values("atrasados_%").round(1).to_string())

# por viagem: atraso acumulado nas viagens com ao menos 80% dos trechos da linha-sentido
print("\n== por viagem (viagens com >= 80% dos trechos)")
n_tr = pc.groupby(["cl", "sentido"])["trecho"].nunique().rename("n_trechos_linha")
v = pc.groupby(["viagem_id", "cl", "sentido"]).agg(
    trechos=("trecho", "nunique"), atraso_total_s=("atraso_s", "sum"),
    tempo_total_s=("tempo_percurso_s", "sum"), tipo=("tipo", "first"),
    hora=("hora", "first"), pico=("pico_util", "mean")).reset_index().join(n_tr, on=["cl", "sentido"])
v = v[v["trechos"] >= 0.8 * v["n_trechos_linha"]]
v["min_5"] = v["atraso_total_s"] >= 300
print(f"   viagens: {len(v)} | tempo mediano {v['tempo_total_s'].median() / 60:.1f} min | atraso acumulado "
      f"mediano {v['atraso_total_s'].median() / 60:.1f} min | 5 min ou mais: {100 * v['min_5'].mean():.1f}% "
      f"(uma em cada {1 / v['min_5'].mean():.1f}) | 10 min ou mais: {100 * (v['atraso_total_s'] >= 600).mean():.1f}%")
v["faixa"] = pd.cut(v["hora"], [-1, 5, 8, 15, 19, 23], labels=["madrugada", "pico manha", "entrepico", "pico tarde", "noite"])
print(v.groupby(["tipo", "faixa"], observed=True).agg(
    viagens=("min_5", "size"), pct_5min=("min_5", "mean"),
    atraso_mediano_min=("atraso_total_s", lambda x: x.median() / 60),
    tempo_mediano_min=("tempo_total_s", lambda x: x.median() / 60)).assign(
    pct_5min=lambda x: 100 * x["pct_5min"]).round(1).to_string())

# erro da previsao da API contra a chegada real
print("\n== previsao da API x chegada real")
arq = D / "previsao_realizado.parquet"
if not arq.exists():
    cand = sorted(D.glob("previsao_realizado*"))
    arq = cand[0] if cand else None
if arq is None:
    print("   previsao_realizado nao encontrado")
else:
    pr = ds.dataset(arq).to_table(columns=["horizonte_s", "erro_s"]).to_pandas()
    pr["faixa"] = pd.cut(pr["horizonte_s"], [0, 300, 900, 1800, 3600], right=False,
                         labels=["< 5 min", "5-15 min", "15-30 min", "30-60 min"])
    tab = pr.groupby("faixa", observed=True)["erro_s"].agg(
        previsoes="size", erro_abs_medio_s=lambda x: x.abs().mean(), erro_mediano_s="median",
        chega_1min_depois_pct=lambda x: 100 * (x >= 60).mean(),
        chega_1min_antes_pct=lambda x: 100 * (x <= -60).mean())
    print(f"   {len(pr)} previsoes casadas")
    print(tab.round(1).to_string())
    print(f"   geral: erro abs medio {pr['erro_s'].abs().mean():.1f} s | chega 1 min ou mais depois "
          f"{100 * (pr['erro_s'] >= 60).mean():.1f}% | antes {100 * (pr['erro_s'] <= -60).mean():.1f}%")
