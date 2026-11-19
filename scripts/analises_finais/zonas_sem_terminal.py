"""Tabela de zonas OD sem os trechos de ponta (saida e chegada no terminal). So leitura.

Mesmo calculo da tabela de zonas do relatorio (scripts/analise_final.py: zona do ponto medio do trecho que chega na
parada; atrasados, p90 do atraso relativo e velocidade mediana), em tres versoes: todas as passagens; sem o primeiro e
o ultimo trecho bem registrados de cada linha e sentido (e o que fica fora deles); sem os dois primeiros e os dois
ultimos.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
RAIZ = Path("/dados")
pc = pd.read_parquet(RAIZ / "derivado" / "percursos.parquet", columns=[
    "cl", "sentido", "viagem_id", "cp_destino", "ordem_origem", "tempo_percurso_s", "atraso_s", "atraso_rel", "atrasado",
    "velocidade_kmh", "lat", "lon"]).dropna(subset=["tempo_percurso_s", "atraso_s"]).reset_index(drop=True)

nv = pc.groupby("viagem_id")["ordem_origem"].transform("nunique")
nl = pc.groupby(["cl", "sentido"])["ordem_origem"].transform("nunique")
base = pc[nv >= 0.5 * nl]
tot = base.groupby(["cl", "sentido"])["viagem_id"].nunique().rename("tot").reset_index()
cov = base.groupby(["cl", "sentido", "ordem_origem"])["viagem_id"].nunique().rename("n").reset_index().merge(tot, on=["cl", "sentido"])
cov["cob"] = cov["n"] / cov["tot"]
lim = []
for (cl, se), g in cov.groupby(["cl", "sentido"]):
    boa = g[g["cob"] >= 0.7].sort_values("ordem_origem")["ordem_origem"].to_numpy()
    if len(boa) >= 6:
        lim.append((cl, se, boa[0], boa[-1], boa[1], boa[-2]))
lim = pd.DataFrame(lim, columns=["cl", "sentido", "a0", "b0", "a1", "b1"])
mm = pc.groupby(["cl", "sentido"])["ordem_origem"].agg(o_min="min", o_max="max").reset_index().merge(lim, on=["cl", "sentido"], how="left")
for c, d in (("a0", "o_min"), ("b0", "o_max"), ("a1", "o_min"), ("b1", "o_max")):
    mm[c] = mm[c].fillna(mm[d])
x = pc[["cl", "sentido", "ordem_origem"]].merge(mm, on=["cl", "sentido"], how="left")
pc["ponta1"] = ((x["ordem_origem"] <= x["a0"]) | (x["ordem_origem"] >= x["b0"])).to_numpy()
pc["ponta2"] = ((x["ordem_origem"] <= x["a1"]) | (x["ordem_origem"] >= x["b1"])).to_numpy()
print(f"passagens {len(pc):,} | ponta1 {100 * pc['ponta1'].mean():.1f}% | ponta2 {100 * pc['ponta2'].mean():.1f}%")

zonas = json.loads((RAIZ / "externo" / "geosampa" / "zona_origem_destino.geojson").read_text(encoding="utf-8"))["features"]


def aneis(g):
    if g["type"] == "Polygon":
        return [np.asarray(r)[:, :2] for r in g["coordinates"]]
    if g["type"] == "MultiPolygon":
        return [np.asarray(r)[:, :2] for p in g["coordinates"] for r in p]
    return []


def dentro(lon, lat, rings):
    res = np.zeros(len(lon), dtype=bool)
    for r in rings:
        x0, y0, x1, y1 = r[:-1, 0], r[:-1, 1], r[1:, 0], r[1:, 1]
        cruza = ((y0[None, :] > lat[:, None]) != (y1[None, :] > lat[:, None])) & (
            lon[:, None] < (x1 - x0)[None, :] * (lat[:, None] - y0[None, :]) / np.where(y1 == y0, 1e-12, y1 - y0)[None, :] + x0[None, :])
        res ^= (cruza.sum(axis=1) % 2 == 1)
    return res


pontos = pc.groupby("cp_destino").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
pontos["zona"] = None
for f in zonas:
    rings = aneis(f["geometry"])
    if not rings:
        continue
    todos = np.concatenate(rings)
    caixa = (pontos["lon"].between(todos[:, 0].min(), todos[:, 0].max()) & pontos["lat"].between(todos[:, 1].min(), todos[:, 1].max())
             & pontos["zona"].isna())
    if caixa.any():
        idx = pontos.index[caixa]
        m = dentro(pontos.loc[idx, "lon"].to_numpy(), pontos.loc[idx, "lat"].to_numpy(), rings)
        pontos.loc[idx[m], "zona"] = f["properties"].get("nm_zona")
pc["zona"] = pc["cp_destino"].map(dict(zip(pontos["cp_destino"], pontos["zona"])))


def tabela(d, nome):
    z = d.dropna(subset=["zona"]).groupby("zona").agg(
        passagens=("atrasado", "size"), atrasados=("atrasado", lambda s: 100 * s.mean()),
        p90_rel=("atraso_rel", lambda s: 100 * s.quantile(0.9)), vel_kmh=("velocidade_kmh", "median"))
    z = z[z["passagens"] >= 5000]
    print(f"\n== {nome}: {len(d):,} passagens | atrasados {100 * d['atrasado'].mean():.2f}% | zonas >= 5 mil: {len(z)} | "
          f"de {z['atrasados'].min():.1f}% a {z['atrasados'].max():.1f}%")
    print("mais atrasadas:")
    print(z.sort_values("atrasados", ascending=False).head(10).round(1).to_string())
    print("menos atrasadas:")
    print(z.sort_values("atrasados").head(5).round(1).to_string())
    return z


a = tabela(pc, "todas as passagens")
b = tabela(pc[~pc["ponta1"]], "sem a saida e a chegada no terminal")
c = tabela(pc[~pc["ponta2"]], "sem os dois primeiros e os dois ultimos trechos")
f = a[["atrasados"]].rename(columns={"atrasados": "todas"}).join(b[["atrasados", "p90_rel", "vel_kmh", "passagens"]].rename(
    columns={"atrasados": "sem_ponta1", "passagens": "passagens_sem1"}), how="outer").join(c[["atrasados"]].rename(columns={"atrasados": "sem_ponta2"}), how="outer")
print("###ZONAS3")
print(f.reset_index().round(2).to_csv(index=False))
print("###FIM")
