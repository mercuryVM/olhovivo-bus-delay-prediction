"""Sensibilidade ao corte de atraso. So leitura.

Normal = percurso_tipico_s da tabela percursos (mesmo da analise descritiva). Atrasado no corte c: tempo >= (1+c) x normal.
A. % atrasados por tipo de dia para c = 10, 15, 20, 25, 30, 50% (todas as passagens e sem trechos de ponta).
B. % de passagens 10/20/30% ou mais ABAIXO do normal por tipo de dia (cauda x ruido simetrico).
C. Ranking das zonas OD sem trechos de ponta (>= 5 mil passagens, como zonas_ponta.py) para cada corte:
   Spearman contra o ranking de 20% e quantas das 5 do topo e 5 do fundo se mantem.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
RAIZ = Path("/dados")
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}
CORTES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
ORDEM = ["util", "sabado", "domingo/feriado"]

pc = pd.read_parquet(RAIZ / "derivado" / "percursos.parquet", columns=[
    "cl", "sentido", "viagem_id", "cp_destino", "ordem_origem", "t_origem", "tempo_percurso_s", "percurso_tipico_s",
    "atraso_s", "atraso_rel", "atrasado", "lat", "lon"]).dropna(subset=["tempo_percurso_s", "atraso_s"]).reset_index(drop=True)
t = pd.to_datetime(pc["t_origem"], utc=True).dt.tz_convert(FUSO)
dia = t.dt.strftime("%Y-%m-%d")
dow = t.dt.dayofweek
pc["tipo"] = np.where(dia.isin(FERIADOS) | (dow == 6), "domingo/feriado", np.where(dow == 5, "sabado", "util"))
pc["r"] = pc["tempo_percurso_s"] / pc["percurso_tipico_s"]
ok = np.isfinite(pc["r"]) & (pc["percurso_tipico_s"] > 0)
print(f"passagens {len(pc):,} | com normal valido {int(ok.sum()):,} | dias {dia.nunique()}")
pc = pc[ok].reset_index(drop=True)

# conferencia: a coluna atrasado bate com r >= 1,2?
a20 = pc["r"] >= 1.2
print(f"conferencia corte 20%: coluna atrasado {100 * pc['atrasado'].mean():.2f}% | r>=1,2 {100 * a20.mean():.2f}% | "
      f"concordancia {100 * (a20 == pc['atrasado'].astype(bool)).mean():.3f}% | atraso_rel vs r-1 dif max "
      f"{np.nanmax(np.abs(pc['atraso_rel'] - (pc['r'] - 1))):.2e}")

# ---- pontas (mesma regra de zonas_ponta.py: primeiro e ultimo trecho com cobertura >= 0,7)
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
        lim.append((cl, se, boa[0], boa[-1]))
lim = pd.DataFrame(lim, columns=["cl", "sentido", "a0", "b0"])
mm = pc.groupby(["cl", "sentido"])["ordem_origem"].agg(o_min="min", o_max="max").reset_index().merge(lim, on=["cl", "sentido"], how="left")
mm["a0"] = mm["a0"].fillna(mm["o_min"])
mm["b0"] = mm["b0"].fillna(mm["o_max"])
x = pc[["cl", "sentido", "ordem_origem"]].merge(mm, on=["cl", "sentido"], how="left")
pc["ponta"] = ((x["ordem_origem"] <= x["a0"]) | (x["ordem_origem"] >= x["b0"])).to_numpy()
del base, x, nv, nl
print(f"passagens em trechos de ponta: {100 * pc['ponta'].mean():.1f}%")


def tab_tipo(d):
    lin = {}
    for tp in ORDEM + ["geral"]:
        g = d if tp == "geral" else d[d["tipo"] == tp]
        row = {"passagens": len(g)}
        for c in CORTES:
            row[f"ac{int(round(100 * c))}"] = 100 * (g["r"] >= 1 + c).mean()
        for c in (0.10, 0.20, 0.30):
            row[f"ab{int(round(100 * c))}"] = 100 * (g["r"] <= 1 - c).mean()
        row["mediana_r"] = g["r"].median()
        row["p10_r"] = g["r"].quantile(0.1)
        row["p90_r"] = g["r"].quantile(0.9)
        lin[tp] = row
    return pd.DataFrame(lin).T


for nome, d in (("todas as passagens", pc), ("sem trechos de ponta", pc[~pc["ponta"]])):
    tb = tab_tipo(d)
    print(f"\n== A/B. {nome}: % acima do normal (acNN = tempo >= 1+NN% do normal) e abaixo (abNN = tempo <= 1-NN%)")
    print(tb.round(2).to_string())
    print("   ordem util > sabado > domingo/feriado por corte:")
    for c in CORTES:
        k = f"ac{int(round(100 * c))}"
        u, s, dm = tb.loc["util", k], tb.loc["sabado", k], tb.loc["domingo/feriado", k]
        print(f"     {k}: util {u:.1f} | sab {s:.1f} | dom {dm:.1f} | mantem {u > s > dm} | util/dom {u / dm:.2f}x | util-dom {u - dm:.1f} p.p.")
    print("   assimetria: acima de +20% x abaixo de -20% (razao):")
    for tp in ORDEM + ["geral"]:
        print(f"     {tp}: {tb.loc[tp, 'ac20']:.1f}% x {tb.loc[tp, 'ab20']:.1f}% -> {tb.loc[tp, 'ac20'] / tb.loc[tp, 'ab20']:.1f}x | "
              f"+10% {tb.loc[tp, 'ac10']:.1f} x -10% {tb.loc[tp, 'ab10']:.1f} | +30% {tb.loc[tp, 'ac30']:.1f} x -30% {tb.loc[tp, 'ab30']:.1f}")
    print("###TIPO_" + ("TODAS" if nome.startswith("todas") else "SEMPONTA"))
    print(tb.reset_index().rename(columns={"index": "tipo"}).round(3).to_csv(index=False))
    print("###FIM")

# histograma do r (geral e por tipo), em faixas de 5%
bins = np.r_[-np.inf, np.arange(0.5, 2.01, 0.05), np.inf]
h = pc.groupby(["tipo", pd.cut(pc["r"], bins, right=False)], observed=False).size().rename("passagens").reset_index()
h["r"] = h["r"].astype(str)
print("###HIST_R")
print(h.to_csv(index=False))
print("###FIM")

# ---- zonas
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

d = pc[~pc["ponta"]].dropna(subset=["zona"])
agg = {"passagens": ("r", "size")}
for c in CORTES:
    agg[f"ac{int(round(100 * c))}"] = ("r", lambda s, c=c: 100 * (s >= 1 + c).mean())
agg["ab20"] = ("r", lambda s: 100 * (s <= 0.8).mean())
z = d.groupby("zona").agg(**agg)
z = z[z["passagens"] >= 5000]
print(f"\n== C. zonas sem trechos de ponta com >= 5 mil passagens: {len(z)}")


def spearman(a, b):
    ra, rb = a.rank(), b.rank()
    return float(np.corrcoef(ra, rb)[0, 1])


ref = z["ac20"]
top_ref = set(ref.sort_values(ascending=False).head(5).index)
fun_ref = set(ref.sort_values().head(5).index)
top10_ref = set(ref.sort_values(ascending=False).head(10).index)
print(f"   topo 5 (20%): {', '.join(ref.sort_values(ascending=False).head(5).index)}")
print(f"   fundo 5 (20%): {', '.join(ref.sort_values().head(5).index)}")
res = []
for c in CORTES:
    k = f"ac{int(round(100 * c))}"
    s = z[k]
    tp = list(s.sort_values(ascending=False).head(5).index)
    fd = list(s.sort_values().head(5).index)
    res.append(dict(corte=k, spearman=round(spearman(s, ref), 3), topo5_mantidas=len(top_ref & set(tp)),
                    fundo5_mantidas=len(fun_ref & set(fd)),
                    topo10_mantidas=len(top10_ref & set(s.sort_values(ascending=False).head(10).index)),
                    faixa=f"{s.min():.1f}-{s.max():.1f}%", topo5=", ".join(tp), fundo5=", ".join(fd)))
res = pd.DataFrame(res)
print(res.to_string(index=False))
print(f"   Spearman ac20 x ab20 (zonas com mais atraso tem mais passagens rapidas?): {spearman(z['ac20'], z['ab20']):.3f}")
# posicoes das zonas do topo/fundo de 20% em cada corte
rk = z[[f"ac{int(round(100 * c))}" for c in CORTES]].rank(ascending=False).astype(int)
mostrar = list(ref.sort_values(ascending=False).head(5).index) + list(ref.sort_values().head(5).index[::-1])
print("   posicao (1 = mais atrasada) das zonas do topo e do fundo de 20% em cada corte:")
print(rk.loc[mostrar].to_string())
print("###ZONAS_CORTES")
print(z.join(rk.add_prefix("pos_")).reset_index().round(2).to_csv(index=False))
print("###FIM")
print("fim")
