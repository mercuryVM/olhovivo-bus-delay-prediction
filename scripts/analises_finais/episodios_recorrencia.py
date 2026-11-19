"""Fluxos de origem e destino corrigidos e exemplos reais de episodios de atraso. So leitura.

A. Viagem inteira num trecho comum: em cada linha, sentido e variante, duas paradas fixas (a primeira e a
   ultima que quase todas as viagens registram). A duracao e o tempo entre elas. Viagens que perderam a
   parada final nao viram um fluxo a parte nem parecem mais rapidas por somar menos trechos.
B. Episodios ST-DBSCAN (400 m, 30 min, 10), dia a dia, como no od_agrupamentos.py, com exemplos reais
   (pontos, contexto, linha do tempo e recorrencia) para os slides.
C. Sequencia de trechos de cada linha com a coordenada das paradas (para desenhar as rotas).
Saidas CSV entre marcadores ### para desenhar localmente.
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/app")
from olhovivo import geo, regioes  # noqa: E402
from olhovivo.config import carregar  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)
RAIZ = Path("/dados")
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}
FAIXAS = ([-1, 5, 8, 15, 19, 23], ["madrugada", "pico manha", "entrepico", "pico tarde", "noite"])
cfg = carregar()
proj = geo.obter_projecao(int(cfg.get("geo.epsg_metrico", 31983)), float(cfg.get("geo.ancora_lat", -23.5505)),
                          float(cfg.get("geo.ancora_lon", -46.6333)))


def bloco(nome, df, **kw):
    print(f"###{nome}")
    print(df.to_csv(index=False, **kw))
    print("###FIM")


def hhmm(t):
    return pd.to_datetime(t, utc=True).dt.tz_convert(FUSO).dt.strftime("%H:%M:%S")


pc = pd.read_parquet(RAIZ / "derivado" / "percursos.parquet", columns=[
    "cl", "letreiro", "sentido", "viagem_id", "cp_origem", "cp_destino", "ordem_origem", "t_origem", "t_chegada",
    "tempo_percurso_s", "percurso_tipico_s", "atraso_s", "atrasado", "lat", "lon"]).dropna(subset=["tempo_percurso_s"])
pc = pc.reset_index(drop=True)
pc["cp_origem"] = pc["cp_origem"].astype("int64")
pc["cp_destino"] = pc["cp_destino"].astype("int64")
pc["t_origem"] = pd.to_datetime(pc["t_origem"], utc=True)
pc["t_chegada"] = pd.to_datetime(pc["t_chegada"], utc=True)
loc = pc["t_origem"].dt.tz_convert(FUSO)
pc["dia"] = loc.dt.strftime("%Y-%m-%d")
pc["hora"] = loc.dt.hour
dow = loc.dt.dayofweek
pc["tipo"] = np.where(pc["dia"].isin(FERIADOS) | (dow == 6), "domingo/feriado", np.where(dow == 5, "sabado", "util"))
print(f"passagens {len(pc):,} | viagens {pc['viagem_id'].nunique():,} | datas {pc['dia'].nunique()}")

# ------------------------------------------------------------------ zonas e paradas
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


def zona_de(df):
    z = pd.Series([None] * len(df), index=df.index, dtype=object)
    for f in zonas:
        rings = aneis(f["geometry"])
        if not rings:
            continue
        tudo = np.concatenate(rings)
        caixa = (df["lon"].between(tudo[:, 0].min(), tudo[:, 0].max()) & df["lat"].between(tudo[:, 1].min(), tudo[:, 1].max())
                 & z.isna())
        if caixa.any():
            idx = df.index[caixa]
            m = dentro(df.loc[idx, "lon"].to_numpy(), df.loc[idx, "lat"].to_numpy(), rings)
            z.loc[idx[m]] = f["properties"].get("nm_zona")
    return z


# zona pelo ponto medio do trecho que chega na parada (o mesmo criterio do od_agrupamentos.py e da tabela de zonas)
mid = pc.groupby("cp_destino").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
mid["zona"] = zona_de(mid)
zona_mid = dict(zip(mid["cp_destino"], mid["zona"]))
# coordenada real da parada (catalogo), para nomear as pontas dos fluxos e desenhar as rotas
try:
    cat = pd.read_parquet(RAIZ / "catalogo" / "paradas.parquet", columns=["cp", "nome", "endereco", "lat", "lon"])
    cat["cp"] = cat["cp"].astype("int64")
except Exception as e:  # noqa: BLE001
    print("catalogo de paradas indisponivel:", e)
    cat = pd.DataFrame(columns=["cp", "nome", "endereco", "lat", "lon"])
cps = pd.Index(pd.unique(np.r_[pc["cp_origem"].to_numpy(), pc["cp_destino"].to_numpy()]))
par = pd.DataFrame({"cp": cps}).merge(cat.drop_duplicates("cp"), on="cp", how="left")
falta = par["lat"].isna()
mi = mid.set_index("cp_destino")
par.loc[falta, "lat"] = par.loc[falta, "cp"].map(mi["lat"])
par.loc[falta, "lon"] = par.loc[falta, "cp"].map(mi["lon"])
print(f"paradas {len(par)} | coordenada do catalogo {int((~falta).sum())} | pelo ponto medio {int(falta.sum())}")
par = par.dropna(subset=["lat", "lon"]).reset_index(drop=True)
par["zona"] = zona_de(par)
zona_cp = dict(zip(par["cp"], par["zona"]))
dif_zona = sum(1 for c, z in zona_cp.items() if c in zona_mid and z != zona_mid[c])
print(f"paradas com zona diferente entre coordenada real e ponto medio do trecho: {dif_zona} de {len(zona_mid)}")

# ================================================================== A. fluxos com trecho comum
print("\n== A. viagem inteira num trecho comum (duas paradas fixas por linha, sentido e variante)")
s = pc.sort_values(["viagem_id", "ordem_origem"]).reset_index(drop=True)
nv = s.groupby("viagem_id")["ordem_origem"].transform("nunique")
nl = s.groupby(["cl", "sentido"])["ordem_origem"].transform("nunique")
base = s[nv >= 0.5 * nl]
tot = base.groupby(["cl", "sentido"])["viagem_id"].nunique().rename("tot").reset_index()
cov = (base.groupby(["cl", "sentido", "ordem_origem"])
       .agg(n=("viagem_id", "nunique"), cp_o=("cp_origem", lambda x: x.mode().iloc[0]),
            cp_d=("cp_destino", lambda x: x.mode().iloc[0]), cps_o=("cp_origem", "nunique"))
       .reset_index().merge(tot, on=["cl", "sentido"]))
cov["cob"] = cov["n"] / cov["tot"]
print(f"   posicoes (linha, sentido, ordem) com mais de uma parada de origem: {(cov['cps_o'] > 1).sum()} de {len(cov)}")
anc = []
for (cl, se), g in cov.groupby(["cl", "sentido"]):
    g = g.sort_values("ordem_origem")
    boa = g[g["cob"] >= 0.7]
    if len(boa) < 6:
        print(f"   cl {cl} sentido {se}: sem paradas fixas suficientes ({len(boa)} trechos com cobertura >= 0,7)")
        continue
    # o primeiro e o ultimo trecho bem registrados sao a saida e a chegada no terminal: incluem o tempo parado
    # no terminal e ficam de fora; a viagem e medida entre a parada seguinte e a anterior a eles
    a0, b0 = boa.iloc[0], boa.iloc[-1]
    a, b = boa.iloc[1], boa.iloc[-2]
    ab = g[(g["ordem_origem"] >= a["ordem_origem"]) & (g["ordem_origem"] <= b["ordem_origem"])]
    anc.append(dict(cl=int(cl), sentido=int(se), o_a0=int(a0["ordem_origem"]), o_b0=int(b0["ordem_origem"]),
                    o_a=int(a["ordem_origem"]), o_b=int(b["ordem_origem"]), cp_ini=int(a0["cp_o"]), cp_fim=int(b0["cp_d"]),
                    cp_a=int(a["cp_o"]), cp_b=int(b["cp_d"]), trechos_linha=len(g), trechos_ab=len(ab),
                    regulares_ab=int((ab["cob"] >= 0.5).sum()), cob_ini=round(float(g["cob"].iloc[0]), 2),
                    cob_fim=round(float(g["cob"].iloc[-1]), 2), viagens_base=int(g["tot"].iloc[0])))
anc = pd.DataFrame(anc)
reg = cov.loc[cov["cob"] >= 0.5, ["cl", "sentido", "ordem_origem"]].assign(regular=1)
t = s[["viagem_id", "cl", "letreiro", "sentido", "ordem_origem", "t_origem", "t_chegada", "tempo_percurso_s", "tipo", "hora"]]
t = t.merge(anc[["cl", "sentido", "o_a", "o_b", "regulares_ab"]], on=["cl", "sentido"])
t = t[(t["ordem_origem"] >= t["o_a"]) & (t["ordem_origem"] <= t["o_b"])]
t = t.merge(reg, on=["cl", "sentido", "ordem_origem"], how="left")
t["regular"] = t["regular"].fillna(0)
t = t.sort_values(["viagem_id", "ordem_origem"])
v = t.groupby("viagem_id").agg(cl=("cl", "first"), letreiro=("letreiro", "first"), sentido=("sentido", "first"),
                               o_a=("o_a", "first"), o_b=("o_b", "first"), regulares_ab=("regulares_ab", "first"),
                               omin=("ordem_origem", "min"), omax=("ordem_origem", "max"), n_reg=("regular", "sum"),
                               soma_s=("tempo_percurso_s", "sum"), t0=("t_origem", "first"), t1=("t_chegada", "last"),
                               tipo=("tipo", "first"), hora=("hora", "first"))
del t
n0 = len(v)
v = v[(v["omin"] == v["o_a"]) & (v["omax"] == v["o_b"])]
n1 = len(v)
v = v[v["n_reg"] >= 0.9 * v["regulares_ab"]].copy()
n2 = len(v)
v["dur_s"] = (v["t1"] - v["t0"]).dt.total_seconds()
med = v.groupby(["cl", "sentido"])["dur_s"].transform("median")
v = v[(v["dur_s"] > 0) & (v["dur_s"] <= 3 * med)].copy()
n3 = len(v)
print(f"   viagens com trecho entre as paradas fixas: {n0:,} | passam pelas duas: {n1:,} | com >= 90% dos trechos "
      f"regulares entre elas: {n2:,} | duracao valida (0 a 3x a mediana): {n3:,}")
g = (v["dur_s"] - v["soma_s"])
print(f"   ponta a ponta menos soma dos trechos (s): mediana {g.median():.1f} | p90 {g.quantile(0.9):.1f} | p99 {g.quantile(0.99):.1f}")
v["faixa"] = pd.cut(v["hora"], FAIXAS[0], labels=FAIXAS[1])
ch = ["cl", "sentido", "tipo", "faixa"]
v = v[v.groupby(ch, observed=True)["dur_s"].transform("size") >= 20].copy()
v["tipico_s"] = v.groupby(ch, observed=True)["dur_s"].transform("median")
v["atraso_min"] = (v["dur_s"] - v["tipico_s"]) / 60
v["rel"] = v["dur_s"] / v["tipico_s"] - 1
print(f"   viagens comparadas: {len(v):,} | duracao mediana {v['dur_s'].median() / 60:.1f} min | "
      f"trecho comum cobre, na mediana das linhas, {100 * (anc['trechos_ab'] / anc['trechos_linha']).median():.0f}% dos trechos")
for lim in (5, 10, 15):
    p = (v["atraso_min"] >= lim).mean()
    print(f"   {lim} min ou mais acima do normal: {100 * p:.1f}% (uma em cada {1 / p:.1f})")
print(f"   10% ou mais acima: {100 * (v['rel'] >= 0.1).mean():.1f}% | 20% ou mais: {100 * (v['rel'] >= 0.2).mean():.1f}% | "
      f"p90 do atraso {v['atraso_min'].quantile(0.9):.1f} min")
tab = v.groupby(["tipo", "faixa"], observed=True).agg(
    viagens=("atraso_min", "size"), tipico_min=("tipico_s", lambda x: x.median() / 60),
    pct_5min=("atraso_min", lambda x: 100 * (x >= 5).mean()), pct_10min=("atraso_min", lambda x: 100 * (x >= 10).mean()),
    pct_rel10=("rel", lambda x: 100 * (x >= 0.1).mean()))
print(tab.round(1).to_string())
print("   por linha (dois sentidos):")
print(v.groupby("letreiro").agg(
    viagens=("atraso_min", "size"), tipico_min=("tipico_s", lambda x: x.median() / 60),
    pct_5min=("atraso_min", lambda x: 100 * (x >= 5).mean()), pct_rel10=("rel", lambda x: 100 * (x >= 0.1).mean()),
    p90_min=("atraso_min", lambda x: x.quantile(0.9))).round(1).to_string())


def wilson(k, n, z=1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


fl = v.groupby(["letreiro", "cl", "sentido"]).agg(
    viagens=("rel", "size"), tipico_min=("tipico_s", lambda x: x.median() / 60),
    k10=("rel", lambda x: int((x >= 0.1).sum())), pct_rel20=("rel", lambda x: 100 * (x >= 0.2).mean()),
    pct_5min=("atraso_min", lambda x: 100 * (x >= 5).mean()), p90_rel=("rel", lambda x: 100 * x.quantile(0.9))).reset_index()
fl["pct_rel10"] = 100 * fl["k10"] / fl["viagens"]
fl["ic_lo"], fl["ic_hi"] = wilson(fl["k10"], fl["viagens"])
fl = fl.merge(anc, on=["cl", "sentido"])
fl["cobre_pct"] = 100 * fl["trechos_ab"] / fl["trechos_linha"]
fl["zona_a"] = fl["cp_a"].map(zona_cp)
fl["zona_b"] = fl["cp_b"].map(zona_cp)
fl["zona_ini"] = fl["cp_ini"].map(zona_cp)
fl["zona_fim"] = fl["cp_fim"].map(zona_cp)
nomes = par.set_index("cp")["nome"].to_dict()
fl["parada_a"] = fl["cp_a"].map(nomes)
fl["parada_b"] = fl["cp_b"].map(nomes)
fl["parada_ini"] = fl["cp_ini"].map(nomes)
fl["parada_fim"] = fl["cp_fim"].map(nomes)
fl = fl.sort_values("pct_rel10", ascending=False)
print(f"   fluxos (linha, sentido e variante): {len(fl)} | com >= 300 viagens: {(fl['viagens'] >= 300).sum()} | "
      f"media simples dos fluxos >= 300: {fl.loc[fl['viagens'] >= 300, 'pct_rel10'].mean():.1f}%")
bloco("FLUXOS3", fl.round(2))
ex = v[v["tipo"] == "util"].groupby(["letreiro", "cl", "sentido", "faixa"], observed=True).agg(
    viagens=("rel", "size"), tipico_min=("tipico_s", lambda x: x.median() / 60),
    pct_rel10=("rel", lambda x: 100 * (x >= 0.1).mean())).reset_index()
bloco("FLUXOS_FAIXA_UTIL", ex.round(2))
v["bin"] = (np.clip(v["rel"], -0.4, 0.8) // 0.05 * 5).astype(int)
hist = v.groupby(["letreiro", "cl", "sentido", "bin"]).size().rename("viagens").reset_index()
bloco("FLUXOS_HIST", hist)

# ================================================================== E. passagens sem os trechos de ponta
print("\n== E. passagens sem os trechos de ponta (saida e chegada no terminal)")
lims = pc.groupby(["cl", "sentido"])["ordem_origem"].agg(o_min="min", o_max="max").reset_index()
lims = lims.merge(anc[["cl", "sentido", "o_a0", "o_b0"]], on=["cl", "sentido"], how="left")
lims["o_a0"] = lims["o_a0"].fillna(lims["o_min"])
lims["o_b0"] = lims["o_b0"].fillna(lims["o_max"])
px = pc[["cl", "sentido", "letreiro", "ordem_origem", "cp_destino", "atrasado", "atraso_s", "tempo_percurso_s", "tipo", "hora"]].merge(
    lims[["cl", "sentido", "o_a0", "o_b0"]], on=["cl", "sentido"], how="left")
px["ponta"] = (px["ordem_origem"] <= px["o_a0"]) | (px["ordem_origem"] >= px["o_b0"])
px = px.dropna(subset=["atraso_s"])
print(f"   passagens em trechos de ponta: {int(px['ponta'].sum()):,} ({100 * px['ponta'].mean():.1f}%)")
print(px.groupby("ponta").agg(passagens=("atraso_s", "size"), atrasados_pct=("atrasado", lambda x: 100 * x.mean()),
                              tempo_mediano_s=("tempo_percurso_s", "median"),
                              p90_atraso_s=("atraso_s", lambda x: x.quantile(0.9))).round(1).to_string())
sem = px[~px["ponta"]]


def resumo(d):
    return pd.Series({"passagens": len(d), "atrasados_pct": 100 * d["atrasado"].mean(), "p90_atraso_s": d["atraso_s"].quantile(0.9)})


print("   geral (todas x sem pontas):")
print(pd.DataFrame({"todas": resumo(px), "sem_pontas": resumo(sem)}).round(2).to_string())
print("   por tipo de dia (atrasados %):")
print(pd.DataFrame({"todas": px.groupby("tipo")["atrasado"].mean() * 100,
                    "sem_pontas": sem.groupby("tipo")["atrasado"].mean() * 100}).round(1).to_string())
pu = px["tipo"].eq("util") & (px["hora"].between(6, 8) | px["hora"].between(16, 19))
su = sem["tipo"].eq("util") & (sem["hora"].between(6, 8) | sem["hora"].between(16, 19))
print(f"   pico de dia util: todas {100 * pu.mean():.1f}% das passagens e {100 * px.loc[pu, 'atrasado'].sum() / px['atrasado'].sum():.1f}% dos atrasados"
      f" | sem pontas {100 * su.mean():.1f}% e {100 * sem.loc[su, 'atrasado'].sum() / sem['atrasado'].sum():.1f}%")
print("   por linha (atrasados %):")
print(pd.DataFrame({"todas": px.groupby("letreiro")["atrasado"].mean() * 100,
                    "sem_pontas": sem.groupby("letreiro")["atrasado"].mean() * 100}).round(1).sort_values("todas").to_string())
px["zona"] = px["cp_destino"].map(zona_mid)
zz = px.dropna(subset=["zona"]).groupby("zona").agg(passagens=("atrasado", "size"), atrasados_pct=("atrasado", lambda x: 100 * x.mean()),
                                                   ponta_pct=("ponta", lambda x: 100 * x.mean()))
zs = px[~px["ponta"]].dropna(subset=["zona"]).groupby("zona").agg(passagens_sem=("atrasado", "size"),
                                                                 atrasados_sem_pct=("atrasado", lambda x: 100 * x.mean()))
zz = zz.join(zs).reset_index()
z5 = zz[zz["passagens"] >= 5000]
print(f"   zonas com >= 5 mil passagens: {len(z5)} | sem pontas e >= 5 mil: {(zz['passagens_sem'] >= 5000).sum()}")
print("   mais atrasadas (todas as passagens):")
print(z5.sort_values("atrasados_pct", ascending=False).head(8).round(1).to_string(index=False))
print("   menos atrasadas (todas as passagens):")
print(z5.sort_values("atrasados_pct").head(5).round(1).to_string(index=False))
z5s = zz[zz["passagens_sem"] >= 5000]
print("   mais atrasadas (sem pontas):")
print(z5s.sort_values("atrasados_sem_pct", ascending=False).head(8).round(1).to_string(index=False))
print("   menos atrasadas (sem pontas):")
print(z5s.sort_values("atrasados_sem_pct").head(5).round(1).to_string(index=False))
bloco("ZONAS_PONTA", zz.round(3))
del px, sem

# ================================================================== C. trechos de cada linha
cov2 = cov.merge(par[["cp", "lat", "lon", "nome"]].rename(columns={"cp": "cp_o"}), on="cp_o", how="left")
cov2["letreiro"] = cov2["cl"].map(pc.groupby("cl")["letreiro"].first())
bloco("TRECHOS_LINHA", cov2[["letreiro", "cl", "sentido", "ordem_origem", "cp_o", "cp_d", "cob", "lat", "lon", "nome"]],
      float_format="%.6f")
bloco("PARADAS", par[["cp", "nome", "endereco", "lat", "lon", "zona"]], float_format="%.6f")
del s, base, v

# ================================================================== B. episodios
print("\n== B. episodios ST-DBSCAN (400 m, 30 min, minimo 10), dia a dia")
pe = pc.dropna(subset=["atraso_s"]).reset_index(drop=True)
base_ev = pd.DataFrame({"valor_s": pe["atraso_s"].astype("float64"), "evento": pe["atrasado"].astype(bool),
                        "t": pe["t_chegada"], "cp": pe["cp_destino"].astype("int64"),
                        "lat": pe["lat"], "lon": pe["lon"], "dia": pe["dia"], "i": np.arange(len(pe))}).dropna(subset=["lat", "lon"])
ev = regioes.preparar_amostras(base_ev, proj, "eventos").reset_index(drop=True)


def expandir(viz, mp):
    n = len(viz)
    rot = np.full(n, -2, dtype="int64")
    c = 0
    for i in range(n):
        if rot[i] != -2:
            continue
        if len(viz[i]) < mp:
            rot[i] = -1
            continue
        rot[i] = c
        fila = deque(viz[i])
        while fila:
            j = fila.popleft()
            if rot[j] == -1:
                rot[j] = c
                continue
            if rot[j] != -2:
                continue
            rot[j] = c
            if len(viz[j]) >= mp:
                fila.extend(k for k in viz[j] if rot[k] == -2)
        c += 1
    rot[rot == -2] = -1
    return rot


partes = []
for k, (d, a) in enumerate(ev.groupby("dia")):
    a = a.reset_index(drop=True)
    ts = (pd.to_datetime(a["t"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    viz = regioes._vizinhanca_espaco_tempo(a["x_m"].to_numpy(), a["y_m"].to_numpy(), ts, 400.0, 1800.0)
    r = expandir(viz, 10)
    partes.append(a.assign(ep=np.where(r >= 0, r + 1_000_000 * k, -1)))
    del viz
ep = pd.concat(partes, ignore_index=True)
E = ep[ep["ep"] >= 0].copy()
E = E.join(pe[["letreiro", "sentido", "cp_origem", "tempo_percurso_s", "percurso_tipico_s", "tipo", "hora"]], on="i")
E["zona"] = E["cp"].map(zona_mid)
cx = E.groupby("ep")["x_m"].transform("mean")
cy = E.groupby("ep")["y_m"].transform("mean")
E["dist_m"] = np.hypot(E["x_m"] - cx, E["y_m"] - cy)
S = E.groupby("ep").agg(n=("dist_m", "size"), r90=("dist_m", lambda d: d.quantile(0.9)), t0=("t", "min"), t1=("t", "max"),
                        dia=("dia", "first"), tipo=("tipo", "first"), lat=("lat", "mean"), lon=("lon", "mean"),
                        x=("x_m", "mean"), y=("y_m", "mean"), linhas=("letreiro", "nunique"),
                        lista_linhas=("letreiro", lambda s: " ".join(sorted(set(map(str, s))))), paradas=("cp", "nunique"),
                        zona=("zona", lambda s: s.mode().iloc[0] if s.notna().any() else None))
S["dur_min"] = (S["t1"] - S["t0"]).dt.total_seconds() / 60
S["h0"] = pd.to_datetime(S["t0"], utc=True).dt.tz_convert(FUSO).dt.hour
print(f"   ocorrencias de atraso {len(ep):,} | episodios {len(S):,} | ruido {100 * (1 - len(E) / len(ep)):.1f}% | mediana: "
      f"{S['n'].median():.0f} ocorrencias, raio p90 {S['r90'].median():.0f} m, duracao {S['dur_min'].median():.0f} min")
print(f"   episodios com 2 linhas ou mais: {100 * (S['linhas'] >= 2).mean():.1f}% | com 3 ou mais: {100 * (S['linhas'] >= 3).mean():.1f}%")
pico = (S["tipo"] == "util") & (S["h0"].between(6, 8) | S["h0"].between(16, 19))
print(f"   episodios que comecam no pico de dia util: {100 * pico.mean():.1f}%")
dias_cp = E.groupby("cp")["dia"].nunique()
print(f"   paradas em algum episodio: {len(dias_cp)} | em 5 dias ou mais: {100 * (dias_cp >= 5).mean():.1f}%")
pz = pe.assign(zona=pe["cp_destino"].map(zona_mid)).dropna(subset=["zona"]).groupby("zona").agg(passagens=("atraso_s", "size"))
pz["episodios"] = S["zona"].value_counts().reindex(pz.index).fillna(0).astype(int)
pz["ep_10mil"] = 1e4 * pz["episodios"] / pz["passagens"]
print("   por 10 mil passagens (zonas >= 5 mil passagens):")
print(pz[pz["passagens"] >= 5000].sort_values("ep_10mil", ascending=False).head(8).round(1).to_string())

du = sorted(pe.loc[pe["tipo"] == "util", "dia"].unique())
Su = S[S["tipo"] == "util"]
dz = Su.groupby("zona")["dia"].nunique()
dzt = Su[Su["h0"].between(16, 19)].groupby("zona")["dia"].nunique()
dpz = pe[pe["tipo"] == "util"].assign(zona=pe["cp_destino"].map(zona_mid)).dropna(subset=["zona"]).groupby("zona")["dia"].nunique()
out = pz.assign(dias_util_ep=dz.reindex(pz.index).fillna(0).astype(int),
                dias_util_ep_tarde=dzt.reindex(pz.index).fillna(0).astype(int),
                dias_util_com_dados=dpz.reindex(pz.index).fillna(0).astype(int)).reset_index()
print("dias uteis na base:", len(du))
# sem trechos de terminal: tira as ocorrencias de ponta antes de atribuir zona; descarta episodio com mais da metade em ponta
pp = pe[["cl", "sentido", "ordem_origem"]].merge(lims[["cl", "sentido", "o_a0", "o_b0"]], on=["cl", "sentido"], how="left")
pe["ponta"] = ((pp["ordem_origem"] <= pp["o_a0"]) | (pp["ordem_origem"] >= pp["o_b0"])).to_numpy()
E["ponta"] = pe["ponta"].to_numpy()[E["i"].to_numpy()]
frac = E.groupby("ep")["ponta"].mean()
zs = E[~E["ponta"]].groupby("ep")["zona"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)
S2 = S.assign(ponta_frac=frac, zona_sp=zs)
S2 = S2[S2["ponta_frac"] <= 0.5]
Su2 = S2[S2["tipo"] == "util"]
dzt2 = Su2[Su2["h0"].between(16, 19)].groupby("zona_sp")["dia"].nunique()
pes = pe[~pe["ponta"]].assign(zona=pe.loc[~pe["ponta"], "cp_destino"].map(zona_mid)).dropna(subset=["zona"])
pas_sp = pes.groupby("zona")["atraso_s"].size()
ponta_pct = pe.assign(zona=pe["cp_destino"].map(zona_mid)).dropna(subset=["zona"]).groupby("zona")["ponta"].mean() * 100
out["dias_util_ep_tarde_sem_ponta"] = out["zona"].map(dzt2).fillna(0).astype(int)
out["passagens_sem_ponta"] = out["zona"].map(pas_sp).fillna(0).astype(int)
out["ponta_pct"] = out["zona"].map(ponta_pct).round(1)
print(f"   episodios descartados por ponta (> metade): {100 * (frac > 0.5).mean():.1f}%")
v = out[out["passagens_sem_ponta"] >= 5000]
print(f"   zonas >= 5 mil passagens sem ponta: {len(v)} | com >= 15 dias (tarde, sem ponta): {(v['dias_util_ep_tarde_sem_ponta'] >= 15).sum()}")
print(v.sort_values("dias_util_ep_tarde_sem_ponta", ascending=False).head(15)[["zona", "dias_util_ep_tarde", "dias_util_ep_tarde_sem_ponta", "ponta_pct", "passagens_sem_ponta"]].to_string(index=False))
bloco("ZONAS_DIAS", out)
sys.exit(0)
# sensibilidade: quanto dos episodios vem dos trechos de ponta (saida e chegada no terminal)
pp = pe[["cl", "sentido", "ordem_origem"]].merge(lims[["cl", "sentido", "o_a0", "o_b0"]], on=["cl", "sentido"], how="left")
pe["ponta"] = ((pp["ordem_origem"] <= pp["o_a0"]) | (pp["ordem_origem"] >= pp["o_b0"])).to_numpy()
E["ponta"] = pe["ponta"].to_numpy()[E["i"].to_numpy()]
S["ponta_frac"] = E.groupby("ep")["ponta"].mean()
print(f"   ocorrencias de episodio em trechos de ponta: {100 * E['ponta'].mean():.1f}% | passagens de ponta na base: {100 * pe['ponta'].mean():.1f}%"
      f" | episodios com mais da metade em ponta: {100 * (S['ponta_frac'] > 0.5).mean():.1f}%")
pz["episodios_sem_ponta"] = S.loc[S["ponta_frac"] <= 0.5, "zona"].value_counts().reindex(pz.index).fillna(0).astype(int)
pz["ep_10mil_sem_ponta"] = 1e4 * pz["episodios_sem_ponta"] / pz["passagens"]
print("   por 10 mil passagens, sem episodios de ponta (zonas >= 5 mil passagens):")
print(pz[pz["passagens"] >= 5000].sort_values("ep_10mil_sem_ponta", ascending=False).head(8).round(1).to_string())
bloco("ZONAS_EPISODIOS", pz.reset_index().round(3))

TOP = ["Vila Olímpia", "Pamplona", "Moema", "Tatuapé", "Lapa de Baixo"]
cand = S[(S["tipo"] == "util") & S["h0"].between(16, 19) & S["n"].between(15, 60) & S["dur_min"].between(40, 150)
         & S["r90"].between(120, 450) & (S["linhas"] >= 2) & (S["ponta_frac"] <= 0.2)].copy()
cand["top"] = cand["zona"].isin(TOP)
cand["dist_tipico"] = (cand["n"] - 25).abs()
cand = cand.sort_values(["top", "linhas", "dist_tipico"], ascending=[False, False, True])
mostra = cand.head(30).copy()
mostra["ini"] = hhmm(mostra["t0"])
mostra["fim"] = hhmm(mostra["t1"])
print(f"   candidatos a exemplo (dia util, comeca entre 16h e 19h, 15 a 60 ocorrencias, 2+ linhas): {len(cand)}")
print(mostra[["dia", "ini", "fim", "dur_min", "n", "r90", "linhas", "lista_linhas", "paradas", "zona"]].round(0).to_string())

# um exemplo por zona do topo (o mais tipico), mais o melhor candidato geral se faltar zona
esc = []
for z in TOP:
    c = cand[cand["zona"] == z]
    if len(c):
        esc.append(c.index[0])
for e_id in cand.index:
    if len(esc) >= 6:
        break
    if e_id not in esc:
        esc.append(e_id)
pe_xy = np.column_stack(proj.para_xy(pe["lat"].to_numpy(), pe["lon"].to_numpy()))
ep_de = pd.Series(E["ep"].to_numpy(), index=E["i"].to_numpy())
pe["ep"] = ep_de.reindex(np.arange(len(pe))).fillna(-1).astype("int64").to_numpy()
util = pe["tipo"] == "util"
ex_ev, ex_ctx, ex_tl, ex_rec, ex_perf = [], [], [], [], []
for nume, e_id in enumerate(esc, 1):
    r = S.loc[e_id]
    ee = E[E["ep"] == e_id]
    ex_ev.append(pd.DataFrame({"ex": nume, "hora": hhmm(ee["t"]), "lat": ee["lat"], "lon": ee["lon"], "letreiro": ee["letreiro"],
                               "sentido": ee["sentido"], "cp_origem": ee["cp_origem"], "cp_destino": ee["cp"],
                               "tempo_s": ee["tempo_percurso_s"], "tipico_s": ee["percurso_tipico_s"], "atraso_s": ee["valor_s"]}))
    dd = np.hypot(pe_xy[:, 0] - r["x"], pe_xy[:, 1] - r["y"])
    jan = (pe["dia"] == r["dia"]) & (pe["t_chegada"] >= r["t0"] - pd.Timedelta(minutes=90)) & (pe["t_chegada"] <= r["t1"] + pd.Timedelta(minutes=90))
    c = pe[jan & (dd <= 700)]
    ex_ctx.append(pd.DataFrame({"ex": nume, "hora": hhmm(c["t_chegada"]), "lat": c["lat"], "lon": c["lon"], "letreiro": c["letreiro"],
                                "sentido": c["sentido"], "cp_origem": c["cp_origem"], "cp_destino": c["cp_destino"],
                                "tempo_s": c["tempo_percurso_s"], "tipico_s": c["percurso_tipico_s"], "atraso_s": c["atraso_s"],
                                "atrasado": c["atrasado"].astype(int), "ep": np.where(c["ep"] == e_id, 1, np.where(c["ep"] >= 0, 2, 0))}))
    # linha do tempo no dia, nos trechos do episodio
    tre = set(zip(ee["cp_origem"], ee["cp"]))
    no_tre = pd.Series([(a, b) in tre for a, b in zip(pe["cp_origem"], pe["cp_destino"])], index=pe.index)
    dia = pe[no_tre & (pe["dia"] == r["dia"])]
    slot = pd.to_datetime(dia["t_chegada"], utc=True).dt.tz_convert(FUSO).dt.floor("15min").dt.strftime("%H:%M")
    tl = dia.groupby(slot).agg(passagens=("atrasado", "size"), atrasadas=("atrasado", "sum")).reset_index().rename(columns={"t_chegada": "slot"})
    tl.insert(0, "ex", nume)
    ex_tl.append(tl)
    # perfil desses trechos em todos os dias uteis
    tu = pe[no_tre & util]
    pf = tu.groupby("hora").agg(passagens=("atrasado", "size"), atrasadas=("atrasado", "sum"), dias=("dia", "nunique")).reset_index()
    pf.insert(0, "ex", nume)
    ex_perf.append(pf)
    # recorrencia: episodios (qualquer dia) com centro a ate 300 m deste
    rec = S[np.hypot(S["x"] - r["x"], S["y"] - r["y"]) <= 300].copy()
    ex_rec.append(pd.DataFrame({"ex": nume, "dia": rec["dia"], "tipo": rec["tipo"], "ini": hhmm(rec["t0"]), "fim": hhmm(rec["t1"]),
                                "n": rec["n"], "linhas": rec["lista_linhas"]}))
    print(f"   exemplo {nume}: {r['dia']} {hhmm(pd.Series([r['t0']])).iloc[0]}-{hhmm(pd.Series([r['t1']])).iloc[0]} | zona {r['zona']} | "
          f"{r['n']} ocorrencias, {r['linhas']} linhas ({r['lista_linhas']}), raio p90 {r['r90']:.0f} m | "
          f"contexto {len(c)} passagens | dias uteis com episodio a ate 300 m: {rec.loc[rec['tipo'] == 'util', 'dia'].nunique()} "
          f"de {pe.loc[util, 'dia'].nunique()}")
bloco("EX_EVENTOS", pd.concat(ex_ev), float_format="%.6f")
bloco("EX_CONTEXTO", pd.concat(ex_ctx), float_format="%.6f")
bloco("EX_LINHA_TEMPO", pd.concat(ex_tl))
bloco("EX_PERFIL_UTIL", pd.concat(ex_perf))
bloco("EX_RECORRENCIA", pd.concat(ex_rec))
print("fim")
