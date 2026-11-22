"""Excesso de atraso por zona contra o esperado pela linha, sentido, hora e tipo de dia. So leitura; saida so por print.

(1) taxa por zona (sem terminal, >= 5 mil passagens) com a zona da PARADA DE ORIGEM x zona do ponto medio
    do trecho (criterio do relatorio); Spearman e quantas das 5+5 extremas se mantem.
(2) bootstrap por dia (2000 reamostras) da diferenca, na 5300, entre Praca Joao Mendes e Centro Cultural.
(3) decomposicao zona x linha: celulas >= 1000 passagens; dispersao entre linhas na mesma zona x entre zonas na
    mesma linha; regressao logistica (IRLS em numpy) atrasado ~ tipo x hora + linha + zona, queda de deviance.
(4) correlacao entre a taxa da zona e o tempo normal mediano do trecho.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
RAIZ = Path("/dados")
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}
rng = np.random.default_rng(20261116)

pc = pd.read_parquet(RAIZ / "derivado" / "percursos.parquet", columns=[
    "cl", "letreiro", "sentido", "viagem_id", "cp_origem", "cp_destino", "ordem_origem", "t_origem",
    "tempo_percurso_s", "percurso_tipico_s", "atraso_s", "atrasado", "lat", "lon", "velocidade_kmh"]).dropna(
    subset=["tempo_percurso_s", "atraso_s"]).reset_index(drop=True)
pc["cp_origem"] = pc["cp_origem"].astype("int64")
pc["cp_destino"] = pc["cp_destino"].astype("int64")
loc = pd.to_datetime(pc["t_origem"], utc=True).dt.tz_convert(FUSO)
pc["dia"] = loc.dt.strftime("%Y-%m-%d")
pc["hora"] = loc.dt.hour
dow = loc.dt.dayofweek
pc["tipo"] = np.where(pc["dia"].isin(FERIADOS) | (dow == 6) | (pc["dia"] == "2026-10-25"), "domingo/feriado",
                      np.where(dow == 5, "sabado", "util"))
pc["atrasado"] = pc["atrasado"].astype(bool)
reg = pc["tempo_percurso_s"] >= 1.2 * pc["percurso_tipico_s"]
print(f"passagens {len(pc):,} | dias {pc['dia'].nunique()} ({pc['dia'].min()} a {pc['dia'].max()}) | atrasadas {100 * pc['atrasado'].mean():.2f}%"
      f" | coluna atrasado = (tempo >= 1,2 x normal) em {100 * (reg == pc['atrasado']).mean():.2f}% das passagens")

# ---------------------------------------------------------------- trechos de terminal (mesmo criterio do zonas_ponta.py)
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
del x, base, nv, nl
print(f"trechos de terminal: {100 * pc['ponta'].mean():.1f}% das passagens")

# ---------------------------------------------------------------- zonas
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
        caixa = (df["lon"].between(tudo[:, 0].min(), tudo[:, 0].max()) & df["lat"].between(tudo[:, 1].min(), tudo[:, 1].max()) & z.isna())
        if caixa.any():
            idx = df.index[caixa]
            m = dentro(df.loc[idx, "lon"].to_numpy(), df.loc[idx, "lat"].to_numpy(), rings)
            z.loc[idx[m]] = f["properties"].get("nm_zona")
    return z


mid = pc.groupby("cp_destino").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
mid["zona"] = zona_de(mid)
pc["zona_mid"] = pc["cp_destino"].map(dict(zip(mid["cp_destino"], mid["zona"])))
try:
    cat = pd.read_parquet(RAIZ / "catalogo" / "paradas.parquet", columns=["cp", "nome", "lat", "lon"])
    cat["cp"] = cat["cp"].astype("int64")
    cat = cat.drop_duplicates("cp").dropna(subset=["lat", "lon"])
except Exception as e:  # noqa: BLE001
    print("catalogo de paradas indisponivel:", e)
    cat = pd.DataFrame(columns=["cp", "nome", "lat", "lon"])
orig = pd.DataFrame({"cp": pc["cp_origem"].unique()}).merge(cat, on="cp", how="left")
falta = orig["lat"].isna()
# parada de origem sem coordenada no catalogo: usa a coordenada do ponto medio do trecho que chega nela
mi = mid.set_index("cp_destino")
orig.loc[falta, "lat"] = orig.loc[falta, "cp"].map(mi["lat"])
orig.loc[falta, "lon"] = orig.loc[falta, "cp"].map(mi["lon"])
falta2 = orig["lat"].isna()
orig = orig.dropna(subset=["lat", "lon"]).reset_index(drop=True)
orig["zona"] = zona_de(orig)
pc["zona_ori"] = pc["cp_origem"].map(dict(zip(orig["cp"], orig["zona"])))
print(f"paradas de origem {len(orig) + int(falta2.sum())} | do catalogo {int((~falta).sum())} | pelo ponto medio {int((falta & ~falta2).sum())}"
      f" | sem coordenada {int(falta2.sum())}")
print(f"passagens com zona: ponto medio {100 * pc['zona_mid'].notna().mean():.1f}% | parada de origem {100 * pc['zona_ori'].notna().mean():.1f}%"
      f" | mesma zona nos dois criterios (entre as com as duas) {100 * (pc['zona_mid'] == pc['zona_ori'])[pc['zona_mid'].notna() & pc['zona_ori'].notna()].mean():.1f}%")

def rank(s):
    return s.rank(method="average")


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])



sem = pc[~pc["ponta"]].copy()
# segundo e penultimo trecho (robustez usada no artigo)
o2 = sem.groupby(["cl", "sentido"])["ordem_origem"].transform(lambda s: np.sort(s.unique())[1] if s.nunique() > 2 else -1)
op = sem.groupby(["cl", "sentido"])["ordem_origem"].transform(lambda s: np.sort(s.unique())[-2] if s.nunique() > 2 else -1)
sem["borda"] = ((sem["ordem_origem"] == o2) | (sem["ordem_origem"] == op)).to_numpy()
print(f"passagens sem terminal: {len(sem):,} | atrasadas {100 * sem['atrasado'].mean():.2f}% | 2o/penultimo trecho {100 * sem['borda'].mean():.1f}%")
dias = np.array(sorted(sem["dia"].unique()))
D = len(dias)
B = 4000


def excesso(d, col, rotulo, minimo=5000, B=B):
    """Excesso = taxa observada da zona - taxa esperada pela mistura de celulas (cl x hora x tipo) da zona.
    Bootstrap por dia: reamostra dias, recalcula a taxa das celulas e o esperado de cada zona."""
    d = d.dropna(subset=[col])
    d = d.assign(cel=d["cl"].astype(str) + "|" + d["hora"].astype(str) + "|" + d["tipo"])
    g = d.groupby([col, "cel", "dia"])["atrasado"].agg(k="sum", n="size").reset_index()
    zcod, zlv = pd.factorize(g[col])
    ccod, clv = pd.factorize(g["cel"])
    dcod = pd.Index(dias).get_indexer(g["dia"])
    k = g["k"].to_numpy(float)
    n = g["n"].to_numpy(float)
    nz, nc = len(zlv), len(clv)

    def calc(w):
        wd = w[dcod]
        ck = np.bincount(ccod, wd * k, nc)
        cn = np.bincount(ccod, wd * n, nc)
        r = np.divide(ck, cn, out=np.zeros(nc), where=cn > 0)
        zk = np.bincount(zcod, wd * k, nz)
        zn = np.bincount(zcod, wd * n, nz)
        ze = np.bincount(zcod, wd * n * r[ccod], nz)
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100 * zk / zn, 100 * ze / zn, zn

    obs, esp, zn = calc(np.ones(D))
    rng2 = np.random.default_rng(7)
    boots = np.empty((B, nz))
    for b in range(B):
        w = np.bincount(rng2.integers(0, D, D), minlength=D).astype(float)
        o, e, _ = calc(w)
        boots[b] = o - e
    nvalid = int((zn >= minimo).sum())
    alfa_b = 0.05 / max(1, nvalid)
    t = pd.DataFrame({"zona": zlv, "passagens": zn.astype(int), "taxa": obs, "esperada": esp, "excesso": obs - esp,
                      "ic_lo": np.nanpercentile(boots, 2.5, axis=0), "ic_hi": np.nanpercentile(boots, 97.5, axis=0),
                      "bonf_lo": np.nanpercentile(boots, 100 * alfa_b / 2, axis=0),
                      "bonf_hi": np.nanpercentile(boots, 100 * (1 - alfa_b / 2), axis=0),
                      "p_acima": (boots <= 0).mean(0), "p_abaixo": (boots >= 0).mean(0)})
    t = t[t["passagens"] >= minimo].sort_values("excesso", ascending=False).reset_index(drop=True)
    p = np.minimum(1, 2 * np.minimum(t["p_acima"], t["p_abaixo"]).clip(lower=1 / B)).to_numpy()
    o = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    acc = 1.0
    for i in range(m - 1, -1, -1):
        acc = min(acc, p[o[i]] * m / (i + 1))
        q[o[i]] = acc
    t["q_bh"] = q
    acima95 = t[t["ic_lo"] > 0]
    abaixo95 = t[t["ic_hi"] < 0]
    acimaB = t[t["bonf_lo"] > 0]
    abaixoB = t[t["bonf_hi"] < 0]
    print(f"\n== [{rotulo}] {len(t)} zonas >= {minimo} passagens | B={B} reamostras de {D} dias")
    print(f"   DP da taxa bruta entre zonas {t['taxa'].std():.2f} pp | DP do excesso {t['excesso'].std():.2f} pp | excesso de {t['excesso'].min():.1f} a {t['excesso'].max():.1f} pp")
    print(f"   IC 95% inteiro acima de 0: {len(acima95)} zonas | abaixo: {len(abaixo95)}")
    print(f"   com correcao de Bonferroni ({len(t)} zonas): acima {len(acimaB)} | abaixo {len(abaixoB)}")
    print(f"   Benjamini-Hochberg q<0,05: acima {int(((t['q_bh'] < 0.05) & (t['excesso'] > 0)).sum())} | abaixo {int(((t['q_bh'] < 0.05) & (t['excesso'] < 0)).sum())}")
    print("   acima (Bonferroni):", ", ".join(f"{r.zona} (+{r.excesso:.1f}, {r.bonf_lo:.1f} a {r.bonf_hi:.1f})" for r in acimaB.itertuples()) or "nenhuma")
    print("   abaixo (Bonferroni):", ", ".join(f"{r.zona} ({r.excesso:.1f}, {r.bonf_lo:.1f} a {r.bonf_hi:.1f})" for r in abaixoB.itertuples()) or "nenhuma")
    print(t.head(12).round(2).to_string())
    print("   ...")
    print(t.tail(8).round(2).to_string())
    return t


t_mid = excesso(sem, "zona_mid", "ponto medio, sem terminal (criterio do artigo)")
t_ori = excesso(sem, "zona_ori", "parada de origem, sem terminal")
t_brd = excesso(sem[~sem["borda"]], "zona_mid", "ponto medio, sem terminal e sem 2o/penultimo trecho")
t_utl = excesso(sem[sem["tipo"] == "util"], "zona_mid", "ponto medio, so dia util", minimo=3500)

j = t_mid.set_index("zona")[["excesso", "ic_lo", "ic_hi", "bonf_lo", "bonf_hi"]].join(
    t_ori.set_index("zona")[["excesso", "bonf_lo", "bonf_hi"]].add_suffix("_ori"), how="left").join(
    t_brd.set_index("zona")[["excesso", "bonf_lo", "bonf_hi"]].add_suffix("_brd"), how="left")
jo = j.dropna(subset=["excesso_ori"])
jb = j.dropna(subset=["excesso_brd"])
print(f"\n== concordancia: Spearman excesso ponto medio x origem {spearman(jo['excesso'], jo['excesso_ori']):.3f} ({len(jo)} zonas)"
      f" | x sem 2o/penultimo {spearman(jb['excesso'], jb['excesso_brd']):.3f} ({len(jb)} zonas)")
j["acima_mid"] = j["bonf_lo"] > 0
j["acima_ori"] = j["bonf_lo_ori"] > 0
j["acima_brd"] = j["bonf_lo_brd"] > 0
j["abaixo_mid"] = j["bonf_hi"] < 0
j["abaixo_ori"] = j["bonf_hi_ori"] < 0
j["abaixo_brd"] = j["bonf_hi_brd"] < 0
rob_a = j[j["acima_mid"] & j["acima_ori"] & j["acima_brd"]]
rob_b = j[j["abaixo_mid"] & j["abaixo_ori"] & j["abaixo_brd"]]
print(f"   acima nos 3 criterios (Bonferroni): {len(rob_a)}: " + ", ".join(f"{z} (+{r.excesso:.1f})" for z, r in rob_a.iterrows()))
print(f"   abaixo nos 3 criterios (Bonferroni): {len(rob_b)}: " + ", ".join(f"{z} ({r.excesso:.1f})" for z, r in rob_b.iterrows()))
rob_a2 = j[j["acima_mid"] & (j["acima_ori"] | j["acima_brd"])]
print(f"   acima no artigo e em pelo menos 1 dos outros 2: {len(rob_a2)}: " + ", ".join(rob_a2.index))
rob_a3 = j[j["acima_mid"] & j["acima_brd"]]
print(f"   acima no artigo e sem 2o/penultimo: {len(rob_a3)}: " + ", ".join(rob_a3.index))

print("\n== velocidade: queda no fim de tarde de dia util (16h-19h) contra domingo/feriado de manha (5h-11h), mesmo trecho")
v = sem.dropna(subset=["velocidade_kmh", "zona_mid"])
v = v[(v["velocidade_kmh"] > 0.5) & (v["velocidade_kmh"] < 90)]
a = v[(v["tipo"] == "util") & v["hora"].between(16, 19)].groupby(["cl", "cp_origem", "cp_destino"]).agg(
    vu=("velocidade_kmh", "median"), nu=("velocidade_kmh", "size"))
b = v[(v["tipo"] == "domingo/feriado") & v["hora"].between(5, 11)].groupby(["cl", "cp_origem", "cp_destino"]).agg(
    vd=("velocidade_kmh", "median"), nd=("velocidade_kmh", "size"))
tr = a.join(b, how="inner")
tr = tr[(tr["nu"] >= 30) & (tr["nd"] >= 10)].reset_index()
tr["queda"] = 100 * (1 - tr["vu"] / tr["vd"])
tr["zona"] = tr["cp_destino"].map(dict(zip(mid["cp_destino"], mid["zona"])))
rede = np.average(tr["queda"], weights=tr["nu"])
zq = tr.groupby("zona").apply(lambda g: pd.Series({"trechos": len(g), "queda_pct": np.average(g["queda"], weights=g["nu"]),
                                                    "vel_util": np.average(g["vu"], weights=g["nu"])})).reset_index()
print(f"   trechos pareados {len(tr)} | queda media da rede {rede:.1f}%")
z = t_mid.merge(zq, on="zona", how="left")
z["queda_menos_rede"] = z["queda_pct"] - rede
ok = z.dropna(subset=["queda_pct"])
print(f"   zonas com as duas medidas: {len(ok)} | Spearman excesso de atraso x queda de velocidade {spearman(ok['excesso'], ok['queda_pct']):.3f}")
ac = z[z["bonf_lo"] > 0]
print(f"   zonas acima (Bonferroni, criterio do artigo) com queda de velocidade acima da rede: {int((ac['queda_menos_rede'] > 0).sum())} de {len(ac)}")
for r in ac.itertuples():
    print(f"      {r.zona}: excesso +{r.excesso:.1f} pp ({r.bonf_lo:.1f} a {r.bonf_hi:.1f}) | queda {r.queda_pct:.1f}% (rede {rede:.1f}%) | {r.trechos} trechos")
print("   10 zonas com maior queda de velocidade:")
print(zq.sort_values("queda_pct", ascending=False).head(10).round(1).to_string())

lin_z = sem.dropna(subset=["zona_mid"]).groupby("zona_mid")["letreiro"].agg(lambda s: ",".join(sorted(s.astype(str).str[:4].unique())))
z["linhas"] = z["zona"].map(lin_z)
z["pos_bruta"] = z["taxa"].rank(ascending=False).astype(int)
z["pos_excesso"] = z["excesso"].rank(ascending=False).astype(int)
print(f"\nSpearman taxa bruta x excesso: {spearman(z['taxa'], z['excesso']):.3f}")
for zz in ("Capão Redondo", "Vila Cordeiro", "Sé", "Pamplona", "Jardins", "Praça João Mendes", "Centro Cultural", "Tatuapé", "Vila Ré", "Artur Alvim"):
    r = z[z["zona"] == zz]
    if len(r):
        r = r.iloc[0]
        print(f"   {zz}: taxa {r.taxa:.1f}% (pos {r.pos_bruta}) | esperada {r.esperada:.1f}% | excesso {r.excesso:+.1f} (pos {r.pos_excesso}; IC {r.ic_lo:.1f} a {r.ic_hi:.1f};"
              f" Bonf {r.bonf_lo:.1f} a {r.bonf_hi:.1f}) | queda vel {r.queda_pct:.1f}% | linhas {r.linhas}")
print("###EXCESSO_ZONAS")
print(z.round(3).to_csv(index=False))
print("###FIM")
print("fim")
