"""Robustez de "o lugar pesa mais que a linha". So leitura; saida so por print.

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
    "tempo_percurso_s", "percurso_tipico_s", "atraso_s", "atrasado", "lat", "lon"]).dropna(
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

sem = pc[~pc["ponta"]].copy()
print(f"passagens sem terminal: {len(sem):,} | atrasadas {100 * sem['atrasado'].mean():.2f}%")


def rank(s):
    return s.rank(method="average")


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


# ================================================================ (1) zona da parada de origem
print("\n== (1) zona da parada de origem x zona do ponto medio (sem terminal, >= 5 mil passagens)")


def taxa(d, col):
    t = d.dropna(subset=[col]).groupby(col).agg(passagens=("atrasado", "size"), atrasados_pct=("atrasado", lambda s: 100 * s.mean()))
    return t[t["passagens"] >= 5000]


tm = taxa(sem, "zona_mid")
to = taxa(sem, "zona_ori")
print(f"zonas >= 5 mil: ponto medio {len(tm)} (de {tm['atrasados_pct'].min():.1f}% a {tm['atrasados_pct'].max():.1f}%) | "
      f"parada de origem {len(to)} (de {to['atrasados_pct'].min():.1f}% a {to['atrasados_pct'].max():.1f}%)")
j = tm.join(to, lsuffix="_mid", rsuffix="_ori", how="inner")
print(f"zonas nas duas: {len(j)} | Spearman {spearman(j['atrasados_pct_mid'], j['atrasados_pct_ori']):.3f} | Pearson "
      f"{np.corrcoef(j['atrasados_pct_mid'], j['atrasados_pct_ori'])[0, 1]:.3f} | diferenca absoluta mediana "
      f"{(j['atrasados_pct_mid'] - j['atrasados_pct_ori']).abs().median():.2f} pp | maxima {(j['atrasados_pct_mid'] - j['atrasados_pct_ori']).abs().max():.2f} pp")
for k in (5, 10):
    top_m = set(tm.nlargest(k, "atrasados_pct").index)
    bot_m = set(tm.nsmallest(k, "atrasados_pct").index)
    top_o = set(to.nlargest(k, "atrasados_pct").index)
    bot_o = set(to.nsmallest(k, "atrasados_pct").index)
    print(f"   {k} mais atrasadas que se mantem no top {k}: {len(top_m & top_o)} de {k} | {k} menos atrasadas que se mantem: {len(bot_m & bot_o)} de {k}")
    if k == 5:
        print("   top 5 (ponto medio):", ", ".join(f"{z} ({tm.loc[z, 'atrasados_pct']:.1f}%)" for z in tm.nlargest(5, "atrasados_pct").index))
        print("   top 5 (origem):     ", ", ".join(f"{z} ({to.loc[z, 'atrasados_pct']:.1f}%)" for z in to.nlargest(5, "atrasados_pct").index))
        print("   base 5 (ponto medio):", ", ".join(f"{z} ({tm.loc[z, 'atrasados_pct']:.1f}%)" for z in tm.nsmallest(5, "atrasados_pct").index))
        print("   base 5 (origem):     ", ", ".join(f"{z} ({to.loc[z, 'atrasados_pct']:.1f}%)" for z in to.nsmallest(5, "atrasados_pct").index))
        for z in sorted(top_m | bot_m):
            r_o = int(rank(-to["atrasados_pct"]).get(z, np.nan)) if z in to.index else None
            print(f"      {z}: ponto medio {tm.loc[z, 'atrasados_pct']:.1f}% (posicao {int(rank(-tm['atrasados_pct'])[z])} de {len(tm)}) | origem "
                  + (f"{to.loc[z, 'atrasados_pct']:.1f}% (posicao {r_o} de {len(to)})" if r_o else "fora (< 5 mil)"))
amp_m = tm["atrasados_pct"].max() - tm["atrasados_pct"].min()
amp_o = to["atrasados_pct"].max() - to["atrasados_pct"].min()
lin = sem.groupby("letreiro")["atrasado"].mean() * 100
print(f"amplitude entre zonas: ponto medio {amp_m:.1f} pp | origem {amp_o:.1f} pp | entre as 10 linhas (sem terminal) {lin.max() - lin.min():.1f} pp"
      f" ({lin.min():.1f}% a {lin.max():.1f}%)")
print(f"desvio padrao entre zonas: ponto medio {tm['atrasados_pct'].std():.2f} pp | origem {to['atrasados_pct'].std():.2f} pp | entre linhas {lin.std():.2f} pp")
for zz in ("Sé", "Praça João Mendes", "Centro Cultural", "Vila Ré", "Capão Redondo"):
    print(f"   {zz}: ponto medio {tm['atrasados_pct'].get(zz, np.nan):.1f}% | origem {to['atrasados_pct'].get(zz, np.nan):.1f}%")
print("###ZONAS_ORIGEM")
print(tm.join(to, lsuffix="_mid", rsuffix="_ori", how="outer").reset_index().rename(columns={"index": "zona"}).round(3).to_csv(index=False))
print("###FIM")

# ================================================================ (2) bootstrap por dia, linha 5300
print("\n== (2) bootstrap por dia (2000 reamostras de dias com reposicao), linha 5300, sem terminal")
l53 = sem[sem["letreiro"].astype(str).str.startswith("5300")]
print(f"passagens da 5300 sem terminal: {len(l53):,} | letreiros: {sorted(l53['letreiro'].astype(str).unique())}")
dias = np.array(sorted(pc["dia"].unique()))


def boot(col, za, zb, B=2000):
    d = l53[l53[col].isin([za, zb])]
    t = d.groupby(["dia", col])["atrasado"].agg(["sum", "size"]).unstack(col).reindex(dias).fillna(0)
    ka, na = t[("sum", za)].to_numpy(), t[("size", za)].to_numpy()
    kb, nb = t[("sum", zb)].to_numpy(), t[("size", zb)].to_numpy()
    obs = 100 * (ka.sum() / na.sum() - kb.sum() / nb.sum())
    idx = rng.integers(0, len(dias), size=(B, len(dias)))
    da = ka[idx].sum(1) / na[idx].sum(1)
    db = kb[idx].sum(1) / nb[idx].sum(1)
    dif = 100 * (da - db)
    lo, hi = np.nanpercentile(dif, [2.5, 97.5])
    dd = 100 * (ka / np.where(na > 0, na, np.nan) - kb / np.where(nb > 0, nb, np.nan))
    print(f"   [{col}] {za}: {100 * ka.sum() / na.sum():.1f}% ({int(na.sum()):,} passagens, {int((na > 0).sum())} dias) | {zb}: "
          f"{100 * kb.sum() / nb.sum():.1f}% ({int(nb.sum()):,}, {int((nb > 0).sum())} dias) | diferenca {obs:.1f} pp | IC 95% [{lo:.1f}; {hi:.1f}] pp"
          f" | dias em que {za} > {zb}: {int(np.nansum(dd > 0))} de {int(np.sum(~np.isnan(dd)))} | razao {ka.sum() / na.sum() / (kb.sum() / nb.sum()):.2f}")
    return obs, lo, hi


boot("zona_mid", "Praça João Mendes", "Centro Cultural")
boot("zona_mid", "Sé", "Centro Cultural")
boot("zona_ori", "Praça João Mendes", "Centro Cultural")
boot("zona_ori", "Sé", "Centro Cultural")
# so dias uteis, para descartar efeito de mistura de tipo de dia
l53u = l53
l53 = l53u[l53u["tipo"] == "util"]
print("   so dias uteis:")
boot("zona_mid", "Praça João Mendes", "Centro Cultural")
l53 = l53u
# controle por sentido e hora: taxa padronizada (mesma mistura de sentido x tipo x hora nas duas zonas)
d = l53[l53["zona_mid"].isin(["Praça João Mendes", "Centro Cultural"])]
c = d.groupby(["sentido", "tipo", "hora", "zona_mid"])["atrasado"].agg(["mean", "size"]).unstack("zona_mid").dropna()
w = c[("size", "Praça João Mendes")] + c[("size", "Centro Cultural")]
print(f"   padronizado por sentido x tipo de dia x hora ({len(c)} celulas com as duas zonas): Joao Mendes "
      f"{100 * np.average(c[('mean', 'Praça João Mendes')], weights=w):.1f}% x Centro Cultural {100 * np.average(c[('mean', 'Centro Cultural')], weights=w):.1f}%")
print("   por sentido:")
print((d.groupby(["sentido", "zona_mid"])["atrasado"].agg(["size", "mean"]).assign(mean=lambda t: 100 * t["mean"])).round(1).to_string())

# ================================================================ (3) decomposicao zona x linha
print("\n== (3) decomposicao zona x linha (sem terminal)")
for col in ("zona_mid", "zona_ori"):
    cel = sem.dropna(subset=[col]).groupby([col, "letreiro"])["atrasado"].agg(passagens="size", pct=lambda s: 100 * s.mean()).reset_index()
    cel = cel[cel["passagens"] >= 1000]
    nz = cel.groupby(col)["letreiro"].transform("size")
    multi = cel[nz >= 2]
    sd_lin_na_zona = multi.groupby(col)["pct"].std()
    amp_lin_na_zona = multi.groupby(col)["pct"].agg(lambda s: s.max() - s.min())
    sd_zona_na_lin = cel.groupby("letreiro")["pct"].std()
    sd_zona_na_lin_m = multi.groupby("letreiro")["pct"].std()
    print(f"[{col}] celulas (zona, linha) com >= 1000 passagens: {len(cel)} | zonas {cel[col].nunique()} | zonas com 2+ linhas: "
          f"{multi[col].nunique()} ({len(multi)} celulas, {100 * multi['passagens'].sum() / cel['passagens'].sum():.0f}% das passagens das celulas)")
    print(f"   entre LINHAS dentro da mesma zona: DP mediano {sd_lin_na_zona.median():.2f} pp | medio {sd_lin_na_zona.mean():.2f} pp | "
          f"amplitude mediana {amp_lin_na_zona.median():.2f} pp")
    print(f"   entre ZONAS dentro da mesma linha (todas as celulas): DP mediano {sd_zona_na_lin.median():.2f} pp | medio {sd_zona_na_lin.mean():.2f} pp"
          f" | (so zonas com 2+ linhas) mediano {sd_zona_na_lin_m.median():.2f} pp")
    # pooled: desvio das celulas em relacao a media da zona x em relacao a media da linha (ponderado por passagens)
    w = multi["passagens"]
    mz = multi.groupby(col)["pct"].transform(lambda s: s.mean())
    ml = multi.groupby("letreiro")["pct"].transform(lambda s: s.mean())
    print(f"   nas celulas de zonas com 2+ linhas: DP residual tirando a zona {np.sqrt(np.average((multi['pct'] - mz) ** 2, weights=w)):.2f} pp"
          f" | tirando a linha {np.sqrt(np.average((multi['pct'] - ml) ** 2, weights=w)):.2f} pp")
    if col == "zona_mid":
        print("   por linha: zonas (celulas), DP e amplitude entre zonas (pp)")
        print(cel.groupby("letreiro")["pct"].agg(zonas="size", dp="std", minimo="min", maximo="max").assign(
            amplitude=lambda t: t["maximo"] - t["minimo"]).round(1).to_string())
        print("   zonas com 3+ linhas: taxa de cada linha (pp)")
        z3 = multi[multi.groupby(col)["letreiro"].transform("size") >= 3]
        for z, g in z3.groupby(col):
            print(f"      {z}: " + ", ".join(f"{r.letreiro} {r.pct:.1f}% ({r.passagens:,})" for r in g.itertuples()) + f" | amplitude {g['pct'].max() - g['pct'].min():.1f} pp")
        print("###CELULAS_ZONA_LINHA")
        print(cel.round(3).to_csv(index=False))
        print("###FIM")


def irls(X, y, it=30, ridge=1e-6):
    b = np.zeros(X.shape[1])
    ll_old = -np.inf
    for _ in range(it):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * b
        b = b + np.linalg.solve(H, g)
        eta = X @ b
        ll = float(np.sum(y * eta - np.logaddexp(0, eta)))
        if abs(ll - ll_old) < 1e-6 * abs(ll):
            break
        ll_old = ll
    return -2 * ll, X.shape[1]


def dummies(s, ref=None):
    s = s.astype(str)
    lv = sorted(s.unique())
    ref = ref if ref in lv else lv[0]
    lv = [v for v in lv if v != ref]
    codes = pd.Categorical(s, categories=lv).codes
    M = np.zeros((len(s), len(lv)), dtype=np.float64)
    ok = codes >= 0
    M[np.where(ok)[0], codes[ok]] = 1.0
    return M


print("\n== (3b) regressao logistica atrasado ~ tipo x hora + linha + zona (IRLS, amostra de 500 mil passagens sem terminal)")
for col in ("zona_mid", "zona_ori"):
    d = sem.dropna(subset=[col])
    cnt = d[col].value_counts()
    d = d.assign(z=np.where(d[col].map(cnt) >= 5000, d[col], "outras (< 5 mil)"))
    am = d.sample(n=min(500_000, len(d)), random_state=7)
    y = am["atrasado"].to_numpy(dtype=float)
    th = am["tipo"].astype(str) + "_" + am["hora"].astype(str)
    Xb = np.column_stack([np.ones(len(am)), dummies(th)])
    Xl = dummies(am["letreiro"])
    Xz = dummies(am["z"])
    p0 = y.mean()
    D_nulo = -2 * float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
    D_b, k_b = irls(Xb, y)
    D_bl, k_bl = irls(np.column_stack([Xb, Xl]), y)
    D_bz, k_bz = irls(np.column_stack([Xb, Xz]), y)
    D_blz, k_blz = irls(np.column_stack([Xb, Xl, Xz]), y)
    print(f"[{col}] amostra {len(am):,} | linhas {Xl.shape[1] + 1} | zonas {Xz.shape[1] + 1} (com 'outras') | deviance nula {D_nulo:,.0f}")
    print(f"   so tipo x hora: queda {D_nulo - D_b:,.0f} ({k_b - 1} gl)")
    print(f"   + linha (depois de tipo x hora): queda {D_b - D_bl:,.0f} ({k_bl - k_b} gl)")
    print(f"   + zona  (depois de tipo x hora): queda {D_b - D_bz:,.0f} ({k_bz - k_b} gl)")
    print(f"   ZONA depois de linha: queda {D_bl - D_blz:,.0f} ({k_blz - k_bl} gl, {(D_bl - D_blz) / (k_blz - k_bl):,.1f} por gl)")
    print(f"   LINHA depois de zona: queda {D_bz - D_blz:,.0f} ({k_blz - k_bz} gl, {(D_bz - D_blz) / (k_blz - k_bz):,.1f} por gl)")
    print(f"   razao (zona apos linha)/(linha apos zona): {(D_bl - D_blz) / max(D_bz - D_blz, 1e-9):.1f}x | pseudo-R2 McFadden: tipo x hora "
          f"{1 - D_b / D_nulo:.4f} | +linha {1 - D_bl / D_nulo:.4f} | +zona {1 - D_bz / D_nulo:.4f} | completo {1 - D_blz / D_nulo:.4f}")
    print(f"   AIC: base+linha {D_bl + 2 * k_bl:,.0f} | base+zona {D_bz + 2 * k_bz:,.0f} | completo {D_blz + 2 * k_blz:,.0f}")
    del Xb, Xl, Xz

# ================================================================ (4) taxa da zona x tempo normal do trecho
print("\n== (4) taxa da zona x tempo normal mediano do trecho (sem terminal, zonas >= 5 mil, ponto medio)")
tz = sem.dropna(subset=["zona_mid"]).groupby("zona_mid").agg(
    passagens=("atrasado", "size"), atrasados_pct=("atrasado", lambda s: 100 * s.mean()),
    normal_med_s=("percurso_tipico_s", "median"), tempo_med_s=("tempo_percurso_s", "median"))
tz = tz[tz["passagens"] >= 5000]
print(f"zonas {len(tz)} | Spearman taxa x tempo normal mediano {spearman(tz['atrasados_pct'], tz['normal_med_s']):.3f} | Pearson "
      f"{np.corrcoef(tz['atrasados_pct'], tz['normal_med_s'])[0, 1]:.3f}")
print(f"tempo normal mediano por zona: de {tz['normal_med_s'].min():.0f} s a {tz['normal_med_s'].max():.0f} s (mediana {tz['normal_med_s'].median():.0f} s)")
q = pd.qcut(tz["normal_med_s"], 4, labels=["1o quartil (trechos curtos)", "2o", "3o", "4o (longos)"])
print(tz.groupby(q, observed=True).agg(zonas=("atrasados_pct", "size"), normal_med_s=("normal_med_s", "median"),
                                       atrasados_pct=("atrasados_pct", "mean")).round(1).to_string())
for zz in ("Sé", "Praça João Mendes", "Centro Cultural", "Vila Ré", "Capão Redondo", "Vila Cordeiro", "Pamplona", "Artur Alvim"):
    if zz in tz.index:
        print(f"   {zz}: {tz.loc[zz, 'atrasados_pct']:.1f}% | tempo normal mediano {tz.loc[zz, 'normal_med_s']:.0f} s")
# nivel do trecho: taxa x tempo normal (o trecho curto atrasa mais?)
tr = sem.groupby(["cl", "sentido", "cp_origem", "cp_destino"]).agg(passagens=("atrasado", "size"), pct=("atrasado", lambda s: 100 * s.mean()),
                                                                   normal_s=("percurso_tipico_s", "median"))
tr = tr[tr["passagens"] >= 300]
print(f"nivel do trecho (>= 300 passagens, {len(tr)} trechos): Spearman taxa x tempo normal {spearman(tr['pct'], tr['normal_s']):.3f}")
qt = pd.qcut(tr["normal_s"], 5)
print(tr.groupby(qt, observed=True).agg(trechos=("pct", "size"), normal_s=("normal_s", "median"), pct=("pct", "mean")).round(1).to_string())
# a zona ainda pesa depois de descontar o tempo normal? residuo da taxa do trecho em funcao do quintil de tempo normal
tr = tr.reset_index().merge(mid[["cp_destino", "zona"]], on="cp_destino", how="left")
tr["q"] = pd.qcut(tr["normal_s"], 10, labels=False)
tr["res"] = tr["pct"] - tr.groupby("q")["pct"].transform("mean")
zr = tr.dropna(subset=["zona"]).groupby("zona").agg(trechos=("res", "size"), res=("res", "mean"), pct=("pct", "mean"))
zr = zr[zr["trechos"] >= 5]
print(f"zonas com >= 5 trechos: {len(zr)} | DP entre zonas da taxa media dos trechos {zr['pct'].std():.2f} pp | depois de descontar o decil de tempo "
      f"normal {zr['res'].std():.2f} pp | Spearman entre os dois rankings {spearman(zr['pct'], zr['res']):.3f}")
print("###ZONAS_TEMPO_NORMAL")
print(tz.reset_index().round(2).to_csv(index=False))
print("###FIM")
print("fim")
