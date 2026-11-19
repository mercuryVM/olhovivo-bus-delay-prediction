"""Atraso por data (corte 20%) e 2o turno (25/10). So leitura; saida so por print."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
D = Path("/dados/derivado")
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}
rng = np.random.default_rng(42)

pc = pd.read_parquet(D / "percursos.parquet", columns=["cl", "prefixo", "t_origem", "tempo_percurso_s",
                                                       "percurso_tipico_s", "atraso_rel", "atrasado"])
pc = pc.dropna(subset=["tempo_percurso_s", "percurso_tipico_s"]).reset_index(drop=True)
t = pd.to_datetime(pc["t_origem"], utc=True).dt.tz_convert(FUSO)
pc["dia"] = t.dt.strftime("%Y-%m-%d")
pc["hora"] = t.dt.hour
dow = t.dt.dayofweek
pc["tipo"] = np.where(pc["dia"].isin(FERIADOS) | (dow == 6), "domingo/feriado", np.where(dow == 5, "sabado", "util"))
pc["dsem"] = t.dt.day_name().str[:3]
# corte 20% recalculado e conferido com a coluna da tabela
pc["atr20"] = (pc["tempo_percurso_s"] >= 1.2 * pc["percurso_tipico_s"]).astype(int)
conc = (pc["atr20"] == pc["atrasado"].astype(int)).mean()
print(f"passagens {len(pc):,} | dias {pc['dia'].nunique()} | concordancia atr20 x coluna atrasado {100 * conc:.2f}%")
print(f"geral atrasadas (corte 20%): {100 * pc['atr20'].mean():.2f}% | coluna atrasado {100 * pc['atrasado'].mean():.2f}%")

por = pc.groupby(["dia", "dsem", "tipo"]).agg(passagens=("atr20", "size"), atrasadas_pct=("atr20", lambda s: 100 * s.mean()),
                                              onibus=("prefixo", "nunique")).reset_index()
print("\n== 1. % atrasadas por data")
print(por.round(2).to_string(index=False))


def ic(v, n=10000):
    v = np.asarray(v, float)
    b = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


print("\n== 2. resumo por tipo (media entre datas, IC 95% bootstrap entre datas)")
for tp in ("util", "sabado", "domingo/feriado"):
    g = por[por["tipo"] == tp]
    if len(g) == 0:
        continue
    m, lo, hi = ic(g["atrasadas_pct"])
    print(f"  {tp:16s} n_datas={len(g):2d} media {m:.2f}% IC95 [{lo:.2f}; {hi:.2f}] | mediana {g['atrasadas_pct'].median():.2f}"
          f" | min {g['atrasadas_pct'].min():.2f} ({g.loc[g['atrasadas_pct'].idxmin(), 'dia']})"
          f" | max {g['atrasadas_pct'].max():.2f} ({g.loc[g['atrasadas_pct'].idxmax(), 'dia']})")
    print(f"     % agregado (todas as passagens do tipo): {100 * pc.loc[pc['tipo'] == tp, 'atr20'].mean():.2f}%")
dfer = por[por["tipo"] == "domingo/feriado"]
sem = dfer[dfer["dia"] != "2026-10-25"]
m, lo, hi = ic(sem["atrasadas_pct"])
print(f"  domingo/feriado SEM 25/10: n_datas={len(sem)} media {m:.2f}% IC95 [{lo:.2f}; {hi:.2f}] | max {sem['atrasadas_pct'].max():.2f}"
      f" ({sem.loc[sem['atrasadas_pct'].idxmax(), 'dia']})")
print(f"     % agregado sem 25/10: {100 * pc.loc[(pc['tipo'] == 'domingo/feriado') & (pc['dia'] != '2026-10-25'), 'atr20'].mean():.2f}%")
ut = por[por["tipo"] == "util"]
d = ut["atrasadas_pct"].mean() - dfer["atrasadas_pct"].mean()
bu = rng.choice(ut["atrasadas_pct"].to_numpy(), (10000, len(ut))).mean(1)
bd = rng.choice(dfer["atrasadas_pct"].to_numpy(), (10000, len(dfer))).mean(1)
print(f"  diferenca util - dom/fer: {d:.2f} p.p. IC95 [{np.percentile(bu - bd, 2.5):.2f}; {np.percentile(bu - bd, 97.5):.2f}]")
print("  ranking domingo/feriado:")
print(dfer.sort_values("atrasadas_pct", ascending=False).round(2).to_string(index=False))
print("  datas uteis ordenadas:")
print(ut.sort_values("atrasadas_pct").round(2).to_string(index=False))

print("\n== 3. 2o turno: % atrasadas 9-12h e 12-15h")
pc["janela"] = np.select([pc["hora"].between(9, 11), pc["hora"].between(12, 14)], ["9-12h", "12-15h"], "outra")
j = pc[pc["janela"] != "outra"]
jd = j.groupby(["dia", "tipo", "janela"]).agg(passagens=("atr20", "size"), atr=("atr20", lambda s: 100 * s.mean())).reset_index()
tab = jd[jd["tipo"] == "domingo/feriado"].pivot(index="dia", columns="janela", values=["atr", "passagens"])
print(tab.round(2).to_string())
ju = jd[jd["tipo"] == "util"]
for jn in ("9-12h", "12-15h"):
    g = ju[ju["janela"] == jn]["atr"]
    m, lo, hi = ic(g)
    print(f"  dia util medio {jn}: {m:.2f}% IC95 [{lo:.2f}; {hi:.2f}] (n_datas {len(g)}) | agregado "
          f"{100 * j.loc[(j['tipo'] == 'util') & (j['janela'] == jn), 'atr20'].mean():.2f}%")
    gd = jd[(jd["tipo"] == "domingo/feriado") & (jd["janela"] == jn)].sort_values("atr", ascending=False)
    top = gd.iloc[0]
    seg = gd.iloc[1]
    print(f"  dom/fer {jn}: maior {top['dia']} {top['atr']:.2f}% | segundo {seg['dia']} {seg['atr']:.2f}% | distancia"
          f" {top['atr'] - seg['atr']:.2f} p.p. | media dos outros {gd.iloc[1:]['atr'].mean():.2f}%")
w = j.copy()
w["j9_15"] = 1
a = pc[pc["hora"].between(9, 14)].groupby(["dia", "tipo"]).agg(passagens=("atr20", "size"), atr=("atr20", lambda s: 100 * s.mean())).reset_index()
print("  janela 9-15h inteira, dom/fer:")
print(a[a["tipo"] == "domingo/feriado"].sort_values("atr", ascending=False).round(2).to_string(index=False))
print(f"  dia util medio 9-15h: {a[a['tipo'] == 'util']['atr'].mean():.2f}%")

print("\n== 4. onibus distintos (prefixo) das 10 linhas por hora")
DIAS = {"2026-10-25": "2o turno", "2026-10-18": "dom comum 18/10", "2026-11-01": "dom comum 01/11"}
uteis = sorted(ut["dia"])
h1 = pc.groupby(["dia", "hora"])["prefixo"].nunique().unstack(0)
out = pd.DataFrame({v: h1.get(k) for k, v in DIAS.items()})
out["util_medio"] = h1[uteis].mean(axis=1)
print("  a) pela tabela percursos (onibus com passagem iniciada na hora):")
print(out.round(1).to_string())
s = out.loc[9:14]
print(f"  9-15h: 25/10 {s['2o turno'].sum():.0f} | 18/10 {s['dom comum 18/10'].sum():.0f} | 01/11 {s['dom comum 01/11'].sum():.0f}"
      f" | util medio {s['util_medio'].sum():.0f} (soma onibus-hora)")
print(f"  razao 25/10 / util medio 9-15h: {s['2o turno'].sum() / s['util_medio'].sum():.2f}; 18/10: "
      f"{s['dom comum 18/10'].sum() / s['util_medio'].sum():.2f}; 01/11: {s['dom comum 01/11'].sum() / s['util_medio'].sum():.2f}")
print(f"  onibus distintos no dia inteiro: " + ", ".join(f"{k} {pc.loc[pc['dia'] == k, 'prefixo'].nunique()}" for k in DIAS)
      + f", util medio {por[por['tipo'] == 'util']['onibus'].mean():.1f}")

cls = set(pc["cl"].dropna().astype("int64").unique())
arqs = sorted((D / "frota").glob("dt=*.parquet"))
print(f"\n  b) pela frota/dt=* ({len(arqs)} arquivos), cl_medidor nas {len(cls)} linhas-sentido")
alvo = set(DIAS) | set(uteis)
linhas = []
for a_ in arqs:
    m = pd.read_parquet(a_, columns=["t", "cl_medidor", "prefixo"]).dropna(subset=["cl_medidor", "prefixo"])
    m = m[m["cl_medidor"].astype("int64").isin(cls)]
    if m.empty:
        continue
    tl = pd.to_datetime(m["t"], utc=True).dt.tz_convert(FUSO)
    m = m.assign(dia=tl.dt.strftime("%Y-%m-%d"), hora=tl.dt.hour)
    m = m[m["dia"].isin(alvo)]
    linhas.append(m[["dia", "hora", "prefixo"]].drop_duplicates())
if linhas:
    f = pd.concat(linhas).drop_duplicates()
    h2 = f.groupby(["dia", "hora"])["prefixo"].nunique().unstack(0)
    o2 = pd.DataFrame({v: h2.get(k) for k, v in DIAS.items()})
    o2["util_medio"] = h2[[u for u in uteis if u in h2.columns]].mean(axis=1)
    print(o2.round(1).to_string())
    s2 = o2.loc[9:14]
    print(f"  razao 25/10 / util medio 9-15h: {s2['2o turno'].sum() / s2['util_medio'].sum():.2f}; 18/10: "
          f"{s2['dom comum 18/10'].sum() / s2['util_medio'].sum():.2f}; 01/11: {s2['dom comum 01/11'].sum() / s2['util_medio'].sum():.2f}")
print("\n###DATAS")
print(por.round(3).to_csv(index=False))
print("###FIM")
