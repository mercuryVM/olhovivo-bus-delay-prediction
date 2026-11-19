"""Correlacao entre velocidade da hora e % de passagens atrasadas (dia util). So leitura.

1) 24 horas do dia util: velocidade mediana x % atrasadas (Pearson e Spearman), com e sem 5h e 20h (troca de faixa).
2) celulas dia x hora (dias uteis): correlacao bruta e depois de tirar a media de cada hora (variacao dia a dia na
   mesma hora: quando a hora esta mais lenta que o habitual, atrasa mais?).
3) participacao do pico (6-8h e 16-19h) nas passagens e nos atrasos.
"""
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
RAIZ = Path("/dados")
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}


def bloco(nome, df):
    print(f"###{nome}")
    print(df.to_csv(index=False))
    print("###FIM")


pc = pd.read_parquet(RAIZ / "derivado" / "percursos.parquet",
                     columns=["t_origem", "tempo_percurso_s", "atrasado", "velocidade_kmh"]).dropna(subset=["tempo_percurso_s"])
loc = pd.to_datetime(pc["t_origem"], utc=True).dt.tz_convert(FUSO)
pc["dia"] = loc.dt.strftime("%Y-%m-%d")
pc["hora"] = loc.dt.hour
dow = loc.dt.dayofweek
pc["tipo"] = np.where(pc["dia"].isin(FERIADOS) | (dow == 6), "domingo/feriado", np.where(dow == 5, "sabado", "util"))
u = pc[pc["tipo"] == "util"].dropna(subset=["velocidade_kmh"])
print(f"passagens {len(pc):,} | dia util {len(u):,} | dias uteis {u['dia'].nunique()} | atrasadas util {100 * u['atrasado'].mean():.1f}%")


def corr(a, b):
    a, b = pd.Series(a), pd.Series(b)
    return a.corr(b), a.corr(b, method="spearman")


h = u.groupby("hora").agg(passagens=("atrasado", "size"), atrasadas_pct=("atrasado", lambda s: 100 * s.mean()),
                          vel_mediana=("velocidade_kmh", "median")).reset_index()
p, s = corr(h["vel_mediana"], h["atrasadas_pct"])
print(f"\n[1] 24 horas: Pearson {p:.2f} | Spearman {s:.2f}")
hm = h[(h["passagens"] >= 1000)]
p, s = corr(hm["vel_mediana"], hm["atrasadas_pct"])
print(f"    horas com >= 1000 passagens ({len(hm)}): Pearson {p:.2f} | Spearman {s:.2f}")
hs = hm[~hm["hora"].isin([5, 20])]
p, s = corr(hs["vel_mediana"], hs["atrasadas_pct"])
print(f"    sem 5h e 20h ({len(hs)}): Pearson {p:.2f} | Spearman {s:.2f}")
w = np.average(hm["atrasadas_pct"], weights=hm["passagens"])
print(h.round(2).to_string(index=False))

c = u.groupby(["dia", "hora"]).agg(n=("atrasado", "size"), atr=("atrasado", lambda s: 100 * s.mean()),
                                   vel=("velocidade_kmh", "median")).reset_index()
c = c[c["n"] >= 200]
p, s = corr(c["vel"], c["atr"])
print(f"\n[2] celulas dia x hora (n>=200): {len(c)} | bruta Pearson {p:.2f} Spearman {s:.2f}")
c["vel_r"] = c["vel"] - c.groupby("hora")["vel"].transform("mean")
c["atr_r"] = c["atr"] - c.groupby("hora")["atr"].transform("mean")
p, s = corr(c["vel_r"], c["atr_r"])
print(f"    dentro da mesma hora (dia a dia): Pearson {p:.2f} Spearman {s:.2f}")
b = np.polyfit(c["vel_r"], c["atr_r"], 1)[0]
print(f"    inclinacao: cada 1 km/h a menos que o habitual da hora -> {-b:.2f} pontos percentuais a mais de atrasadas")
for hh in (7, 18):
    g = c[c["hora"] == hh]
    if len(g) >= 8:
        p, s = corr(g["vel"], g["atr"])
        print(f"    so {hh}h ({len(g)} dias): Pearson {p:.2f} Spearman {s:.2f} | vel {g['vel'].min():.1f}-{g['vel'].max():.1f} km/h | atrasadas {g['atr'].min():.0f}-{g['atr'].max():.0f}%")

pico = u["hora"].between(6, 8) | u["hora"].between(16, 19)
print(f"\n[3] pico de dia util: {100 * pico.mean():.1f}% das passagens e {100 * u.loc[pico, 'atrasado'].sum() / u['atrasado'].sum():.1f}% dos atrasos"
      f" | atrasadas no pico {100 * u.loc[pico, 'atrasado'].mean():.1f}% x fora {100 * u.loc[~pico, 'atrasado'].mean():.1f}%")
bloco("HORAS", h.round(3))
bloco("CELULAS", c.round(3))
