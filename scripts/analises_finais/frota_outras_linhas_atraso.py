"""Prova independente: a velocidade de OUTROS onibus no trecho explica o atraso das 10 linhas? So leitura.

Atraso: percursos.parquet (10 linhas). Velocidade: /dados/derivado/frota (frota inteira), tirando as 10 linhas do
estudo quando o arquivo tem a linha. A velocidade da frota vem dos 15 min ANTERIORES ao bloco de 15 min em que a
passagem comeca, entao nao usa o proprio onibus nem o mesmo instante.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.width", 220)
RAIZ = Path("/dados")
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}
BLOCO = "15min"


def bloco(nome, df):
    print(f"###{nome}")
    print(df.to_csv(index=False))
    print("###FIM")


pc = pd.read_parquet(RAIZ / "derivado" / "percursos.parquet",
                     columns=["cl", "cp_origem", "cp_destino", "t_origem", "tempo_percurso_s", "atrasado"]).dropna(
    subset=["tempo_percurso_s"])
pc["cp_origem"] = pc["cp_origem"].astype("int64")
pc["cp_destino"] = pc["cp_destino"].astype("int64")
pc["t_origem"] = pd.to_datetime(pc["t_origem"], utc=True)
loc = pc["t_origem"].dt.tz_convert(FUSO)
pc["dia"] = loc.dt.strftime("%Y-%m-%d")
pc["hora"] = loc.dt.hour
dow = loc.dt.dayofweek
pc["tipo"] = np.where(pc["dia"].isin(FERIADOS) | (dow == 6), "domingo/feriado", np.where(dow == 5, "sabado", "util"))
pc["bloco"] = pc["t_origem"].dt.floor(BLOCO)
cls = set(pc["cl"].dropna().astype("int64").unique())
pares = pc[["cp_origem", "cp_destino"]].drop_duplicates()
print(f"passagens {len(pc):,} | trechos {len(pares)} | linhas-sentido do estudo (cl) {len(cls)}")

arqs = sorted((RAIZ / "derivado" / "frota").glob("dt=*.parquet"))
esq = pq.read_schema(arqs[0])
print("colunas da frota:", esq.names)
col_linha = next((c for c in ("cl_medidor", "cl", "codigo_linha", "linha", "letreiro") if c in esq.names), None)
print("coluna de linha usada para tirar as 10 linhas:", col_linha)
cols = ["t", "vel_ms", "cp_origem", "cp_destino"] + ([col_linha] if col_linha else [])
partes = []
for a in arqs:
    m = pd.read_parquet(a, columns=cols).dropna(subset=["cp_origem", "cp_destino", "vel_ms"])
    m["cp_origem"] = m["cp_origem"].astype("int64")
    m["cp_destino"] = m["cp_destino"].astype("int64")
    m = m.merge(pares, on=["cp_origem", "cp_destino"])
    if col_linha in ("cl", "cl_medidor"):
        m = m[~m[col_linha].astype("int64").isin(cls)]
    partes.append(m)
fr = pd.concat(partes, ignore_index=True)
fr["t"] = pd.to_datetime(fr["t"], utc=True)
fr["kmh"] = fr["vel_ms"] * 3.6
fr = fr[(fr["kmh"] > 0.5) & (fr["kmh"] < 90)]
print(f"medidas da frota nos trechos (sem as 10 linhas: {col_linha in ('cl', 'cl_medidor')}): {len(fr):,}")

fb = fr.assign(bloco=fr["t"].dt.floor(BLOCO)).groupby(["cp_origem", "cp_destino", "bloco"]).agg(
    frota_kmh=("kmh", "median"), n_frota=("kmh", "size")).reset_index()
fb = fb[fb["n_frota"] >= 3]
fb["bloco"] = fb["bloco"] + pd.Timedelta(BLOCO)  # bloco anterior: passa a valer para o bloco seguinte
x = pc.merge(fb, on=["cp_origem", "cp_destino", "bloco"], how="inner")
print(f"passagens com frota no bloco anterior (>= 3 medidas): {len(x):,} ({100 * len(x) / len(pc):.0f}%)")
# velocidade relativa: frota agora / mediana da frota naquele trecho e hora
x["frota_rel"] = x["frota_kmh"] / x.groupby(["cp_origem", "cp_destino", "hora"])["frota_kmh"].transform("median")

u = x[x["tipo"] == "util"]
h = u.groupby("hora").agg(n=("atrasado", "size"), atr=("atrasado", lambda s: 100 * s.mean()),
                          frota=("frota_kmh", "median")).reset_index()
h = h[h["n"] >= 1000]
print(f"\n[A] 24 horas de dia util ({len(h)} horas): frota x % atrasadas Pearson {h['frota'].corr(h['atr']):.2f}"
      f" Spearman {h['frota'].corr(h['atr'], method='spearman'):.2f}")
hs = h[~h["hora"].isin([5, 20])]
print(f"    sem 5h e 20h: Pearson {hs['frota'].corr(hs['atr']):.2f} Spearman {hs['frota'].corr(hs['atr'], method='spearman'):.2f}")
print(h.round(2).to_string(index=False))

c = u.groupby(["dia", "hora"]).agg(n=("atrasado", "size"), atr=("atrasado", lambda s: 100 * s.mean()),
                                   frota=("frota_kmh", "median")).reset_index()
c = c[c["n"] >= 200]
c["fr_r"] = c["frota"] - c.groupby("hora")["frota"].transform("mean")
c["atr_r"] = c["atr"] - c.groupby("hora")["atr"].transform("mean")
inc = np.polyfit(c["fr_r"], c["atr_r"], 1)[0]
print(f"\n[B] dia x hora ({len(c)} celulas): bruta Pearson {c['frota'].corr(c['atr']):.2f} | dentro da mesma hora Pearson "
      f"{c['fr_r'].corr(c['atr_r']):.2f} Spearman {c['fr_r'].corr(c['atr_r'], method='spearman'):.2f} | "
      f"1 km/h a menos na frota -> {-inc:.2f} pontos a mais de atrasadas")

x["faixa"] = pd.qcut(x["frota_rel"], 5, labels=["1 mais lenta", "2", "3", "4", "5 mais rapida"])
q = x.groupby("faixa", observed=True).agg(n=("atrasado", "size"), atr=("atrasado", lambda s: 100 * s.mean()),
                                          frota_rel=("frota_rel", "median")).reset_index()
print(f"\n[C] passagem a passagem (todos os dias, {len(x):,}): correlacao frota_rel x atrasado "
      f"{x['frota_rel'].corr(x['atrasado'].astype(float)):.3f}")
print("    % atrasadas por quinto da velocidade da frota (relativa ao normal do trecho e hora):")
print(q.round(3).to_string(index=False))
r = q["atr"].iloc[0] / q["atr"].iloc[-1]
print(f"    quinto mais lento x mais rapido: {q['atr'].iloc[0]:.1f}% x {q['atr'].iloc[-1]:.1f}% ({r:.1f} vezes)")
bloco("HORAS_FROTA", h.round(3))
bloco("QUINTOS", q.round(3))
