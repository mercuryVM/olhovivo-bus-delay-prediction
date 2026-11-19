"""Tatuape sem depender do normal: velocidade mediana por hora, dia util x domingo/feriado. So leitura.

A. 3459 e 4310 (percursos.parquet, velocidade_kmh) nos trechos que CHEGAM as paradas de Tatuape.
B. Frota de outras linhas (frota/dt=*, cl_medidor fora das 10 linhas), vel_ms*3,6 entre 0,5 e 90:
   B1 nos mesmos pares de paradas (cp_origem, cp_destino) de A; B2 em qualquer trecho que chega a essas paradas.
C. A barra das 5h: % atrasadas com o normal oficial (faixa) x com um normal da propria hora, e velocidade 4h-7h.
Sabado fica de fora da comparacao principal (mostrado so para referencia).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)
RAIZ = Path("/dados")
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}
FAIXAS = ([-1, 5, 8, 15, 19, 23], ["madrugada", "pico manha", "entrepico", "pico tarde", "noite"])
# paradas de Tatuape (TATUAPE_PARADAS do tatuape_ep_cluster.txt)
PRINC = {8209765: "R. Melo Freire, 3216", 8209782: "Parada Metro Tatuape", 820012501: "Metro Carrao - Parada 1"}
EXTRA = {820005545: "Parada 1 - Metro Tatuape", 8210289: "Parada Exclusiva 4310-10"}
CPS = {**PRINC, **EXTRA}
LINHAS = ("3459", "4310")


def bloco(nome, df):
    print(f"###{nome}")
    print(df.to_csv(index=False))
    print("###FIM")


def tipo_dia(ts):
    loc = ts.dt.tz_convert(FUSO)
    dia = loc.dt.strftime("%Y-%m-%d")
    dow = loc.dt.dayofweek
    tipo = np.where(dia.isin(FERIADOS) | (dow == 6), "domingo/feriado", np.where(dow == 5, "sabado", "util"))
    return dia, loc.dt.hour, tipo


def comparar(df, col, nome, chave):
    """Mediana por hora (util x dom/fer) e diferenca pareada por trecho (mesmo trecho, mesma hora)."""
    d = df[df["tipo"].isin(["util", "domingo/feriado"])]
    h = d.groupby(["hora", "tipo"]).agg(med=(col, "median"), n=(col, "size"), dias=("dia", "nunique")).unstack("tipo")
    h.columns = [f"{a}_{'dom' if b != 'util' else 'util'}" for a, b in h.columns]
    h = h.reset_index()
    # pareado: mediana por trecho x hora x tipo; diferenca util - dom no mesmo trecho; media ponderada pelas passagens uteis
    t = d.groupby(chave + ["hora", "tipo"])[col].agg(["median", "size"]).unstack("tipo")
    t = t.dropna()
    t = t[(t[("size", "util")] >= 10) & (t[("size", "domingo/feriado")] >= 5)]
    t["dif"] = t[("median", "util")] - t[("median", "domingo/feriado")]
    t["rel"] = t[("median", "util")] / t[("median", "domingo/feriado")] - 1
    w = t[("size", "util")]
    p = t.groupby(level="hora").apply(lambda g: pd.Series({
        "trechos_par": len(g),
        "dif_par_kmh": np.average(g["dif"], weights=w.loc[g.index]),
        "rel_par_pct": 100 * np.average(g["rel"], weights=w.loc[g.index])})).reset_index()
    h = h.merge(p, on="hora", how="left")
    h["dif_kmh"] = h["med_util"] - h["med_dom"]
    h["dif_pct"] = 100 * (h["med_util"] / h["med_dom"] - 1)
    print(f"\n== {nome}")
    print(h.round(2).to_string(index=False))
    for a, b in ((5, 11), (15, 22), (0, 23)):
        s = h[h["hora"].between(a, b) & (h["n_util"] >= 30) & (h["n_dom"] >= 15)]
        # ponderado pelas passagens de dia util
        wpar = s.dropna(subset=["dif_par_kmh"])
        if len(s) == 0 or len(wpar) == 0:
            print(f"   {a}h-{b}h: sem dados suficientes")
            continue
        print(f"   {a}h-{b}h (horas com dados: {len(s)}): media das horas util {np.average(s['med_util'], weights=s['n_util']):.1f} km/h"
              f" x dom/fer {np.average(s['med_dom'], weights=s['n_util']):.1f} km/h | dif {np.average(s['dif_kmh'], weights=s['n_util']):.1f} km/h"
              f" ({np.average(s['dif_pct'], weights=s['n_util']):.1f}%) | pareado por trecho "
              f"{np.average(wpar['dif_par_kmh'], weights=wpar['n_util']):.1f} km/h ({np.average(wpar['rel_par_pct'], weights=wpar['n_util']):.1f}%)"
              f" | horas com util mais lento: {int((s['dif_kmh'] < 0).sum())} de {len(s)}")
    lento = h[(h["n_util"] >= 30) & (h["n_dom"] >= 15) & (h["dif_pct"] <= -5)]["hora"].tolist()
    print(f"   horas com util pelo menos 5% mais lento que dom/fer: {lento}")
    bloco(nome.split()[0], h.round(3))
    return h


# ================================================================== A. 3459 e 4310
pc = pd.read_parquet(RAIZ / "derivado" / "percursos.parquet", columns=[
    "cl", "letreiro", "sentido", "cp_origem", "cp_destino", "t_origem", "tempo_percurso_s", "percurso_tipico_s",
    "atrasado", "velocidade_kmh"]).dropna(subset=["tempo_percurso_s"])
pc["cp_origem"] = pc["cp_origem"].astype("int64")
pc["cp_destino"] = pc["cp_destino"].astype("int64")
pc["t_origem"] = pd.to_datetime(pc["t_origem"], utc=True)
pc["dia"], pc["hora"], pc["tipo"] = tipo_dia(pc["t_origem"])
cls = set(pc["cl"].dropna().astype("int64").unique())
print(f"passagens (10 linhas) {len(pc):,} | cl do estudo {len(cls)} | datas {pc['dia'].nunique()}")
print("dias por tipo:", pc.groupby("tipo")["dia"].nunique().to_dict())
pc["letreiro"] = pc["letreiro"].astype(str)
alvo = pc[pc["cp_destino"].isin(CPS) & pc["letreiro"].str.contains("|".join(LINHAS))].dropna(subset=["velocidade_kmh"]).copy()
alvo = alvo[(alvo["velocidade_kmh"] > 0.5) & (alvo["velocidade_kmh"] < 90)]
print("\ntrechos que chegam as paradas de Tatuape (3459 e 4310):")
print(alvo.groupby(["letreiro", "sentido", "cp_origem", "cp_destino"]).agg(
    passagens=("velocidade_kmh", "size"), vel_med=("velocidade_kmh", "median"),
    tempo_med_s=("tempo_percurso_s", "median")).reset_index().assign(
    parada=lambda x: x["cp_destino"].map(CPS)).round(1).to_string(index=False))
print("letreiros encontrados:", sorted(alvo["letreiro"].unique()))
ap = alvo[alvo["cp_destino"].isin(PRINC)]
hA = comparar(ap, "velocidade_kmh", "LINHAS_3459_4310 (3 paradas principais)", ["cl", "cp_origem", "cp_destino"])
hA5 = comparar(alvo, "velocidade_kmh", "LINHAS_5PARADAS (com as 2 extras)", ["cl", "cp_origem", "cp_destino"])
for cp, nm in PRINC.items():
    comparar(ap[ap["cp_destino"] == cp], "velocidade_kmh", f"PARADA_{cp} ({nm})", ["cl", "cp_origem", "cp_destino"])

# ================================================================== B. frota de outras linhas
pares = ap[["cp_origem", "cp_destino"]].drop_duplicates()
arqs = sorted((RAIZ / "derivado" / "frota").glob("dt=*.parquet"))
print(f"\narquivos da frota: {len(arqs)} | colunas: {pq.read_schema(arqs[0]).names}")
partes = []
for a in arqs:
    m = pd.read_parquet(a, columns=["t", "vel_ms", "cl_medidor", "cp_origem", "cp_destino"]).dropna(
        subset=["cp_origem", "cp_destino", "vel_ms"])
    m = m[m["cp_destino"].astype("int64").isin(CPS)]
    m["cp_origem"] = m["cp_origem"].astype("int64")
    m["cp_destino"] = m["cp_destino"].astype("int64")
    m = m[~m["cl_medidor"].astype("int64").isin(cls)]
    partes.append(m)
fr = pd.concat(partes, ignore_index=True)
fr["t"] = pd.to_datetime(fr["t"], utc=True)
fr["kmh"] = fr["vel_ms"] * 3.6
fr = fr[(fr["kmh"] > 0.5) & (fr["kmh"] < 90)].copy()
fr["dia"], fr["hora"], fr["tipo"] = tipo_dia(fr["t"])
print(f"medidas da frota (outras linhas) chegando as 5 paradas: {len(fr):,} | linhas (cl_medidor) {fr['cl_medidor'].nunique()} | datas {fr['dia'].nunique()}")
print(fr.groupby("cp_destino").agg(medidas=("kmh", "size"), linhas=("cl_medidor", "nunique"),
                                   origens=("cp_origem", "nunique"), vel_med=("kmh", "median")).round(1).to_string())
f1 = fr.merge(pares, on=["cp_origem", "cp_destino"])
print(f"B1 mesmos pares de paradas: {len(f1):,} medidas | linhas {f1['cl_medidor'].nunique()}")
if len(f1) >= 500:
    comparar(f1, "kmh", "FROTA_MESMOS_PARES (outras linhas, mesmos trechos)", ["cp_origem", "cp_destino"])
f2 = fr[fr["cp_destino"].isin(PRINC)]
hB = comparar(f2, "kmh", "FROTA_CHEGANDO_3PARADAS (outras linhas, qualquer origem)", ["cp_origem", "cp_destino"])
comparar(fr, "kmh", "FROTA_CHEGANDO_5PARADAS (outras linhas, qualquer origem)", ["cp_origem", "cp_destino"])

# ================================================================== C. a barra das 5h
print("\n== C. a barra das 5h (dia util, 3459 e 4310 chegando as 3 paradas)")
u = ap[ap["tipo"] == "util"].copy()
u["ratio_oficial"] = u["tempo_percurso_s"] / u["percurso_tipico_s"]
# normal da propria hora: mediana do trecho, sentido e hora (todos os tipos de dia, como o oficial nao separa tipo?)
ap2 = ap.copy()
ap2["normal_hora"] = ap2.groupby(["cl", "cp_origem", "cp_destino", "hora"])["tempo_percurso_s"].transform("median")
ap2["normal_hora_tipo"] = ap2.groupby(["cl", "cp_origem", "cp_destino", "tipo", "hora"])["tempo_percurso_s"].transform("median")
ap2["faixa"] = pd.cut(ap2["hora"], FAIXAS[0], labels=FAIXAS[1])
ap2["normal_faixa_rec"] = ap2.groupby(["cl", "cp_origem", "cp_destino", "faixa"], observed=True)["tempo_percurso_s"].transform("median")
uu = ap2[ap2["tipo"] == "util"]
c = uu.groupby("hora").agg(
    passagens=("atrasado", "size"),
    atras_oficial=("atrasado", lambda s: 100 * s.mean()),
    atras_faixa_recalc=("tempo_percurso_s", lambda s: 100 * (s >= 1.2 * uu.loc[s.index, "normal_faixa_rec"]).mean()),
    atras_normal_hora=("tempo_percurso_s", lambda s: 100 * (s >= 1.2 * uu.loc[s.index, "normal_hora"]).mean()),
    atras_normal_hora_util=("tempo_percurso_s", lambda s: 100 * (s >= 1.2 * uu.loc[s.index, "normal_hora_tipo"]).mean()),
    tipico_oficial_s=("percurso_tipico_s", "median"), tempo_med_s=("tempo_percurso_s", "median"),
    vel_med=("velocidade_kmh", "median")).reset_index()
print(c.round(1).to_string(index=False))
bloco("BARRA_5H", c.round(3))
# toda a base (10 linhas), para comparar a 5h com o resto
pc["faixa"] = pd.cut(pc["hora"], FAIXAS[0], labels=FAIXAS[1])
pu = pc[pc["tipo"] == "util"]
print("\n   10 linhas, dia util: % atrasadas (oficial) por hora 3h-9h e 18h-22h")
print(pu[pu["hora"].isin([3, 4, 5, 6, 7, 8, 9, 18, 19, 20, 21, 22])].groupby("hora").agg(
    passagens=("atrasado", "size"), atras=("atrasado", lambda s: 100 * s.mean()),
    vel=("velocidade_kmh", "median")).round(1).to_string())
print("\n   peso da 5h na faixa madrugada (passagens, todos os dias):")
print(pc[pc["faixa"] == "madrugada"].groupby("hora").size().to_string())
print("fim")
