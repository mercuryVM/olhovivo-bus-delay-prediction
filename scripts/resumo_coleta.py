"""O que a coleta de outubro acrescenta a semana de setembro (resumo descritivo).

    python scripts/resumo_coleta.py

Usa dados_mes/derivado (trechos com as paradas do GTFS, setembro + outubro) e
responde, sem modelo nenhum:

1. Os padroes se repetem um mes depois? (taxa de atraso, p90, velocidade e erro
   da previsao da API por periodo e tipo de dia)
2. Como e a madrugada (1h-4h), que a semana 1 nao coletou?
3. Os lugares criticos sao os mesmos? (correlacao de postos do p90 relativo por
   parada entre setembro e outubro e sobreposicao dos 20% piores)
4. A chuva atrasa? (comparacao dentro do mesmo dia util e hora, com e sem chuva
   na celula do trecho; chuva da Open-Meteo)
5. A velocidade da frota (frota.py) acompanha a velocidade do transito medida
   pela TomTom nos 20 pontos?

Os atrasos aqui usam a referencia da tabela `percursos` (mediana do periodo
todo por trecho, sentido e faixa), adequada para descrever, nao para prever.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

RAIZ = Path(__file__).resolve().parents[1] / "dados_mes"
DERIV = RAIZ / "derivado"
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12"}

pc = pd.read_parquet(DERIV / "percursos.parquet")
local = pd.to_datetime(pc["t_origem"], utc=True).dt.tz_convert(FUSO)
pc["data"] = local.dt.strftime("%Y-%m-%d")
pc["periodo"] = np.where(pc["data"] < "2026-10-01", "set", "out")
dow = local.dt.dayofweek
pc["tipo"] = np.where(pc["data"].isin(FERIADOS) | (dow == 6), "domingo/feriado",
                      np.where(dow == 5, "sabado", "util"))
pc["hora"] = local.dt.hour
pc["atrasado"] = pc["atraso_rel"] >= 0.20


def resumo(g):
    return pd.Series({"trechos": len(g), "atrasados": g["atrasado"].mean(),
                      "p90_s": g["atraso_s"].quantile(0.9),
                      "vel_kmh": g["velocidade_kmh"].median()})


print("1. padroes por periodo e tipo de dia")
t = pc.groupby(["periodo", "tipo"]).apply(resumo, include_groups=False)
print(t.round(3).to_string())
uteis = pc[pc["tipo"] == "util"]
pico = uteis.groupby(["periodo", uteis["hora"].isin([6, 7, 16, 17, 18])]).apply(resumo, include_groups=False)
print("   dias uteis, pico (6h-8h e 16h-19h) x fora:")
print(pico.round(3).to_string())

prev = pd.read_parquet(DERIV / "previsao_realizado.parquet", columns=["ts_coleta", "erro_s", "horizonte_s"])
pl = pd.to_datetime(prev["ts_coleta"], utc=True).dt.tz_convert(FUSO).dt.strftime("%Y-%m-%d")
prev["periodo"] = np.where(pl < "2026-10-01", "set", "out")
prev["faixa"] = pd.cut(prev["horizonte_s"] / 60, [0, 5, 15, 30], labels=["<5 min", "5-15", "15-30"])
api = prev.groupby(["periodo", "faixa"], observed=True)["erro_s"].agg(
    mae=lambda x: x.abs().mean(), mediana="median", atrasa_1min=lambda x: (x >= 60).mean())
print("\n   erro da previsao da API (s; erro > 0 = onibus chega depois do previsto):")
print(api.round(2).to_string())

print("\n2. madrugada (so outubro tem 1h-4h)")
out_util = pc[(pc["periodo"] == "out")]
faixas = pd.cut(out_util["hora"], [-1, 0, 3, 5, 9, 15, 19, 23],
                labels=["0h", "1h-3h", "4h-5h", "6h-9h", "10h-15h", "16h-19h", "20h-23h"])
print(out_util.groupby(faixas, observed=True).apply(resumo, include_groups=False).round(3).to_string())

print("\n3. os lugares criticos sao os mesmos?")
por_parada = (pc.groupby(["cp_origem", "periodo"])
              .agg(n=("atraso_rel", "size"), p90_rel=("atraso_rel", lambda x: x.quantile(0.9)))
              .unstack("periodo"))
ok = (por_parada[("n", "set")] >= 100) & (por_parada[("n", "out")] >= 100)
a, b = por_parada.loc[ok, ("p90_rel", "set")], por_parada.loc[ok, ("p90_rel", "out")]
rho = spearmanr(a, b).statistic
topo_a, topo_b = set(a[a >= a.quantile(0.8)].index), set(b[b >= b.quantile(0.8)].index)
print(f"   {ok.sum()} paradas com >= 100 passagens nos dois periodos | correlacao de postos do p90 "
      f"relativo: {rho:.2f} | dos 20% piores de setembro, {len(topo_a & topo_b) / len(topo_a):.0%} "
      f"continuam entre os 20% piores em outubro")

print("\n4. a chuva atrasa? (dias uteis, mesma hora; celula de ~5 km do trecho)")
chuva = pd.read_parquet(DERIV / "chuva_openmeteo.parquet")
chuva["hora_utc"] = chuva["hora_local"].dt.tz_convert("UTC")
passo = 0.05
pc["lat_g"] = (np.round(pc["lat"] / passo) * passo).round(2).clip(chuva["lat"].min(), chuva["lat"].max())
pc["lon_g"] = (np.round(pc["lon"] / passo) * passo).round(2).clip(chuva["lon"].min(), chuva["lon"].max())
# chuva da hora em curso: o acumulado registrado no fim da hora
pc["hora_utc"] = pd.to_datetime(pc["t_origem"], utc=True).dt.ceil("h")
u = pc[pc["tipo"] == "util"].merge(
    chuva.rename(columns={"lat": "lat_g", "lon": "lon_g"})[["lat_g", "lon_g", "hora_utc", "chuva_mm"]],
    on=["lat_g", "lon_g", "hora_utc"], how="left")
u["classe"] = pd.cut(u["chuva_mm"], [-0.01, 0.0, 1.0, 100], labels=["seco", "fraca (ate 1 mm/h)", "moderada/forte"])
estr = u.groupby(["hora", "classe"], observed=True).agg(
    n=("atrasado", "size"), atrasados=("atrasado", "mean"), vel=("velocidade_kmh", "median")).reset_index()
seco = estr[estr["classe"] == "seco"].set_index("hora")
for c in ("fraca (ate 1 mm/h)", "moderada/forte"):
    x = estr[estr["classe"] == c].set_index("hora")
    x = x[x["n"] >= 300]
    comum = x.index.intersection(seco.index)
    peso = x.loc[comum, "n"]
    dif_atr = np.average(x.loc[comum, "atrasados"] - seco.loc[comum, "atrasados"], weights=peso)
    dif_vel = np.average(x.loc[comum, "vel"] - seco.loc[comum, "vel"], weights=peso)
    print(f"   {c}: {int(peso.sum())} trechos em {len(comum)} horas | trechos atrasados "
          f"{dif_atr * 100:+.1f} pontos | velocidade mediana {dif_vel:+.1f} km/h (contra seco na mesma hora)")

print("\n5. a velocidade da frota acompanha a da TomTom?")
from olhovivo import config, frota  # noqa: E402
from olhovivo.storage import Armazenamento  # noqa: E402
import os  # noqa: E402

os.environ["OLHOVIVO_ARMAZENAMENTO__RAIZ"] = str(RAIZ)
os.environ["MONGO_URI"] = ""
cfg = config.carregar()
arm = Armazenamento(cfg, backends=["parquet"])
pares, pontos = frota.geometria(arm, cfg)
proj = frota._projecao(cfg)
tt_pontos = json.loads((Path(__file__).resolve().parents[2] / "olhovivo-infra" / "transito" / "pontos.json")
                       .read_text(encoding="utf-8"))
px, py = proj.para_xy(np.array([p["lat"] for p in tt_pontos]), np.array([p["lon"] for p in tt_pontos]))
dist, j = cKDTree(np.c_[pontos["x"], pontos["y"]]).query(np.c_[px, py])
par_do_ponto = {p["id"]: int(pontos["par"][k]) for p, k, d in zip(tt_pontos, j, dist) if d < 60}
linhas = []
for arq in sorted((RAIZ / "bruto" / "transito").glob("dt=*/tomtom.jsonl")):
    for l in arq.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r.get("status") == 200 and r["ponto"] in par_do_ponto:
            linhas.append({"ponto": r["ponto"], "t": r["ts"], "tt_vel": r["currentSpeed"],
                           "tt_livre": r["freeFlowSpeed"], "par": par_do_ponto[r["ponto"]]})
tt = pd.DataFrame(linhas)
tt["t"] = pd.to_datetime(tt["t"], utc=True)
med = frota.ler(RAIZ).merge(pares[["par", "cp_origem", "cp_destino"]], on=["cp_origem", "cp_destino"])
med = med[med["par"].isin(tt["par"].unique())].sort_values("t")
med["t"] = pd.to_datetime(med["t"], utc=True)
vel_frota = []
for r in tt.itertuples():
    m = med[(med["par"] == r.par) & (med["t"] > r.t - pd.Timedelta("10min")) & (med["t"] <= r.t)]
    vel_frota.append(m["vel_ms"].median() * 3.6 if len(m) >= 3 else np.nan)
tt["frota_vel"] = vel_frota
v = tt.dropna(subset=["frota_vel"])
rho_abs = spearmanr(v["tt_vel"], v["frota_vel"]).statistic
v = v.assign(tt_rel=v["tt_vel"] / v["tt_livre"],
             frota_rel=v["frota_vel"] / v.groupby("ponto")["frota_vel"].transform("quantile", 0.9))
rho_rel = spearmanr(v["tt_rel"], v["frota_rel"]).statistic
dentro = v.groupby("ponto").apply(lambda g: spearmanr(g["tt_vel"], g["frota_vel"]).statistic
                                  if len(g) > 30 else np.nan, include_groups=False)
print(f"   {len(par_do_ponto)} dos 20 pontos caem sobre trechos estudados | {len(v)} medicoes com frota "
      f"nos 10 min anteriores | correlacao de postos: velocidades {rho_abs:.2f}, relativas ao fluxo "
      f"livre {rho_rel:.2f}, dentro de cada ponto (mediana) {dentro.median():.2f}")
