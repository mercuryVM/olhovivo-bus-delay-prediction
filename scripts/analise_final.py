"""Tres analises descritivas para o relatorio final (so leem dados):

1. Viagem real contra a programada. O GTFS da SPTrans traz, para cada linha e
   sentido, uma viagem-modelo cuja duracao total e dividida por igual entre as
   paradas e nao muda com a hora. Compara-se a soma dos trechos de cada viagem
   observada com a duracao programada do mesmo numero de trechos.
2. Calibracao do ST-DBSCAN: grade de raio espacial, janela temporal e minimo de
   pontos sobre a mesma amostra de ocorrencias de atraso, com ruido, tamanho,
   raio e duracao dos episodios e recorrencia das paradas.
3. Zonas Origem-Destino (pesquisa OD 2023, camada do GeoSampa em
   <raiz>/externo/geosampa/zona_origem_destino.geojson): atraso por zona, fluxos
   entre a zona de inicio e a de fim das viagens, e episodios por zona.

    python scripts/analise_final.py            # raiz em OLHOVIVO_ARMAZENAMENTO__RAIZ ou dados_mes
    ANALISE_GRADE=curta python scripts/analise_final.py   # grade pequena, para testar
"""
import csv
import io
import json
import os
import sys
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olhovivo import geo, regioes  # noqa: E402
from olhovivo.config import carregar  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
RAIZ = Path(os.environ.get("OLHOVIVO_ARMAZENAMENTO__RAIZ") or Path(__file__).resolve().parents[1] / "dados_mes")
FUSO = "America/Sao_Paulo"
AMOSTRA = int(os.environ.get("ANALISE_AMOSTRA", "60000"))
CURTA = os.environ.get("ANALISE_GRADE") == "curta"

cfg = carregar()
proj = geo.obter_projecao(int(cfg.get("geo.epsg_metrico", 31983)), float(cfg.get("geo.ancora_lat", -23.5505)),
                          float(cfg.get("geo.ancora_lon", -46.6333)))
pc = pd.read_parquet(RAIZ / "derivado" / "percursos.parquet",
                     columns=["cl", "viagem_id", "cp_origem", "cp_destino", "ordem_origem", "t_origem", "t_chegada",
                              "tempo_percurso_s", "atraso_s", "atraso_rel", "atrasado", "velocidade_kmh",
                              "lat", "lon"]).dropna(subset=["tempo_percurso_s", "atraso_s"])
local = pd.to_datetime(pc["t_origem"], utc=True).dt.tz_convert(FUSO)
pc["dia"] = local.dt.strftime("%Y-%m-%d")
pc["hora"] = local.dt.hour
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}
pc["tipo_dia"] = np.where(pc["dia"].isin(FERIADOS) | (local.dt.dayofweek == 6), "domingo e feriado",
                          np.where(local.dt.dayofweek == 5, "sabado", "dia util"))
pc["faixa"] = pd.cut(pc["hora"], [-1, 5, 8, 15, 19, 23],
                     labels=["madrugada", "pico manha", "entrepico", "pico tarde", "noite"])
print(f"base: {len(pc):,} passagens, {pc['viagem_id'].nunique():,} viagens, {pc['dia'].nunique()} dias")

# ----------------------------------------------------- 1. real x programado
print("\n" + "=" * 30, "1. viagem real contra a programada (GTFS)", "=" * 30)
linhas = pd.read_parquet(RAIZ / "catalogo" / "linhas.parquet", columns=["cl", "letreiro_completo", "sentido"]).dropna()
linhas["trip_id"] = linhas["letreiro_completo"].astype(str) + "-" + (linhas["sentido"].astype(int) - 1).astype(str)
alvo = set(linhas.loc[linhas["cl"].isin(pc["cl"].unique()), "trip_id"])
paradas = {}
with zipfile.ZipFile(RAIZ / "gtfs" / "sptrans-gtfs.zip") as z:
    for r in csv.DictReader(io.TextIOWrapper(z.open("stop_times.txt"), encoding="utf-8-sig")):
        if r["trip_id"] in alvo:
            h, m, s = map(int, r["arrival_time"].split(":"))
            paradas.setdefault(r["trip_id"], []).append(h * 3600 + m * 60 + s)
prog = pd.DataFrame([{"trip_id": t, "n_paradas": len(v), "duracao_prog_s": max(v) - min(v)} for t, v in paradas.items()])
prog["trecho_prog_s"] = prog["duracao_prog_s"] / (prog["n_paradas"] - 1)
prog = prog.merge(linhas[["cl", "trip_id", "letreiro_completo", "sentido"]], on="trip_id")
v = pc.groupby("viagem_id").agg(cl=("cl", "first"), trechos=("tempo_percurso_s", "size"),
                                real_s=("tempo_percurso_s", "sum"), inicio=("t_origem", "min"),
                                tipo_dia=("tipo_dia", "first"), faixa=("faixa", "first")).reset_index()
v = v.merge(prog, on="cl")
v["cobertura"] = v["trechos"] / (v["n_paradas"] - 1)
v = v[v["cobertura"] >= 0.8].copy()
v["prog_s"] = v["trechos"] * v["trecho_prog_s"]
v["razao"] = v["real_s"] / v["prog_s"]
print(f"viagens com ao menos 80% dos trechos: {len(v):,} | duracao programada por viagem (min): "
      f"{(prog['duracao_prog_s'] / 60).round().astype(int).tolist()}")
r = lambda g: pd.Series({"viagens": len(g), "real/programado (mediana)": g["razao"].median(),
                         "acima do programado %": 100 * (g["razao"] > 1).mean(),
                         "20% ou mais acima %": 100 * (g["razao"] >= 1.2).mean(),
                         "real mediano min": g["real_s"].median() / 60, "programado mediano min": g["prog_s"].median() / 60})
print(v.groupby(["tipo_dia", "faixa"], observed=True).apply(r).round(2).to_string())
print("\npor linha e sentido (todos os dias):")
print(v.groupby(["letreiro_completo", "sentido"]).apply(r).round(2).sort_values("real/programado (mediana)").to_string())

# --------------------------------------------------- 2. calibracao ST-DBSCAN
print("\n" + "=" * 30, "2. calibracao do ST-DBSCAN", "=" * 30)
base = pd.DataFrame({"valor_s": pc["atraso_s"].astype("float64"), "evento": pc["atrasado"].astype(bool),
                     "t": pd.to_datetime(pc["t_chegada"], utc=True), "cp": pc["cp_destino"].astype("int64"),
                     "lat": pc["lat"], "lon": pc["lon"], "dia": pc["dia"]}).dropna(subset=["lat", "lon"])
eventos = regioes.preparar_amostras(base, proj, "eventos").reset_index(drop=True)
# Sortear uma amostra do periodo inteiro dilui a densidade e nenhum episodio se forma. Como um
# episodio dura no maximo algumas horas, o agrupamento roda dia a dia, com todos os eventos do dia.
por_dia = {d: g.reset_index(drop=True) for d, g in eventos.groupby("dia")}
DIAS_CAL = [d for d in ("2026-10-18", "2026-10-21", "2026-10-22", "2026-10-24") if d in por_dia]
print(f"ocorrencias de atraso: {len(eventos):,} em {len(por_dia)} dias | calibracao em {DIAS_CAL}")


def tempo_s(a):
    return (pd.to_datetime(a["t"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def expandir(vizinhos, min_pontos):
    """A mesma expansao de regioes.rodar_st_dbscan, sem o delta de valor, reaproveitando a vizinhanca."""
    n = len(vizinhos)
    rot = np.full(n, -2, dtype="int64")
    c = 0
    for i in range(n):
        if rot[i] != -2:
            continue
        if len(vizinhos[i]) < min_pontos:
            rot[i] = -1
            continue
        rot[i] = c
        fila = deque(vizinhos[i])
        while fila:
            j = fila.popleft()
            if rot[j] == -1:
                rot[j] = c
                continue
            if rot[j] != -2:
                continue
            rot[j] = c
            if len(vizinhos[j]) >= min_pontos:
                fila.extend(k for k in vizinhos[j] if rot[k] == -2)
        c += 1
    rot[rot == -2] = -1
    return rot


def medidas(a):
    """`a`: eventos com a coluna `ep` (-1 = ruido), ja com ids unicos entre os dias."""
    e = a[a["ep"] >= 0]
    if e.empty:
        return {"episodios": 0, "ruido %": 100.0, "pontos (mediana)": np.nan, "raio p90 (m, mediana)": np.nan,
                "duracao (min, mediana)": np.nan, "maior episodio": np.nan, "paradas em 5+ dias %": np.nan}
    g = e.groupby("ep")
    cx, cy = g["x_m"].transform("mean"), g["y_m"].transform("mean")
    raio = np.hypot(e["x_m"] - cx, e["y_m"] - cy).groupby(e["ep"]).quantile(0.9)
    dur = (g["t"].max() - g["t"].min()).dt.total_seconds() / 60
    # recorrencia: paradas de episodios que aparecem em episodios de ao menos 5 dias diferentes
    dias_por_parada = e.groupby("cp")["dia"].nunique()
    return {"episodios": int(e["ep"].nunique()), "ruido %": round(100 * (a["ep"] < 0).mean(), 1),
            "pontos (mediana)": int(g.size().median()), "raio p90 (m, mediana)": int(raio.median()),
            "duracao (min, mediana)": int(dur.median()), "maior episodio": int(g.size().max()),
            "paradas em 5+ dias %": round(100 * (dias_por_parada >= 5).mean(), 1)}


grade_s = [400] if CURTA else [200, 400, 800]
grade_t = [1800] if CURTA else [900, 1800, 3600]
grade_m = [15] if CURTA else [5, 10, 15, 30]
def rotular(dias, es, et, mps):
    """Eventos dos `dias` com um rotulo por valor de min_pontos, ids unicos entre os dias."""
    saida = {mp: [] for mp in mps}
    for d in dias:
        a = por_dia[d]
        viz = regioes._vizinhanca_espaco_tempo(a["x_m"].to_numpy(), a["y_m"].to_numpy(), tempo_s(a), es, et)
        for mp in mps:
            r = expandir(viz, mp)
            saida[mp].append(a.assign(ep=np.where(r >= 0, r + 1_000_000 * len(saida[mp]), -1)))
        del viz
    return {mp: pd.concat(v, ignore_index=True) for mp, v in saida.items()}


linhas_grade = []
for es in grade_s:
    for et in grade_t:
        for mp, a in rotular(DIAS_CAL, es, et, grade_m).items():
            linhas_grade.append({"raio m": es, "janela min": et // 60, "min pontos": mp, **medidas(a)})
grade = pd.DataFrame(linhas_grade)
print(grade.to_string(index=False))
# regra de escolha: ruido abaixo de 30%; episodio tipico maior que uma parada (raio p90 mediano de
# ao menos 200 m, metade do espacamento tipico) e menor que um bairro (ate 1 km); duracao mediana ate
# 3 h; e sem encadeamento, isto e, nenhum episodio com mais de 5% das ocorrencias da calibracao
n_cal = sum(len(por_dia[d]) for d in DIAS_CAL)
ok = grade[(grade["ruido %"] < 30) & grade["raio p90 (m, mediana)"].between(200, 1000)
           & (grade["duracao (min, mediana)"] <= 180) & (grade["maior episodio"] <= 0.05 * n_cal)]
print(f"\nocorrencias na calibracao: {n_cal:,}; combinacoes que passam na regra:\n" + ok.to_string(index=False))
# entre as que passam, a de mais pontos minimos (mais robusta) e, empatando, a de menor ruido
escolha = (ok.sort_values(["min pontos", "ruido %"], ascending=[False, True]).iloc[0]
           if len(ok) else grade.sort_values("ruido %").iloc[0])
print("\nescolhida:", escolha.to_dict(), "" if len(ok) else "(nenhuma passou na regra: a de menor ruido)")
mp_esc = int(escolha["min pontos"])
todos = rotular(sorted(por_dia), float(escolha["raio m"]), float(escolha["janela min"]) * 60, [mp_esc])[mp_esc]
for k, val in medidas(todos).items():
    print(f"  todos os dias, {k}: {val}")
parcial = rotular(sorted(por_dia), 400.0, 1800.0, [15])[15]
print("  com os parametros do parcial (400 m, 30 min, 15):", {k: medidas(parcial)[k] for k in ("episodios", "ruido %", "raio p90 (m, mediana)")})
ep = todos[todos["ep"] >= 0]
loc_ep = pd.to_datetime(ep.groupby("ep")["t"].min(), utc=True).dt.tz_convert(FUSO)
pico = loc_ep.dt.hour.isin([6, 7, 8, 16, 17, 18, 19]) & (loc_ep.dt.dayofweek < 5)
print(f"episodios: {ep['ep'].nunique()} | comecam no pico de dia util: {100 * pico.mean():.1f}% "
      f"(passagens no pico de dia util: {100 * (pc['faixa'].isin(['pico manha', 'pico tarde']) & (pc['tipo_dia'] == 'dia util')).mean():.1f}%)")

# ------------------------------------------------------------ 3. zonas OD
print("\n" + "=" * 30, "3. zonas Origem-Destino (OD 2023)", "=" * 30)
arq = RAIZ / "externo" / "geosampa" / "zona_origem_destino.geojson"
zonas = json.loads(arq.read_text(encoding="utf-8"))["features"]


def aneis(g):
    if g["type"] == "Polygon":
        return [np.asarray(r)[:, :2] for r in g["coordinates"]]
    if g["type"] == "MultiPolygon":
        return [np.asarray(r)[:, :2] for p in g["coordinates"] for r in p]
    return []


def dentro(lon, lat, rings):
    """Par-impar sobre todos os aneis (furos incluidos)."""
    res = np.zeros(len(lon), dtype=bool)
    for r in rings:
        x0, y0, x1, y1 = r[:-1, 0], r[:-1, 1], r[1:, 0], r[1:, 1]
        cruza = ((y0[None, :] > lat[:, None]) != (y1[None, :] > lat[:, None])) & (
            lon[:, None] < (x1 - x0)[None, :] * (lat[:, None] - y0[None, :]) / np.where(y1 == y0, 1e-12, y1 - y0)[None, :] + x0[None, :])
        res ^= (cruza.sum(axis=1) % 2 == 1)
    return res


pontos = pc.groupby("cp_destino")[["lat", "lon"]].first().reset_index()
pontos["zona"] = None
for f in zonas:
    rings = aneis(f["geometry"])
    if not rings:
        continue
    todos = np.concatenate(rings)
    caixa = (pontos["lon"].between(todos[:, 0].min(), todos[:, 0].max())
             & pontos["lat"].between(todos[:, 1].min(), todos[:, 1].max()) & pontos["zona"].isna())
    if caixa.any():
        idx = pontos.index[caixa]
        m = dentro(pontos.loc[idx, "lon"].to_numpy(), pontos.loc[idx, "lat"].to_numpy(), rings)
        pontos.loc[idx[m], "zona"] = f["properties"].get("nm_zona")
print(f"paradas: {len(pontos)}, com zona: {pontos['zona'].notna().sum()}, zonas tocadas: {pontos['zona'].nunique()}")
mapa = dict(zip(pontos["cp_destino"], pontos["zona"]))
pc["zona"] = pc["cp_destino"].map(mapa)
z = pc.dropna(subset=["zona"]).groupby("zona").agg(
    passagens=("atraso_s", "size"), atrasados=("atrasado", "mean"),
    p90_rel=("atraso_rel", lambda s: s.quantile(0.9)), vel_kmh=("velocidade_kmh", "median"))
z = z[z["passagens"] >= 5000]
z["atrasados"] *= 100
z["p90_rel"] *= 100
print(f"zonas com ao menos 5 mil passagens: {len(z)} | atrasados de {z['atrasados'].min():.1f}% a {z['atrasados'].max():.1f}%")
print("mais atrasadas:\n" + z.sort_values("atrasados", ascending=False).head(10).round(1).to_string())
print("menos atrasadas:\n" + z.sort_values("atrasados").head(5).round(1).to_string())

viag = pc.sort_values(["viagem_id", "ordem_origem"]).groupby("viagem_id").agg(
    z_origem=("zona", "first"), z_destino=("zona", "last"), trechos=("atraso_s", "size"),
    atraso_acumulado_s=("atraso_s", "sum"))
viag = viag[viag["trechos"] >= 10].dropna()
fl = viag.groupby(["z_origem", "z_destino"]).agg(viagens=("trechos", "size"),
                                                 acumulado_mediano_min=("atraso_acumulado_s", lambda s: s.median() / 60),
                                                 viagens_5min_ou_mais=("atraso_acumulado_s", lambda s: 100 * (s >= 300).mean()))
fl = fl[fl["viagens"] >= 300]
print(f"\nfluxos entre zonas com ao menos 300 viagens: {len(fl)}")
print(fl.sort_values("viagens_5min_ou_mais", ascending=False).round(1).head(12).to_string())

ep_z = ep.assign(zona=ep["cp"].map(mapa)).dropna(subset=["zona"])
cont = ep_z.groupby("ep")["zona"].agg(lambda s: s.mode().iloc[0]).value_counts()
print(f"\nepisodios por zona (zona predominante de cada episodio), {cont.size} zonas:")
print(cont.head(10).to_string())
