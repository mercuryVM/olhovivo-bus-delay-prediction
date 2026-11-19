"""Episodios ST-DBSCAN: (1) onibus distintos por episodio; (2) modelo ao acaso (rotulo embaralhado). So leitura.

Mesma rotina de zonas_dias2.py / tatuape_ep.py: eventos = passagens atrasadas, ST-DBSCAN 400 m, 30 min, minimo 10,
dia a dia. Zona do episodio = moda das zonas (ponto medio do trecho que chega na parada) das ocorrencias fora dos
trechos de ponta; episodios com mais da metade das ocorrencias em ponta saem. Contagem: zonas com >= 5 mil
passagens sem ponta e episodio comecando 16h-19h59 de dia util em >= 15 dos 20 dias uteis (observado: 17).
O agrupamento e independente por dia, entao rodar so os dias uteis da exatamente o mesmo resultado para a contagem.
Nada e treinado nem ajustado; nao usa a semana de teste de outro modo alem do que ja estava nas contagens anteriores.
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/app")
from olhovivo import geo, regioes  # noqa: E402
from olhovivo.config import carregar  # noqa: E402

T0 = time.time()
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
RAIZ = Path("/dados")
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}
cfg = carregar()
proj = geo.obter_projecao(int(cfg.get("geo.epsg_metrico", 31983)), float(cfg.get("geo.ancora_lat", -23.5505)),
                          float(cfg.get("geo.ancora_lon", -46.6333)))


def tempo():
    return f"[{(time.time() - T0) / 60:.1f} min]"


ARQ = RAIZ / "derivado" / "percursos.parquet"
nomes_col = pq.read_schema(ARQ).names
print("colunas de percursos:", nomes_col)
COL_BUS = next((c for c in ("prefixo", "p", "veiculo", "id_veiculo", "onibus", "prefixo_veiculo") if c in nomes_col), None)
print("coluna de onibus (prefixo):", COL_BUS)
cols = ["cl", "letreiro", "sentido", "viagem_id", "cp_origem", "cp_destino", "ordem_origem", "t_origem", "t_chegada",
        "tempo_percurso_s", "atraso_s", "atrasado", "lat", "lon"] + ([COL_BUS] if COL_BUS else [])
pc = pd.read_parquet(ARQ, columns=cols).dropna(subset=["tempo_percurso_s"]).reset_index(drop=True)
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
print("exemplos de viagem_id:", pc["viagem_id"].drop_duplicates().head(5).tolist())
if COL_BUS:
    print(f"onibus distintos: {pc[COL_BUS].nunique():,} | viagens por onibus (mediana): "
          f"{pc.groupby(COL_BUS)['viagem_id'].nunique().median():.0f}")

# ------------------------------------------------------------------ zonas (ponto medio do trecho que chega na parada)
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


mid = pc.groupby("cp_destino").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
mid["zona"] = zona_de(mid)
zona_mid = dict(zip(mid["cp_destino"], mid["zona"]))

# ------------------------------------------------------------------ trechos de ponta (saida/chegada no terminal)
s = pc.sort_values(["viagem_id", "ordem_origem"]).reset_index(drop=True)
nv = s.groupby("viagem_id")["ordem_origem"].transform("nunique")
nl = s.groupby(["cl", "sentido"])["ordem_origem"].transform("nunique")
base = s[nv >= 0.5 * nl]
tot = base.groupby(["cl", "sentido"])["viagem_id"].nunique().rename("tot").reset_index()
cov = (base.groupby(["cl", "sentido", "ordem_origem"]).agg(n=("viagem_id", "nunique")).reset_index()
       .merge(tot, on=["cl", "sentido"]))
cov["cob"] = cov["n"] / cov["tot"]
anc = []
for (cl, se), g in cov.groupby(["cl", "sentido"]):
    g = g.sort_values("ordem_origem")
    boa = g[g["cob"] >= 0.7]
    if len(boa) < 6:
        continue
    anc.append(dict(cl=cl, sentido=se, o_a0=int(boa.iloc[0]["ordem_origem"]), o_b0=int(boa.iloc[-1]["ordem_origem"])))
anc = pd.DataFrame(anc)
del s, base, nv, nl
lims = pc.groupby(["cl", "sentido"])["ordem_origem"].agg(o_min="min", o_max="max").reset_index()
lims = lims.merge(anc, on=["cl", "sentido"], how="left")
lims["o_a0"] = lims["o_a0"].fillna(lims["o_min"])
lims["o_b0"] = lims["o_b0"].fillna(lims["o_max"])

pe = pc.dropna(subset=["atraso_s", "lat", "lon"]).reset_index(drop=True)
del pc
pp = pe[["cl", "sentido", "ordem_origem"]].merge(lims[["cl", "sentido", "o_a0", "o_b0"]], on=["cl", "sentido"], how="left")
PONTA = ((pp["ordem_origem"] <= pp["o_a0"]) | (pp["ordem_origem"] >= pp["o_b0"])).to_numpy()
del pp
ZONA = pe["cp_destino"].map(zona_mid).to_numpy(dtype=object)
TIPO = pe["tipo"].to_numpy()
x, y = proj.para_xy(pe["lat"].to_numpy(), pe["lon"].to_numpy())
X = np.asarray(x, dtype="float64")
Y = np.asarray(y, dtype="float64")
TS = (pe["t_chegada"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
DIA = pe["dia"].to_numpy()
ATR = pe["atrasado"].astype(bool).to_numpy()
pas_sp = pd.Series(ZONA[~PONTA]).dropna().value_counts()
Z5 = set(pas_sp[pas_sp >= 5000].index)
du = sorted(pe.loc[pe["tipo"] == "util", "dia"].unique())
print(f"{tempo()} passagens com atraso medido {len(pe):,} | atrasadas {ATR.mean() * 100:.1f}% | dias uteis {len(du)} | "
      f"zonas >= 5 mil passagens sem ponta {len(Z5)}")
# conferencia: projecao igual a de preparar_amostras
chk = regioes.preparar_amostras(pd.DataFrame({"valor_s": [1.0], "evento": [True], "lat": pe["lat"].iloc[:1].to_numpy(),
                                              "lon": pe["lon"].iloc[:1].to_numpy()}), proj, "eventos")
print("conferencia da projecao:", np.allclose([chk["x_m"].iloc[0], chk["y_m"].iloc[0]], [X[0], Y[0]]))


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


def episodios(evento, dias):
    """ST-DBSCAN dia a dia. Devolve (indices das ocorrencias em episodio, rotulo do episodio, total de ocorrencias)."""
    idx_all, ep_all, n_ev = [], [], 0
    for k, d in enumerate(dias):
        ii = np.flatnonzero(evento & (DIA == d))  # mesma ordem de linhas que o groupby('dia') do original
        n_ev += len(ii)
        if len(ii) == 0:
            continue
        viz = regioes._vizinhanca_espaco_tempo(X[ii], Y[ii], TS[ii], 400.0, 1800.0)
        r = expandir(viz, 10)
        del viz
        m = r >= 0
        idx_all.append(ii[m])
        ep_all.append(r[m] + 1_000_000 * k)
    return np.concatenate(idx_all), np.concatenate(ep_all), n_ev


def resumo_zonas(idx, ep):
    """Dias uteis (de 20) com episodio comecando 16-19h, por zona (sem ponta), e numero de zonas com >= 15."""
    E = pd.DataFrame({"ep": ep, "i": idx, "ponta": PONTA[idx], "zona": ZONA[idx], "t": TS[idx], "dia": DIA[idx],
                      "tipo": TIPO[idx]})
    S = E.groupby("ep").agg(t0=("t", "min"), dia=("dia", "first"), tipo=("tipo", "first"), frac=("ponta", "mean"))
    c = E[~E["ponta"]].dropna(subset=["zona"]).groupby(["ep", "zona"]).size().rename("c").reset_index()
    c = c.sort_values(["ep", "c", "zona"], ascending=[True, False, True]).drop_duplicates("ep").set_index("ep")["zona"]
    S["zona_sp"] = c
    S["h0"] = pd.to_datetime(S["t0"], unit="s", utc=True).dt.tz_convert(FUSO).dt.hour
    S2 = S[(S["frac"] <= 0.5) & (S["tipo"] == "util") & S["h0"].between(16, 19)]
    dz = S2.groupby("zona_sp")["dia"].nunique()
    dz = dz[dz.index.isin(Z5)]
    return dz, int((dz >= 15).sum()), len(S)


# ================================================================== 1. observado, todos os dias, onibus por episodio
print(f"\n{tempo()} == 1. episodios observados (todos os dias) e onibus distintos")
dias_todos = sorted(np.unique(DIA))
idx, ep, n_ev = episodios(ATR, dias_todos)
print(f"{tempo()} ocorrencias de atraso {n_ev:,} | ocorrencias em episodio {len(idx):,} | episodios {len(np.unique(ep)):,} "
      f"(referencia: 708.006 ocorrencias, 7.792 episodios)")
E = pd.DataFrame({"ep": ep, "i": idx, "letreiro": pe["letreiro"].to_numpy()[idx], "viagem": pe["viagem_id"].to_numpy()[idx],
                  "zona": ZONA[idx], "t": TS[idx], "dia": DIA[idx], "tipo": TIPO[idx], "ponta": PONTA[idx]})
if COL_BUS:
    E["bus"] = pe[COL_BUS].to_numpy()[idx]
S = E.groupby("ep").agg(n=("i", "size"), t0=("t", "min"), dia=("dia", "first"), tipo=("tipo", "first"),
                        linhas=("letreiro", "nunique"), viagens=("viagem", "nunique"), frac=("ponta", "mean"),
                        zona=("zona", lambda s_: s_.mode().iloc[0] if s_.notna().any() else None))
if COL_BUS:
    S["onibus"] = E.groupby("ep")["bus"].nunique()
S["h0"] = pd.to_datetime(S["t0"], unit="s", utc=True).dt.tz_convert(FUSO).dt.hour
S["ocorr_por_viagem"] = S["n"] / S["viagens"]
UN = "onibus" if COL_BUS else "viagens"
print(f"unidade de 'onibus': {'prefixo do veiculo (' + COL_BUS + ')' if COL_BUS else 'viagem distinta (sem coluna de prefixo)'}")


def perfil(g, nome):
    r = {"grupo": nome, "episodios": len(g), "ocorr_med": g["n"].median(), "viagens_med": g["viagens"].median(),
         "viagens_p25": g["viagens"].quantile(0.25), "viagens_p75": g["viagens"].quantile(0.75),
         "pct_1viagem": 100 * (g["viagens"] == 1).mean(), "pct_3mais_viagens": 100 * (g["viagens"] >= 3).mean(),
         "ocorr_por_viagem_med": g["ocorr_por_viagem"].median(),
         "pct_2mais_linhas": 100 * (g["linhas"] >= 2).mean(), "pct_3mais_linhas": 100 * (g["linhas"] >= 3).mean()}
    if COL_BUS:
        r.update({"onibus_med": g["onibus"].median(), "onibus_p25": g["onibus"].quantile(0.25),
                  "onibus_p75": g["onibus"].quantile(0.75), "pct_1onibus": 100 * (g["onibus"] == 1).mean(),
                  "pct_3mais_onibus": 100 * (g["onibus"] >= 3).mean(),
                  "pct_2linhas_e_3onibus": 100 * ((g["linhas"] >= 2) & (g["onibus"] >= 3)).mean()})
    r["pct_2linhas_e_3viagens"] = 100 * ((g["linhas"] >= 2) & (g["viagens"] >= 3)).mean()
    r["n_2mais_linhas"] = int((g["linhas"] >= 2).sum())
    r["n_3mais_" + UN] = int((g[UN] >= 3).sum())
    return r


ut = (S["tipo"] == "util") & S["h0"].between(16, 19)
tat = S["zona"] == "Tatuapé"
P = pd.DataFrame([perfil(S, "todos"), perfil(S[ut], "util 16-19h"), perfil(S[tat], "Tatuape (todos)"),
                  perfil(S[tat & ut], "Tatuape util 16-19h"), perfil(S[S["frac"] <= 0.5], "sem ponta (<= metade)")])
print(P.round(1).T.to_string())
print("distribuicao de " + UN + " por episodio (todos):")
print(S[UN].clip(upper=30).value_counts().sort_index().to_string())
del E

# ================================================================== 2. modelo ao acaso
print(f"\n{tempo()} == 2. modelo ao acaso: rotulo 'atrasado' embaralhado entre passagens da mesma linha, sentido, hora e "
      f"tipo de dia; so dias uteis (o agrupamento e por dia, entao o observado nao muda)")
utl = TIPO == "util"
idx, ep, n_ev = episodios(ATR & utl, du)
dz_obs, k_obs, ne_obs = resumo_zonas(idx, ep)
print(f"{tempo()} observado (so dias uteis): zonas >= 15 dias = {k_obs} | episodios {ne_obs:,} | ocorrencias {n_ev:,}")
print("dias por zona (observado, top 20):")
print(dz_obs.sort_values(ascending=False).head(20).to_string())

iu = np.flatnonzero(utl)
pu = pe.iloc[iu]


def grupos(chaves):
    return pu.groupby(chaves, sort=False).ngroup().to_numpy()


def embaralhar(gid, rng):
    lab = ATR[iu]
    ordem_g = np.argsort(gid, kind="stable")
    ordem_r = np.lexsort((rng.random(len(gid)), gid))
    novo = np.empty_like(lab)
    novo[ordem_g] = lab[ordem_r]
    ev = np.zeros(len(ATR), dtype=bool)
    ev[iu] = novo
    return ev


def rodar(gid, nome, reps, limite_min, semente):
    rng = np.random.default_rng(semente)
    ks, nes, dzs = [], [], []
    t_ini = time.time()
    for r in range(reps):
        ev = embaralhar(gid, rng)
        a, b, _ = episodios(ev, du)
        dz, k, ne = resumo_zonas(a, b)
        ks.append(k)
        nes.append(ne)
        dzs.append(dz)
        print(f"{tempo()} {nome} rep {r + 1}: zonas >= 15 dias = {k} | episodios {ne:,} | max dias {int(dz.max()) if len(dz) else 0}",
              flush=True)
        if (time.time() - T0) / 60 > limite_min and r + 1 >= 10:
            print(f"   parando por tempo apos {r + 1} repeticoes")
            break
    ks = np.array(ks)
    D = pd.concat(dzs, axis=1).fillna(0)
    print(f"== {nome}: {len(ks)} repeticoes | zonas >= 15 dias: media {ks.mean():.2f}, mediana {np.median(ks):.0f}, "
          f"min {ks.min()}, max {ks.max()} | repeticoes com >= {k_obs}: {(ks >= k_obs).sum()} | episodios media {np.mean(nes):,.0f} "
          f"(observado {ne_obs:,}) | tempo por rep {(time.time() - t_ini) / len(ks) / 60:.2f} min")
    cmp = pd.DataFrame({"dias_obs": dz_obs, "dias_acaso_media": D.mean(axis=1), "dias_acaso_max": D.max(axis=1),
                        "pct_reps_15mais": 100 * (D >= 15).mean(axis=1)}).fillna(0)
    print("zonas observadas com >= 15 dias, comparadas com o acaso:")
    print(cmp[cmp["dias_obs"] >= 15].sort_values("dias_obs", ascending=False).round(1).to_string())
    print("zonas que mais chegam a 15+ no acaso:")
    print(cmp.sort_values(["pct_reps_15mais", "dias_acaso_media"], ascending=False).head(12).round(1).to_string())
    print("distribuicao do numero de zonas >= 15 dias no acaso:", dict(zip(*np.unique(ks, return_counts=True))))
    return ks


g1 = grupos(["cl", "sentido", "hora", "tipo"])
print(f"grupos (linha, sentido, hora, tipo): {g1.max() + 1}")
ks1 = rodar(g1, "acaso A (linha, sentido, hora, tipo)", 50, 30.0, 2026)
# sensibilidade: embaralha so dentro do mesmo dia (mantem a taxa de cada dia: chuva, feriado ponte, etc.)
g2 = grupos(["cl", "sentido", "hora", "dia"])
ks2 = rodar(g2, "acaso B (linha, sentido, hora, dia)", 10, 37.0, 11)
print(f"\n{tempo()} fim")
