"""Novas variaveis para o XGBoost de atraso, uma de cada vez e todas juntas.

    python scripts/experimento_variaveis.py dados_semana

Mesma base, divisao e parametros de comparar_atraso.py (cache_atraso_v2).
Tudo e conhecido em t_origem:

* intervalo: segundos desde que o onibus anterior da linha saiu da parada de
  origem (o `headway_s` de percursos e medido no DESTINO com a chegada deste
  onibus, entao nao serve: olharia o futuro) e a razao contra o intervalo
  mediano do treino no mesmo ponto e faixa.
* vizinhos_na_linha: quantas paradas a frente esta o onibus anterior e quantas
  atras esta o seguinte, pela ultima parada que cada um passou antes de
  t_origem. O seguinte e identificado como o proximo a passar na origem, o que
  um sistema ao vivo faria pelo GPS (o mais proximo atras na rota).
* variabilidade: p75, p90 e desvio-padrao do alvo no treino, por trecho e faixa.
* tendencia: desvio recente do trecho menos o desvio da janela de 30 a 60 min
  antes.
* viagem: atraso acumulado da viagem ate o trecho anterior e fracao da rota.
* feriado: 07/09/2026.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olhovivo import vizinhanca  # noqa: E402
from olhovivo.atraso import COLUNAS_MODELO  # noqa: E402

raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "dados")
saida = raiz / "derivado" / "modelos"
base = pickle.loads((saida / "cache_atraso_v2" / "base.pkl").read_bytes())
viz = pickle.loads((saida / "cache_atraso_v2" / "vizinhanca.pkl").read_bytes())
df = base["df"].join(viz)
tr, va, te = base["treino"], base["validacao"], base["teste"]
NS = "datetime64[ns, UTC]"
df["_to"] = pd.to_datetime(df["t_origem"], utc=True).astype(NS)
df["_tc"] = pd.to_datetime(df["t_chegada"], utc=True).astype(NS)

# ---------------------------------------------------------------- intervalo
o = df.sort_values(["cl", "cp_origem", "_to"])
g = o.groupby(["cl", "cp_origem"], sort=False)
df["intervalo_s"] = g["_to"].diff().dt.total_seconds()
df["_anterior"] = g["viagem_id"].shift(1)
df["_seguinte"] = g["viagem_id"].shift(-1)
ref_int = (df.loc[tr].groupby(["cl", "cp_origem", "faixa"])["intervalo_s"].median()
           .rename("_int_ref"))
df = df.join(ref_int, on=["cl", "cp_origem", "faixa"])
df["intervalo_rel"] = df["intervalo_s"] / df["_int_ref"]

# ------------------------------------------------------- vizinhos na linha
# ultima parada que cada viagem passou ate um instante: chegadas (ordem_destino em t_chegada)
pas = (df[["viagem_id", "ordem_destino", "_tc"]].rename(columns={"_tc": "t", "ordem_destino": "ordem"})
       .dropna().sort_values("t"))


def posicao(col_viagem: str) -> pd.Series:
    q = (df[[col_viagem, "_to"]].dropna().rename(columns={col_viagem: "viagem_id", "_to": "t"})
         .reset_index().sort_values("t"))
    m = pd.merge_asof(q, pas, on="t", by="viagem_id", allow_exact_matches=False)
    return m.set_index("index")["ordem"]


df["paradas_ate_anterior"] = posicao("_anterior") - df["ordem_origem"]
df["paradas_ate_seguinte"] = df["ordem_origem"] - posicao("_seguinte")

# ------------------------------------------------------------- variabilidade
v = df.loc[tr].groupby(["cl", "ordem_origem", "faixa"])["alvo_s"]
var = pd.concat({"trecho_p75_s": v.quantile(0.75), "trecho_p90_s": v.quantile(0.90),
                 "trecho_dp_s": v.std()}, axis=1)
df = df.join(var, on=["cl", "ordem_origem", "faixa"])

# ---------------------------------------------------------------- tendencia
ev = (df[["cl", "ordem_origem", "_tc", "alvo_s"]].dropna().sort_values(["cl", "ordem_origem", "_tc"])
      .reset_index(drop=True))
ev["_med"] = (ev.set_index("_tc").groupby(["cl", "ordem_origem"])["alvo_s"]
              .rolling("30min").median().to_numpy())
q = df[["cl", "ordem_origem", "_to"]].reset_index()
q["t"] = q["_to"] - pd.Timedelta("30min")
m = pd.merge_asof(q.sort_values("t"), ev.rename(columns={"_tc": "t"})[["cl", "ordem_origem", "t", "_med"]]
                  .sort_values("t"), on="t", by=["cl", "ordem_origem"],
                  allow_exact_matches=False, tolerance=pd.Timedelta("30min"))
df["desvio_30_60min_s"] = m.set_index("index")["_med"]
df["tendencia_s"] = df["desvio_recente_trecho_s"] - df["desvio_30_60min_s"]

# ------------------------------------------------------------------- viagem
s = df.sort_values(["viagem_id", "ordem_origem"])
df["atraso_acumulado_s"] = s.groupby("viagem_id")["alvo_s"].transform(
    lambda x: x.cumsum().shift(1)).fillna(0)
df["fracao_rota"] = df["ordem_origem"] / df.groupby("cl")["ordem_destino"].transform("max")

# ------------------------------------------------------------------ feriado
dia_local = df["_to"].dt.tz_convert("America/Sao_Paulo").dt.date.astype(str)
df["feriado"] = dia_local.isin(["2026-09-07"]).astype("int8")

GRUPOS = {
    "intervalo": ["intervalo_s", "intervalo_rel"],
    "vizinhos_na_linha": ["paradas_ate_anterior", "paradas_ate_seguinte"],
    "variabilidade": ["trecho_p75_s", "trecho_p90_s", "trecho_dp_s"],
    "tendencia": ["desvio_30_60min_s", "tendencia_s"],
    "viagem": ["atraso_acumulado_s", "fracao_rota"],
    "feriado": ["feriado"],
}
print("cobertura:", {c: f"{df[c].notna().mean():.0%}" for cs in GRUPOS.values() for c in cs})

BASE = [c for c in COLUNAS_MODELO + vizinhanca.COLUNAS_VIZINHANCA if c in df.columns]
y = df["alvo_s"]
yte = y.loc[te].to_numpy()


def rodar(colunas: list[str]):
    X = df[colunas].apply(pd.to_numeric, errors="coerce")
    reg = xgb.XGBRegressor(
        objective="reg:absoluteerror", eval_metric="mae", n_estimators=1500, max_depth=6,
        learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        n_jobs=-1, early_stopping_rounds=50,
    )
    reg.fit(X.loc[tr], y.loc[tr], eval_set=[(X.loc[va], y.loc[va])], verbose=False)
    return reg.predict(X.loc[te]), dict(zip(colunas, reg.feature_importances_.astype(float)))


_, grupo_viagem = np.unique(df.loc[te, "viagem_id"].to_numpy(), return_inverse=True)
cont = np.bincount(grupo_viagem)
pesos = np.random.default_rng(0).multinomial(
    len(cont), np.full(len(cont), 1 / len(cont)), size=2000)


def ganho(p_ref, p_novo):
    d = np.bincount(grupo_viagem, np.abs(yte - p_ref) - np.abs(yte - p_novo))
    amostras = (pesos @ d) / (pesos @ cont)
    return float(d.sum() / cont.sum()), [float(x) for x in np.percentile(amostras, [2.5, 97.5])]


if "--combinacoes" in sys.argv:
    # a variabilidade calculada sobre o proprio treino vaza o alvo de cada linha
    # para ela mesma; aqui cada dia do treino usa so as estatisticas dos outros
    # dias (validacao e teste seguem com as do treino inteiro)
    dia = df["_to"].dt.tz_convert("America/Sao_Paulo").dt.date
    chave = ["cl", "ordem_origem", "faixa"]
    tr_df, partes = df.loc[tr], []
    for d in sorted(dia.loc[tr].unique()):
        outros = tr_df[dia.loc[tr] != d].groupby(chave)["alvo_s"]
        est = pd.concat({"oof_p75_s": outros.quantile(0.75), "oof_p90_s": outros.quantile(0.90),
                         "oof_dp_s": outros.std()}, axis=1)
        idx = tr_df.index[dia.loc[tr] == d]
        partes.append(df.loc[idx, chave].join(est, on=chave).drop(columns=chave))
    fora = df.index.difference(tr)
    partes.append(df.loc[fora, ["trecho_p75_s", "trecho_p90_s", "trecho_dp_s"]].set_axis(
        ["oof_p75_s", "oof_p90_s", "oof_dp_s"], axis=1))
    df = df.join(pd.concat(partes))
    OOF = ["oof_p75_s", "oof_p90_s", "oof_dp_s"]
    TRES = GRUPOS["viagem"] + GRUPOS["tendencia"] + GRUPOS["feriado"]
    p_base, _ = rodar(BASE)
    print(f"| atual | {np.abs(yte - p_base).mean():.2f} | — | — |", flush=True)
    for nome, cols in (("viagem + tendencia + feriado", TRES),
                       ("so variabilidade fora do dia", OOF),
                       ("viagem + tendencia + feriado + variabilidade fora do dia", TRES + OOF)):
        p, _ = rodar(BASE + cols)
        g, ic = ganho(p_base, p)
        print(f"| {nome} | {np.abs(yte - p).mean():.2f} | {g:+.2f} | {ic[0]:+.2f} a {ic[1]:+.2f} |",
              flush=True)
    sys.exit(0)

p_base, _ = rodar(BASE)
linhas =[{"variaveis": "atual (referencia)", "mae_s": float(np.abs(yte - p_base).mean())}]
print(f"atual: {linhas[0]['mae_s']:.2f}", flush=True)
todas = []
for nome, cols in GRUPOS.items():
    p, _ = rodar(BASE + cols)
    g, ic = ganho(p_base, p)
    linhas.append({"variaveis": f"+ {nome}", "mae_s": float(np.abs(yte - p).mean()),
                   "ganho_s": g, "ic95": ic})
    todas += cols
    print(f"+ {nome}: {linhas[-1]['mae_s']:.2f} (ganho {g:+.2f}, IC {ic[0]:+.2f} a {ic[1]:+.2f})", flush=True)

p, imp = rodar(BASE + todas)
g, ic = ganho(p_base, p)
linhas.append({"variaveis": "+ todas", "mae_s": float(np.abs(yte - p).mean()), "ganho_s": g, "ic95": ic})
top = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:12])

print(f"\n| Variaveis | MAE (s) | Ganho (s) | IC95% |\n|---|---|---|---|")
for l in linhas:
    extra = (f"{l['ganho_s']:+.2f} | {l['ic95'][0]:+.2f} a {l['ic95'][1]:+.2f}"
             if "ganho_s" in l else "— | —")
    print(f"| {l['variaveis']} | {l['mae_s']:.2f} | {extra} |")
print("\nimportancia (todas):", {k: round(v, 3) for k, v in top.items()})
(saida / "experimento_variaveis.json").write_text(json.dumps(
    {"linhas": linhas, "importancia_todas": top}, indent=1, ensure_ascii=False), encoding="utf-8")
