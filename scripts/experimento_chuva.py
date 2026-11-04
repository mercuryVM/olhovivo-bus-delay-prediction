"""A chuva melhora o XGBoost de atraso? Mesma base e teste de comparar_atraso.py.

    python scripts/experimento_chuva.py dados_semana

Atributos de chuva (chuva_openmeteo.parquet), todos conhecidos em t_origem: a
precipitacao da Open-Meteo no instante T e o acumulado da hora ANTERIOR a T,
entao usar T = t_origem arredondado para baixo nao olha o futuro.

* chuva_1h_mm / chuva_3h_mm: na celula da grade (~5 km) do trecho.
* chuva_cidade_1h_mm: media da grade inteira na mesma hora.

Treina o XGBoost com e sem esses atributos (mesmos parametros) e compara o
MAE no teste inteiro e so nos trechos com chuva.
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

chuva = pd.read_parquet(raiz / "derivado" / "chuva_openmeteo.parquet")
chuva["hora"] = chuva["hora_local"].dt.tz_convert("UTC")
grade = chuva.pivot_table(index="hora", columns=["lat", "lon"], values="chuva_mm").sort_index()
tres = grade.rolling(3, min_periods=1).sum()
cidade = grade.mean(axis=1)

passo = 0.05
lat_g = (np.round(df["lat"].to_numpy() / passo) * passo).round(2)
lon_g = (np.round(df["lon"].to_numpy() / passo) * passo).round(2)
lat_g = np.clip(lat_g, chuva["lat"].min(), chuva["lat"].max())
lon_g = np.clip(lon_g, chuva["lon"].min(), chuva["lon"].max())
hora = pd.to_datetime(df["t_origem"], utc=True).dt.floor("h").to_numpy()
colunas_grade = {c: i for i, c in enumerate(grade.columns)}
j = np.array([colunas_grade.get((a, b), -1) for a, b in zip(lat_g, lon_g)])
i = grade.index.get_indexer(hora)
ok = (i >= 0) & (j >= 0)
for nome, tabela in (("chuva_1h_mm", grade), ("chuva_3h_mm", tres)):
    v = np.full(len(df), np.nan)
    v[ok] = tabela.to_numpy()[i[ok], j[ok]]
    df[nome] = v
df["chuva_cidade_1h_mm"] = cidade.reindex(hora).to_numpy()
CHUVA = ["chuva_1h_mm", "chuva_3h_mm", "chuva_cidade_1h_mm"]
print("cobertura dos atributos de chuva:", {c: f"{df[c].notna().mean():.1%}" for c in CHUVA})

base_cols = [c for c in COLUNAS_MODELO + vizinhanca.COLUNAS_VIZINHANCA if c in df.columns]
tr, va, te = base["treino"], base["validacao"], base["teste"]
y = df["alvo_s"]


def rodar(colunas: list[str]) -> np.ndarray:
    X = df[colunas].apply(pd.to_numeric, errors="coerce")
    reg = xgb.XGBRegressor(
        objective="reg:absoluteerror", eval_metric="mae", n_estimators=1500, max_depth=6,
        learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        n_jobs=-1, early_stopping_rounds=50,
    )
    reg.fit(X.loc[tr], y.loc[tr], eval_set=[(X.loc[va], y.loc[va])], verbose=False)
    imp = dict(zip(colunas, reg.feature_importances_))
    return reg.predict(X.loc[te]), {c: round(float(imp[c]), 3) for c in colunas if c in CHUVA}


p_sem, _ = rodar(base_cols)
p_com, imp = rodar(base_cols + CHUVA)
yte = y.loc[te].to_numpy()
c1 = df.loc[te, "chuva_1h_mm"].to_numpy()
treino_chuva = (df.loc[tr, "chuva_1h_mm"] > 0.5).mean()

grupos = {
    "teste inteiro": np.ones(len(te), bool),
    "sem chuva (1h = 0)": c1 == 0,
    "chuva fraca (0 a 2 mm/h)": (c1 > 0) & (c1 < 2),
    "chuva forte (2 mm/h ou mais)": c1 >= 2,
}
linhas = []
for nome, m in grupos.items():
    linhas.append({
        "grupo": nome, "trechos": int(m.sum()),
        "trivial": float(np.abs(yte[m]).mean()),
        "sem_chuva": float(np.abs(yte[m] - p_sem[m]).mean()),
        "com_chuva": float(np.abs(yte[m] - p_com[m]).mean()),
    })

print(f"\ntrechos do treino com chuva > 0,5 mm/h: {treino_chuva:.1%} | "
      f"do teste: {(c1 > 0.5).mean():.1%}")
print("importancia dos atributos de chuva:", imp)
print("\n| Grupo do teste | Trechos | Trivial | XGBoost sem chuva | XGBoost com chuva |")
print("|---|---|---|---|---|")
for l in linhas:
    print(f"| {l['grupo']} | {l['trechos']} | {l['trivial']:.2f} | {l['sem_chuva']:.2f} | {l['com_chuva']:.2f} |")

# bootstrap por viagem da diferenca no teste inteiro
viagem = df.loc[te, "viagem_id"].to_numpy()
_, g = np.unique(viagem, return_inverse=True)
dif = np.bincount(g, np.abs(yte - p_sem) - np.abs(yte - p_com))
cont = np.bincount(g)
rng = np.random.default_rng(0)
pesos = rng.multinomial(g.max() + 1, np.full(g.max() + 1, 1 / (g.max() + 1)), size=2000)
amostras = (pesos @ dif) / (pesos @ cont)
lo, hi = np.percentile(amostras, [2.5, 97.5])
print(f"\nganho com chuva no teste inteiro: {dif.sum() / cont.sum():+.2f} s (IC95% {lo:+.2f} a {hi:+.2f})")
(saida / "experimento_chuva.json").write_text(json.dumps(
    {"grupos": linhas, "importancia": imp, "ganho_s": float(dif.sum() / cont.sum()),
     "ic95": [float(lo), float(hi)]}, indent=1, ensure_ascii=False), encoding="utf-8")
