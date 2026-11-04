"""Onde esta o erro do XGBoost de atraso (semana 1, cache_atraso_v3)?

    python scripts/diagnostico_erro.py dados_semana

1. Ruido de medicao. O horario de passagem em cada parada e interpolado entre
   leituras de GPS. O erro dessa interpolacao numa parada entra com sinal + no
   trecho que termina nela e com sinal - no que comeca nela, e aparece como
   covariancia NEGATIVA entre residuos de trechos consecutivos da mesma viagem.
   Se a parte real do residuo decai devagar com a distancia, a variancia do
   ruido por parada e aproximadamente cov(lag 2) - cov(lag 1); a de um trecho
   (duas paradas) e o dobro.
2. Erro pela confianca da interpolacao.
3. Erro pela idade da informacao mais recente no trecho fisico e no ultimo
   onibus: mede quanto vale ter um "sensor" passando ha pouco tempo.
4. Concentracao do erro por trecho e as paradas com maior erro.
5. Vies na cauda: o que o modelo preve nos 10% de maior atraso.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "dados")
cache = raiz / "derivado" / "modelos" / "cache_atraso_v3"
base = pickle.loads((cache / "base.pkl").read_bytes())
df = base["df"]
for parte in ("vizinhanca", "contexto"):
    df = df.join(pickle.loads((cache / f"{parte}.pkl").read_bytes()))
xgb = pickle.loads((cache / "xgboost.pkl").read_bytes())

te = df.loc[base["teste"]].copy()
te["prev"] = xgb["previsoes"]["p50"]
te["res"] = te["alvo_s"] - te["prev"]
mae, triv = te["res"].abs().mean(), te["alvo_s"].abs().mean()
print(f"teste: {len(te)} trechos | MAE modelo {mae:.2f} s | trivial {triv:.2f} s | "
      f"dp do residuo {te['res'].std():.1f} s | tempo tipico mediano {te['percurso_tipico_s'].median():.0f} s")

# ------------------------------------------------------------ 1. ruido
s = te.sort_values(["viagem_id", "ordem_origem"])
g = s.groupby("viagem_id", sort=False)
print("\n1. covariancia entre trechos consecutivos da mesma viagem (s^2)")
cov = {}
for col in ("res", "alvo_s"):
    for lag in (1, 2, 3):
        prox = g[col].shift(-lag)
        ok = (g["ordem_origem"].shift(-lag) - s["ordem_origem"]) == lag
        cov[col, lag] = float(np.cov(s.loc[ok, col], prox[ok])[0, 1])
    print(f"   {col:7s} lag1 {cov[col, 1]:8.1f} | lag2 {cov[col, 2]:8.1f} | lag3 {cov[col, 3]:8.1f}")
var_parada = cov["res", 2] - cov["res", 1]
var_res = float(te["res"].var())
if var_parada > 0:
    dp_trecho = np.sqrt(2 * var_parada)
    print(f"   ruido por parada ~ dp {np.sqrt(var_parada):.1f} s -> por trecho ~ dp {dp_trecho:.1f} s, "
          f"{2 * var_parada / var_res:.0%} da variancia do residuo")
else:
    print("   sem covariancia negativa: ruido de interpolacao nao detectado por este metodo")

# ------------------------------------------------------------ 2. confianca
print("\n2. erro por confianca da interpolacao (quartis)")
if "confianca" in te:
    q = pd.qcut(te["confianca"], 4, duplicates="drop")
    t = te.groupby(q, observed=True).agg(n=("res", "size"), modelo=("res", lambda x: x.abs().mean()),
                                         trivial=("alvo_s", lambda x: x.abs().mean()))
    print(t.round(2).to_string())

# ------------------------------------------------------------ 3. idade da informacao
print("\n3. erro pela idade da informacao mais recente")
cortes = [0, 60, 180, 600, 1800]
rotulos = ["< 1 min", "1-3 min", "3-10 min", "10-30 min"]
for col, nome in (("fisico_idade", "trecho fisico, todas as linhas"),
                  ("desde_ultimo_veiculo_s", "ultimo onibus da linha no trecho")):
    idade = pd.to_numeric(te[col], errors="coerce")
    faixa = pd.cut(idade, cortes, labels=rotulos, right=False).cat.add_categories("sem dado").fillna("sem dado")
    t = te.groupby(faixa, observed=True).agg(n=("res", "size"), modelo=("res", lambda x: x.abs().mean()),
                                             trivial=("alvo_s", lambda x: x.abs().mean()))
    t["ganho"] = 1 - t["modelo"] / t["trivial"]
    t["% do teste"] = t["n"] / len(te)
    print(f"   {nome}:")
    print(t.round(3).to_string())

# ------------------------------------------------------------ 4. concentracao
print("\n4. concentracao do erro")
por_trecho = te.groupby(["cl", "ordem_origem"])["res"].agg(soma=lambda x: x.abs().sum(), n="size")
por_trecho = por_trecho.sort_values("soma", ascending=False)
topo = max(1, int(len(por_trecho) * 0.05))
print(f"   {len(por_trecho)} trechos; os 5% piores ({topo}) somam "
      f"{por_trecho['soma'].iloc[:topo].sum() / por_trecho['soma'].sum():.0%} do erro absoluto "
      f"e {por_trecho['n'].iloc[:topo].sum() / len(te):.0%} das passagens")
por_parada = te.groupby("cp_origem").agg(n=("res", "size"), erro=("res", lambda x: x.abs().mean()),
                                         alvo_medio=("alvo_s", "mean"),
                                         tipico=("percurso_tipico_s", "median"))
por_parada = por_parada[por_parada["n"] >= 200].sort_values("erro", ascending=False)
try:
    nomes = pd.read_parquet(raiz / "catalogo" / "paradas.parquet")
    chave = "cp" if "cp" in nomes else nomes.columns[0]
    nome_col = next(c for c in ("np", "nome", "stop_name") if c in nomes)
    por_parada = por_parada.join(nomes.drop_duplicates(chave).set_index(chave)[nome_col])
except Exception:
    pass
print("   paradas de origem com maior erro medio (>= 200 passagens):")
print(por_parada.head(8).round(1).to_string())

# ------------------------------------------------------------ 5. cauda
print("\n5. vies na cauda")
p90 = te["alvo_s"].quantile(0.9)
cauda = te[te["alvo_s"] >= p90]
print(f"   10% de maior atraso (>= {p90:.0f} s): atraso medio real {cauda['alvo_s'].mean():.0f} s, "
      f"previsto {cauda['prev'].mean():.0f} s; respondem por "
      f"{cauda['res'].abs().sum() / te['res'].abs().sum():.0%} do erro absoluto")
