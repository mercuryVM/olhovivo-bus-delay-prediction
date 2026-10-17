"""Diferencas de MAE entre modelos no teste, com IC95% por bootstrap de viagens.

    python scripts/bootstrap_atraso.py dados_semana [cache_atraso_v2] [reamostras]

Le as previsoes gravadas por comparar_atraso.py. Reamostra viagens inteiras,
e nao trechos, porque os trechos de uma mesma viagem sao correlacionados: o
intervalo sairia estreito demais. Redes neurais entram como a media das
previsoes das sementes, como na tabela da comparacao.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "dados")
cache = raiz / "derivado" / "modelos" / (sys.argv[2] if len(sys.argv) > 2 else "cache_atraso_v2")
reamostras = int(sys.argv[3]) if len(sys.argv) > 3 else 2000


def carregar(nome):
    return pickle.loads((cache / f"{nome}.pkl").read_bytes())


def media_sementes(prefixo, chave=None):
    rodadas = []
    for arq in sorted(cache.glob(f"{prefixo}_*.pkl")):
        r = pickle.loads(arq.read_bytes())
        rodadas.append((r[chave] if chave else r)["q"][:, 1])
    return np.mean(rodadas, axis=0), len(rodadas)


base = carregar("base")
teste = base["df"].loc[base["teste"]]
y = teste["alvo_s"].to_numpy("float64")
viagem = teste["viagem_id"].to_numpy()

previsoes = {"XGBoost": carregar("xgboost")["previsoes"]["p50"]}
for nome, prefixo, chave in (("LSTM", "lstm_completa", None), ("GNN", "gnn_gcn", "rodada"),
                             ("GNN sem grafo", "gnn_sem_grafo", "rodada")):
    previsoes[nome], n = media_sementes(prefixo, chave)
    print(f"{nome}: {n} sementes")

erro = {k: np.abs(np.asarray(v, "float64") - y) for k, v in previsoes.items()}
for k, e in erro.items():
    print(f"MAE {k}: {e.mean():.2f} s")

# soma do erro e contagem por viagem: a reamostragem vira soma ponderada
_, grupo = np.unique(viagem, return_inverse=True)
n_viagens = grupo.max() + 1
contagem = np.bincount(grupo, minlength=n_viagens)
somas = {k: np.bincount(grupo, weights=e, minlength=n_viagens) for k, e in erro.items()}
rng = np.random.default_rng(0)
pesos = rng.multinomial(n_viagens, np.full(n_viagens, 1 / n_viagens), size=reamostras)
total = pesos @ contagem

print(f"\nteste: {len(y)} trechos em {n_viagens} viagens | {reamostras} reamostras")
for a, b in (("XGBoost", "LSTM"), ("XGBoost", "GNN"), ("GNN", "LSTM"), ("GNN sem grafo", "GNN")):
    dif = (pesos @ (somas[a] - somas[b])) / total
    lo, hi = np.percentile(dif, [2.5, 97.5])
    print(f"{b} erra {erro[a].mean() - erro[b].mean():+.2f} s a menos que {a} "
          f"(IC95% {lo:+.2f} a {hi:+.2f})")
