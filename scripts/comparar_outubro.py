"""Modelos de atraso com os dados de outubro: o modelo de setembro ainda vale?
Mais dados ajudam?

    python scripts/comparar_outubro.py [--redes]

Junta os percursos da semana 1 (dados_semana) com os de outubro (dados_mes) e
testa todos os cenarios no MESMO teste, os dias uteis de 21 a 23/10:

* set:         treino so na semana de setembro
* set+out:     setembro + 16 a 20/10
* out:         so 16 a 20/10

Em cada cenario a referencia do alvo (tempo tipico do trecho) e os atributos
saem so do treino, como em `modelos.preparar_atraso`; a validacao e a fatia
final do treino. Com --redes roda tambem LSTM e GNN (demora); sem, so XGBoost.
Cache por cenario em dados_mes/derivado/modelos/cache_outubro/.
"""
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olhovivo import contexto_viagem, modelos, modelos_atraso, vizinhanca  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("outubro")
RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "dados_mes" / "derivado" / "modelos"
CACHE = SAIDA / "cache_outubro"
CACHE.mkdir(parents=True, exist_ok=True)
FUSO = "America/Sao_Paulo"
REDES = "--redes" in sys.argv

TESTE = ("2026-10-21", "2026-10-23")
CENARIOS = {
    "set": ("2026-09-02", "2026-09-09"),
    "set+out": ("2026-09-02", "2026-10-20"),
    "out": ("2026-10-16", "2026-10-20"),
}


def etapa(nome, calcular):
    arquivo = CACHE / f"{nome}.pkl"
    if arquivo.exists():
        return pickle.loads(arquivo.read_bytes())
    inicio = time.perf_counter()
    valor = calcular()
    arquivo.write_bytes(pickle.dumps(valor))
    log.info("%s: %.0f s", nome, time.perf_counter() - inicio)
    return valor


set_ = pd.read_parquet(RAIZ / "dados_semana" / "derivado" / "percursos.parquet")
out_ = pd.read_parquet(RAIZ / "dados_mes" / "derivado" / "percursos.parquet")
# viagem_id de periodos diferentes nao pode colidir
set_["viagem_id"] = "set-" + set_["viagem_id"].astype(str)
out_["viagem_id"] = "out-" + out_["viagem_id"].astype(str)
df = pd.concat([set_, out_], ignore_index=True).dropna(subset=["tempo_percurso_s"]).reset_index(drop=True)
dia = pd.to_datetime(df["t_origem"], utc=True).dt.tz_convert(FUSO).dt.strftime("%Y-%m-%d")
previsoes = pd.concat([
    pd.read_parquet(RAIZ.parent / "release-dados-v1" / "previsao_realizado.parquet",
                    columns=["prefixo", "cp", "ts_coleta", "t_previsto"]),
    pd.read_parquet(RAIZ / "dados_mes" / "derivado" / "previsao_realizado.parquet",
                    columns=["prefixo", "cp", "ts_coleta", "t_previsto"]),
], ignore_index=True)
log.info("percursos: setembro %d, outubro %d | dias: %s", len(set_), len(out_),
         dict(dia.value_counts().sort_index()))

i_teste = df.index[(dia >= TESTE[0]) & (dia <= TESTE[1])]


def montar_base(nome: str) -> dict:
    ini, fim = CENARIOS[nome]
    i_tr = df.index[(dia >= ini) & (dia <= fim)]
    treino = df.loc[i_tr].sort_values("t_origem")
    treino, validacao = modelos.separar_validacao(treino, "t_origem")
    usar = df.loc[i_tr.union(i_teste)]
    d = modelos._atributos_atraso(usar, treino.index, 0.20)
    d = d.dropna(subset=["alvo_s"])
    d = d.join(vizinhanca.atributos(d, previsoes))
    d = d.join(contexto_viagem.atributos(d))
    return {
        "df": d,
        "treino": d.index.intersection(treino.index),
        "validacao": d.index.intersection(validacao.index),
        "teste": d.index.intersection(i_teste),
        "semana": df["percurso_tipico_s"],
        "sem_referencia": int(len(i_teste) - len(d.index.intersection(i_teste))),
        "limiar_percentual": 0.20,
    }


resultados = {}
for nome in CENARIOS:
    base = etapa(f"base_{nome}", lambda: montar_base(nome))
    xgb = etapa(f"xgboost_{nome}", lambda: modelos.treinar_atraso(None, base=base))
    xgb = {"resultado": xgb["resultado"], "previsoes": xgb["previsoes"]}
    r = {"teste": base["teste"], "baselines": modelos.baselines_atraso(base),
         "xgboost": xgb, "treino_n": len(base["treino"])}
    if REDES:
        r["lstm"] = etapa(f"lstm_{nome}", lambda: modelos_atraso.treinar_lstm_atraso(
            base=base, sementes=(1, 2, 3), variantes=("completa",)))
        r["gnn"] = etapa(f"gnn_{nome}", lambda: modelos_atraso.treinar_gnn_atraso(
            base=base, sementes=(1, 2, 3), variantes=("gcn",)))
    resultados[nome] = r
    log.info("cenario %s pronto", nome)

# mesmos trechos de teste em todos os cenarios (alguns trechos nao tem referencia
# quando o treino e curto)
comum = i_teste
for r in resultados.values():
    comum = comum.intersection(r["teste"])


# o erro do alvo e o erro do tempo de percurso (alvo = tempo - referencia), entao
# e comparavel entre cenarios com referencias diferentes
linhas = []
for nome, r in resultados.items():
    b = pickle.loads((CACHE / f"base_{nome}.pkl").read_bytes())
    alvo = b["df"]["alvo_s"]
    recente = pd.to_numeric(b["df"]["desvio_recente_trecho_s"], errors="coerce").fillna(0)

    def erro(prev):
        s = pd.Series(np.asarray(prev, "float64"), index=b["teste"])
        return float((alvo.loc[b["teste"]] - s).abs().loc[comum].mean())

    linha = {
        "cenario": nome, "trechos_treino": r["treino_n"],
        "trivial": erro(np.zeros(len(b["teste"]))),
        "recente": erro(recente.loc[b["teste"]].to_numpy()),
        "xgboost": erro(r["xgboost"]["previsoes"]["p50"]),
    }
    if REDES:
        for m in ("lstm", "gnn"):
            v = "completa" if m == "lstm" else "gcn"
            rodadas = r[m]["rodadas"][v]
            linha[m] = erro(np.mean([x["q"][:, 1] for x in rodadas], axis=0))
    linhas.append(linha)

print(f"\nteste: {len(comum)} trechos de {TESTE[0]} a {TESTE[1]} (comuns a todos os cenarios)")
cols = ["trivial", "recente", "xgboost"] + (["lstm", "gnn"] if REDES else [])
print("| Treino | Trechos de treino | " + " | ".join(cols) + " |")
print("|---|---|" + "---|" * len(cols))
for l in linhas:
    print(f"| {l['cenario']} | {l['trechos_treino']} | " + " | ".join(f"{l[c]:.2f}" for c in cols) + " |")
(SAIDA / "comparacao_outubro.json").write_text(
    json.dumps({"teste_comum": len(comum), "linhas": linhas}, indent=1), encoding="utf-8")
