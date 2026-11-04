"""Compara XGBoost, LSTM e GNN sobre o atraso por trecho, na MESMA base.

    python scripts/comparar_atraso.py [pasta_dos_dados] [previsao_realizado.parquet]

Todos usam `modelos.preparar_atraso`: mesmo alvo (desvio contra a mediana
historica do trecho, so com o treino), mesmas linhas de treino, validacao e
teste, mesmas linhas de base. Grava o JSON completo em
derivado/modelos/comparacao_atraso.json e imprime a tabela.

Os tres modelos recebem tambem os atributos de vizinhanca (`vizinhanca.py`):
trechos vizinhos da linha, o trecho fisico com todas as linhas e a previsao da
API, esta lida do parquet de previsoes (padrao: derivado/previsao_realizado).
E os de viagem, tendencia e feriado (`contexto_viagem.py`).

Cada etapa (a base, o XGBoost e cada rodada de rede neural, por variante e
semente) e gravada assim que termina em derivado/modelos/cache_atraso_v3/. Se
o processo cair no meio, rodar de novo retoma de onde parou. Apague a pasta
para recalcular do zero depois de mudar o codigo. As versoes anteriores ficaram
em cache_atraso/ (sem vizinhanca, comparacao_atraso_v1.json) e
cache_atraso_v2/ (sem contexto da viagem, comparacao_atraso_v2.json).
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
log = logging.getLogger("comparar_atraso")
raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "dados")
arq_previsoes = Path(sys.argv[2]) if len(sys.argv) > 2 else raiz / "derivado" / "previsao_realizado.parquet"
saida = raiz / "derivado" / "modelos"
cache = saida / "cache_atraso_v3"
cache.mkdir(parents=True, exist_ok=True)

LSTM = (("completa", (1, 2, 3)), ("so_viagem", (1,)), ("so_trecho", (1,)))
GNN = (("gcn", (1, 2, 3)), ("sem_grafo", (1, 2, 3)))


def etapa(nome: str, calcular):
    arquivo = cache / f"{nome}.pkl"
    if arquivo.exists():
        log.info("etapa %s: do cache", nome)
        return pickle.loads(arquivo.read_bytes())
    inicio = time.perf_counter()
    valor = calcular()
    arquivo.write_bytes(pickle.dumps(valor))
    log.info("etapa %s: %.0f s", nome, time.perf_counter() - inicio)
    return valor


inicio = time.perf_counter()
percursos = pd.read_parquet(raiz / "derivado" / "percursos.parquet")
base = etapa("base", lambda: modelos.preparar_atraso(percursos, 0.20))
viz = etapa("vizinhanca", lambda: vizinhanca.atributos(base["df"], arq_previsoes))
base["df"] = base["df"].join(viz)
contexto = etapa("contexto", lambda: contexto_viagem.atributos(base["df"]))
base["df"] = base["df"].join(contexto)
log.info("vizinhanca: %s", {c: f"{viz[c].notna().mean():.0%}" for c in
                            ("acima_med", "abaixo_med", "fisico_med", "api_desvio_s")})


def xgboost():
    r = modelos.treinar_atraso(percursos, base=base)
    return {"resultado": r["resultado"], "previsoes": r["previsoes"]}


xgb = etapa("xgboost", xgboost)

rodadas_lstm: dict[str, list[dict]] = {}
for variante, sementes in LSTM:
    for semente in sementes:
        rodadas_lstm.setdefault(variante, []).append(etapa(
            f"lstm_{variante}_{semente}",
            lambda: modelos_atraso.treinar_lstm_atraso(
                base=base, sementes=(semente,), variantes=(variante,)
            )["rodadas"][variante][0],
        ))

rodadas_gnn: dict[str, list[dict]] = {}
grafo_info = None
for variante, sementes in GNN:
    for semente in sementes:
        def gnn():
            r = modelos_atraso.treinar_gnn_atraso(
                base=base, sementes=(semente,), variantes=(variante,)
            )
            return {"rodada": r["rodadas"][variante][0], "grafo": r["resultado"]["grafo"]}

        feito = etapa(f"gnn_{variante}_{semente}", gnn)
        rodadas_gnn.setdefault(variante, []).append(feito["rodada"])
        grafo_info = feito["grafo"]

baselines = modelos.baselines_atraso(base)
resultado_lstm = {"baselines": baselines, **modelos_atraso._resumir(base, rodadas_lstm)}
resultado_gnn = {"baselines": baselines, "grafo": grafo_info,
                 **modelos_atraso._resumir(base, rodadas_gnn)}

# ------------------------------------------------- conjunto dos tres modelos
d = base["df"]
yte = d.loc[base["teste"], "alvo_s"].to_numpy("float32")
ctr = d.loc[base["treino"], "alvo_atrasado"].to_numpy("int8")
cva = d.loc[base["validacao"], "alvo_atrasado"].to_numpy("int8")
cte = d.loc[base["teste"], "alvo_atrasado"].to_numpy("int8")


def media(rodadas, chave):
    return np.mean([r[chave] for r in rodadas], axis=0)


p50 = np.mean([
    xgb["previsoes"]["p50"],
    media(rodadas_lstm["completa"], "q")[:, 1],
    media(rodadas_gnn["gcn"], "q")[:, 1],
], axis=0)
prob_va = np.mean([
    xgb["previsoes"]["prob_validacao"],
    media(rodadas_lstm["completa"], "prob_va"),
    media(rodadas_gnn["gcn"], "prob_va"),
], axis=0)
prob_te = np.mean([
    xgb["previsoes"]["prob_teste"],
    media(rodadas_lstm["completa"], "prob_te"),
    media(rodadas_gnn["gcn"], "prob_te"),
], axis=0)
conjunto = {
    "regressao": modelos._metricas(yte, p50),
    "classificacao": modelos._classificacao(ctr, cva, cte, prob_va, prob_te),
}

# --------------------------------------------------------------- tabela
linhas = []
for nome, chave in (("Trivial: mediana histórica do trecho", "trivial_mediana_historica"),
                    ("Desvio recente do trecho", "desvio_recente_do_trecho")):
    linhas.append((nome, baselines[chave]["mae_s"], None, baselines[chave]["r2"],
                   None, None, None, None))
r = xgb["resultado"]
linhas.append(("XGBoost", r["regressao"]["mae_s"], None, r["regressao"]["r2"],
               r["classificacao"]["f1"], r["classificacao"]["auc"],
               r["classificacao"]["brier_skill"], None))
for nome, res, chave in (
    ("LSTM (viagem + trecho)", resultado_lstm, "completa"),
    ("LSTM só viagem", resultado_lstm, "so_viagem"),
    ("LSTM só trecho", resultado_lstm, "so_trecho"),
    ("GNN (com grafo)", resultado_gnn, "gcn"),
    ("GNN sem grafo", resultado_gnn, "sem_grafo"),
):
    v = res[chave]
    cj = v["conjunto"]
    linhas.append((nome, cj["regressao"]["mae_s"], v["mae_s_dp"], cj["regressao"]["r2"],
                   cj["classificacao"]["f1"], cj["classificacao"]["auc"],
                   cj["classificacao"]["brier_skill"],
                   (cj["cobertura_p10"], cj["cobertura_p90"])))
linhas.append(("Conjunto XGBoost + LSTM + GNN", conjunto["regressao"]["mae_s"], None,
               conjunto["regressao"]["r2"], conjunto["classificacao"]["f1"],
               conjunto["classificacao"]["auc"], conjunto["classificacao"]["brier_skill"],
               None))


def fmt(v, casas=2):
    return "—" if v is None else f"{v:.{casas}f}"


print("\n| Modelo | MAE (s) | dp sementes | R² | F1 | AUC | Brier skill | cobertura p10/p90 |")
print("|---|---|---|---|---|---|---|---|")
for nome, mae, dp, r2, f1, auc, bs, cob in linhas:
    cobertura = "—" if cob is None else f"{cob[0]:.1%} / {cob[1]:.1%}"
    print(f"| {nome} | {fmt(mae)} | {fmt(dp)} | {fmt(r2, 3)} | {fmt(f1, 3)} | "
          f"{fmt(auc, 3)} | {fmt(bs, 3)} | {cobertura} |")
print(f"\nteste: {len(yte)} trechos | F1 de marcar tudo como atrasado: "
      f"{r['classificacao']['f1_sempre_atrasado']:.3f} | redes: conjunto das sementes "
      f"| tempo total {time.perf_counter() - inicio:.0f} s")

(saida / "comparacao_atraso.json").write_text(json.dumps({
    "xgboost": xgb["resultado"], "lstm": resultado_lstm, "gnn": resultado_gnn,
    "conjunto": conjunto,
}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
print("JSON:", saida / "comparacao_atraso.json")
