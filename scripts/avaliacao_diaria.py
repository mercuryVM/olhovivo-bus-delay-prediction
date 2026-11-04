"""Avaliacao dia a dia (walk-forward) das escolhas de referencia, janela e frota.

    python scripts/avaliacao_diaria.py [--dias 2026-10-21,2026-10-22,...] [--final] [--redes]

Para cada dia de teste d e cada configuracao, treina so com os dias anteriores
(janela de W dias ou todos), valida no fim da propria janela (15% finais, como
em modelos.separar_validacao) e testa no dia d. Nenhuma escolha olha o teste:
as configuracoes sao comparadas nos dias de DESENVOLVIMENTO (padrao: 21 a
23/10, ja vistos), e a semana de 02 a 08/11 fica intocada para o teste final
(--final), rodado uma vez so com a configuracao escolhida.

Configuracoes:
* todos_fixa:   todos os dias anteriores, referencia sem tipo de dia, sem frota
                (o que o relatorio parcial fazia)
* 7d_fixa:      ultimos 7 dias
* 7d_tipo:      ultimos 7 dias, referencia por tipo de dia (util/sabado/domingo)
* 7d_tipo_frota: + velocidade da frota inteira no trecho (frota.py)

Metricas no conjunto de trechos comum a todas as configuracoes do dia: MAE (s),
erro relativo (soma dos erros / soma dos tempos reais, comparavel entre
trechos de tamanhos diferentes), MAE dos trechos de ate 3 min e as linhas de
base (trivial e atraso recente).

Com --redes, cada dia tambem treina LSTM e GNN (3 sementes, media das
previsoes) e o conjunto dos tres; sem, so o XGBoost (para escolher rapido).

Dados: dados_mes/derivado (percursos com as paradas do GTFS, previsoes da API e
medidas da frota). Cache em dados_mes/derivado/modelos/cache_diaria/.
"""
import argparse
import json
import logging
import pickle
import time
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olhovivo import contexto_viagem, frota, modelos, modelos_atraso, vizinhanca  # noqa: E402
from olhovivo.atraso import COLUNAS_MODELO  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("diaria")
RAIZ = Path(__file__).resolve().parents[1] / "dados_mes"
DERIV = RAIZ / "derivado"
CACHE = DERIV / "modelos" / "cache_diaria"
CACHE.mkdir(parents=True, exist_ok=True)
FUSO = "America/Sao_Paulo"

CONFIGS = {
    "todos_fixa": {"janela": None, "tipo_dia": False, "frota": False},
    "7d_fixa": {"janela": 7, "tipo_dia": False, "frota": False},
    "7d_tipo": {"janela": 7, "tipo_dia": True, "frota": False},
    "7d_tipo_frota": {"janela": 7, "tipo_dia": True, "frota": True},
    # a referencia por tipo de dia e a janela de 7 dias perderam nos 3 dias de
    # desenvolvimento (21-23/10); a frota ganhou nos 3 sobre 7d_tipo
    "todos_frota": {"janela": None, "tipo_dia": False, "frota": True},
}
DEV = ["2026-10-21", "2026-10-22", "2026-10-23"]
FINAL = [f"2026-11-0{d}" for d in range(2, 9)]

ap = argparse.ArgumentParser()
ap.add_argument("--dias", default=",".join(DEV))
ap.add_argument("--final", action="store_true", help="semana intocada 02-08/11")
ap.add_argument("--configs", default=",".join(CONFIGS))
ap.add_argument("--redes", action="store_true", help="LSTM, GNN e conjunto alem do XGBoost")
args = ap.parse_args()
dias_teste = FINAL if args.final else args.dias.split(",")
configs = {k: CONFIGS[k] for k in args.configs.split(",")}

# ------------------------------------------------------------------ dados
pc = pd.read_parquet(DERIV / "percursos.parquet")
pc = pc.dropna(subset=["tempo_percurso_s"]).reset_index(drop=True)
pc["_dia"] = pd.to_datetime(pc["t_origem"], utc=True).dt.tz_convert(FUSO).dt.strftime("%Y-%m-%d")
previsoes = pd.read_parquet(DERIV / "previsao_realizado.parquet",
                            columns=["cl", "prefixo", "cp", "ts_coleta", "t_previsto"])
log.info("percursos: %d | dias: %s", len(pc), dict(pc["_dia"].value_counts().sort_index()))

arq_frota = DERIV / "frota_atributos.parquet"
if any(c["frota"] for c in configs.values()):
    if arq_frota.exists() and len(pd.read_parquet(arq_frota, columns=["viagem_id"])) == len(pc):
        fa = pd.read_parquet(arq_frota)
    else:
        inicio = time.perf_counter()
        medidas = frota.ler(RAIZ)
        partes = []
        for dia, grupo in pc.groupby("_dia"):
            d0 = pd.Timestamp(dia, tz=FUSO).tz_convert("UTC")
            m = medidas[(medidas["t"] >= d0 - pd.Timedelta("30min")) & (medidas["t"] < d0 + pd.Timedelta("1D"))]
            partes.append(frota.atributos(grupo, m))
        fa = pd.concat(partes).reindex(pc.index)
        fa.insert(0, "viagem_id", pc["viagem_id"].to_numpy())
        fa.to_parquet(arq_frota)
        log.info("atributos da frota: %.0f s | cobertura %s", time.perf_counter() - inicio,
                 {c: f"{fa[c].notna().mean():.0%}" for c in frota.COLUNAS_FROTA})
    fa = fa.drop(columns="viagem_id")


def ajustar(d: pd.DataFrame, tr, va, te) -> np.ndarray:
    colunas = [c for c in COLUNAS_MODELO + vizinhanca.COLUNAS_VIZINHANCA
               + contexto_viagem.COLUNAS_CONTEXTO + frota.COLUNAS_FROTA if c in d.columns]
    X = d[colunas].apply(pd.to_numeric, errors="coerce")
    reg = xgb.XGBRegressor(
        objective="reg:absoluteerror", eval_metric="mae", n_estimators=1500, max_depth=6,
        learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        n_jobs=-1, early_stopping_rounds=50,
    )
    reg.fit(X.loc[tr], d.loc[tr, "alvo_s"], eval_set=[(X.loc[va], d.loc[va, "alvo_s"])], verbose=False)
    return reg.predict(X.loc[te])


def rodar(nome: str, cfg: dict, dia: str) -> dict:
    arq = CACHE / f"{nome}_{dia}{'_redes' if args.redes else ''}.pkl"
    if arq.exists():
        return pickle.loads(arq.read_bytes())
    inicio = time.perf_counter()
    d_test = date.fromisoformat(dia)
    dias = sorted(x for x in pc["_dia"].unique() if x < dia)
    if cfg["janela"]:
        dias = [x for x in dias if date.fromisoformat(x) >= d_test - timedelta(days=cfg["janela"])]
    i_tr = pc.index[pc["_dia"].isin(dias)]
    i_te = pc.index[pc["_dia"] == dia]
    treino, validacao = modelos.separar_validacao(pc.loc[i_tr].sort_values("t_origem"), "t_origem")
    usar = pc.loc[i_tr.union(i_te)]
    d = modelos._atributos_atraso(usar, treino.index, 0.20, por_tipo_dia=cfg["tipo_dia"])
    d = d.dropna(subset=["alvo_s"])
    d = d.join(vizinhanca.atributos(d, previsoes)).join(contexto_viagem.atributos(d))
    if cfg["frota"]:
        d = d.join(fa)
    tr, va, te = (d.index.intersection(x) for x in (treino.index, validacao.index, i_te))
    p = ajustar(d, tr, va, te)
    r = {"indice": te, "alvo": d.loc[te, "alvo_s"].to_numpy(), "previsto": p,
         "previsto_por_modelo": {"xgboost": p},
         "recente": pd.to_numeric(d.loc[te, "desvio_recente_trecho_s"], errors="coerce").fillna(0).to_numpy(),
         "dias_treino": dias, "n_treino": len(tr)}
    if args.redes:
        base = {"df": d, "treino": tr, "validacao": va, "teste": te, "semana": pc["percurso_tipico_s"],
                "sem_referencia": 0, "limiar_percentual": 0.20}
        lstm = modelos_atraso.treinar_lstm_atraso(base=base, sementes=(1, 2, 3), variantes=("completa",))
        gnn = modelos_atraso.treinar_gnn_atraso(base=base, sementes=(1, 2, 3), variantes=("gcn",))
        r["previsto_por_modelo"]["lstm"] = np.mean([x["q"][:, 1] for x in lstm["rodadas"]["completa"]], axis=0)
        r["previsto_por_modelo"]["gnn"] = np.mean([x["q"][:, 1] for x in gnn["rodadas"]["gcn"]], axis=0)
        r["previsto_por_modelo"]["conjunto"] = np.mean(list(r["previsto_por_modelo"].values()), axis=0)
        r["quantis_lstm"] = np.mean([x["q"] for x in lstm["rodadas"]["completa"]], axis=0)
    arq.write_bytes(pickle.dumps(r))
    log.info("%s %s: treino %d trechos (%s a %s), %.0f s", nome, dia, len(tr), dias[0], dias[-1],
             time.perf_counter() - inicio)
    return r


linhas = []
for dia in dias_teste:
    res = {nome: rodar(nome, cfg, dia) for nome, cfg in configs.items()}
    comum = None
    for r in res.values():
        comum = r["indice"] if comum is None else comum.intersection(r["indice"])
    real = pc.loc[comum, "tempo_percurso_s"].to_numpy("float64")
    curto = (pc.loc[comum, "tempo_percurso_s"].groupby(
        [pc.loc[comum, "cl"], pc.loc[comum, "ordem_origem"]]).transform("median") <= 180).to_numpy()
    for nome, r in res.items():
        pos = pd.Index(r["indice"]).get_indexer(comum)
        erro = np.abs(r["alvo"][pos] - r["previsto"][pos])
        extra = {f"mae_{m}_s": np.abs(r["alvo"][pos] - v[pos]).mean()
                 for m, v in r.get("previsto_por_modelo", {}).items() if m != "xgboost"}
        linhas.append({
            **extra,
            "dia": dia, "config": nome, "trechos": len(comum),
            "cobertura": len(comum) / (pc["_dia"] == dia).sum(),
            "mae_s": erro.mean(), "erro_rel": erro.sum() / real.sum(),
            "mae_curtos_s": erro[curto].mean(),
            "trivial_s": np.abs(r["alvo"][pos]).mean(),
            "recente_s": np.abs(r["alvo"][pos] - r["recente"][pos]).mean(),
            "n_treino": r["n_treino"],
        })

t = pd.DataFrame(linhas)
print("\n| Dia | Configuracao | Trechos | MAE (s) | Erro relativo | MAE trechos <= 3 min | Trivial | Recente |")
print("|---|---|---|---|---|---|---|---|")
for l in t.itertuples():
    print(f"| {l.dia[5:]} | {l.config} | {l.trechos} | {l.mae_s:.2f} | {l.erro_rel:.1%} | "
          f"{l.mae_curtos_s:.2f} | {l.trivial_s:.2f} | {l.recente_s:.2f} |")
media = t.groupby("config")[[c for c in t.columns if c.startswith("mae") or c in ("erro_rel", "trivial_s")]].mean()
print("\nmedia dos dias:\n" + media.round(4).to_string())
ref = t[t.config == list(configs)[0]].set_index("dia")["mae_s"]
for nome in configs:
    dif = (ref - t[t.config == nome].set_index("dia")["mae_s"])
    print(f"{nome}: ganho sobre {list(configs)[0]} por dia {dif.round(2).to_dict()} | "
          f"venceu em {(dif > 0).sum()} de {len(dif)} dias")
saida = DERIV / "modelos" / ("avaliacao_final.json" if args.final else "avaliacao_dev.json")
saida.write_text(t.to_json(orient="records", indent=1, force_ascii=False), encoding="utf-8")
print("JSON:", saida)
