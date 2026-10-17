"""
Modelos preditivos de atraso.

Tres abordagens complementares, cada uma com um vies util:

* **XGBoost** — tabular, captura interacoes nao lineares entre horario,
  horizonte de previsao, posicao no itinerario e headway. Alem da regressao
  pontual, treina uma bateria de **regressoes quantilicas** (p10/p50/p90): e o
  que transforma "o atraso medio e 4 min" em "ha 10 % de chance de passar de
  12 min" — que e a leitura probabilistica que o estudo pede.
* **LSTM** — sequencial, aprende a dinamica de propagacao ao longo do dia numa
  mesma parada (o atraso das 17h30 depende do que aconteceu as 17h00).
* **GNN** — relacional, aprende que o atraso de uma parada depende das paradas
  vizinhas e das que vem antes no itinerario. E o unico que enxerga a rede.

Regra de avaliacao: **divisao temporal, nunca aleatoria**. Com 7 dias, treina
nos 5 primeiros e valida nos 2 ultimos. Split aleatorio vaza informacao do
futuro (o mesmo veiculo, a mesma viagem, aparece dos dois lados) e produz
metrica boa demais para ser verdade.

Tres regras que decorrem dessa:

* **o teste so e olhado no fim.** Parada antecipada, escolha da melhor epoca e
  limiar de decisao usam uma fatia de validacao tirada do FIM do treino
  (`separar_validacao`).
* **atributo so entra se for conhecido no instante da previsao.** Nada que
  dependa da chegada que se quer prever (tempo de percurso, velocidade, headway
  ou horario do proprio veiculo), nem estatistica calculada com o periodo de
  teste — inclusive a referencia contra a qual o alvo e medido.
* **toda metrica vem com as linhas de base que ela precisa superar**, tambem
  calculadas so com o treino. Superar so a mais fraca nao prova nada.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .features import COLUNAS_MODELO, FRACAO_TREINO, NS

log = logging.getLogger("olhovivo.modelos")

# faixas de antecedencia para estratificar o erro da previsao da API
BORDAS_HORIZONTE = [0, 300, 600, 1200, 1800, 3600]
FAIXAS_HORIZONTE = ["0-5min", "5-10min", "10-20min", "20-30min", "30-60min"]


def divisao_temporal(
    df: pd.DataFrame,
    coluna_tempo: str = "t_chegada",
    fracao_treino: float = FRACAO_TREINO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.sort_values(coluna_tempo)
    corte = int(len(d) * fracao_treino)
    return d.iloc[:corte], d.iloc[corte:]


def separar_validacao(
    treino: pd.DataFrame, coluna_tempo: str, fracao_validacao: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fatia final do treino, para parada antecipada sem espiar o teste."""
    return divisao_temporal(treino, coluna_tempo, 1 - fracao_validacao)


def _metricas(y: np.ndarray, previsto: np.ndarray) -> dict:
    """MAE, RMSE e MAPE, conforme o protocolo de avaliacao do estudo."""
    erro = previsto - y
    # MAPE e indefinido em y=0 e explode perto de zero; restringe-se aos casos
    # com magnitude minima e o denominador efetivo fica declarado no retorno
    relevante = np.abs(y) >= 1.0
    mape = (
        float(np.mean(np.abs(erro[relevante] / y[relevante])) * 100)
        if relevante.any() else float("nan")
    )
    return {
        "n": int(len(y)),
        "mae_s": float(np.mean(np.abs(erro))),
        "rmse_s": float(np.sqrt(np.mean(erro**2))),
        "mape_pct": mape,
        "n_mape": int(relevante.sum()),
        "vies_s": float(np.mean(erro)),
        "r2": float(1 - np.sum(erro**2) / max(np.sum((y - y.mean()) ** 2), 1e-9)),
    }


def metricas_classificacao(y: np.ndarray, previsto: np.ndarray) -> dict:
    """
    Acuracia, precisao, revocacao e F1 a partir da matriz de confusao.

    Calculadas explicitamente, e nao por biblioteca, para que a matriz fique
    no resultado — com classes desbalanceadas, acuracia sozinha engana.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(previsto).astype(int)
    vp = int(np.sum((p == 1) & (y == 1)))
    vn = int(np.sum((p == 0) & (y == 0)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    precisao = vp / (vp + fp) if (vp + fp) else 0.0
    revocacao = vp / (vp + fn) if (vp + fn) else 0.0
    f1 = (
        2 * precisao * revocacao / (precisao + revocacao)
        if (precisao + revocacao) else 0.0
    )
    return {
        "n": int(len(y)),
        "taxa_base": float(y.mean()),
        "acuracia": (vp + vn) / max(len(y), 1),
        "precisao": precisao,
        "revocacao": revocacao,
        "f1": f1,
        "matriz_confusao": {"vp": vp, "fp": fp, "vn": vn, "fn": fn},
    }


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def _pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y, score))
    except Exception:
        return float("nan")


def _metricas_probabilidade(y: np.ndarray, prob: np.ndarray, taxa_treino: float) -> dict:
    """
    Qualidade da probabilidade prevista, alem da ordenacao.

    AUC mede so se os atrasados recebem nota maior; PR-AUC olha a classe
    positiva, que e a minoria. `brier_skill` compara com prever sempre a taxa
    de atraso do treino: 0 e nao acrescentar nada a ela, negativo e piorar.
    """
    y = np.asarray(y, dtype="float64")
    prob = np.asarray(prob, dtype="float64")
    brier = float(np.mean((prob - y) ** 2))
    referencia = float(np.mean((taxa_treino - y) ** 2))
    return {
        "auc": _auc(y, prob),
        "pr_auc": _pr_auc(y, prob),
        "brier": brier,
        "brier_taxa_treino": referencia,
        "brier_skill": float(1 - brier / referencia) if referencia > 0 else float("nan"),
    }


def _limiar_f1(y: np.ndarray, prob: np.ndarray) -> float:
    """
    Limiar de decisao que maximiza o F1 na VALIDACAO (nunca no teste). Com a
    classe positiva minoritaria, o 0,5 padrao deixa o classificador quase mudo.
    """
    melhor, limiar = -1.0, 0.5
    for t in np.linspace(0.02, 0.98, 49):
        f1 = metricas_classificacao(y, (prob >= t).astype("int8"))["f1"]
        if f1 > melhor:
            melhor, limiar = f1, float(t)
    return limiar


def _classificacao(
    ctr: np.ndarray, cva: np.ndarray, cte: np.ndarray,
    prob_va: np.ndarray, prob_te: np.ndarray,
) -> dict:
    """Classificacao no limiar da validacao, com as referencias triviais."""
    limiar = _limiar_f1(cva, prob_va)
    return {
        "limiar_validacao": round(limiar, 2),
        **metricas_classificacao(cte, (prob_te >= limiar).astype("int8")),
        **_metricas_probabilidade(cte, prob_te, float(np.mean(ctr))),
        # marcar tudo como atrasado: o F1 de quem so conhece a taxa
        "f1_sempre_atrasado": metricas_classificacao(cte, np.ones_like(cte))["f1"],
    }


# ------------------------------------------------------------------ XGBoost
def _faixa_horizonte(horizonte: pd.Series) -> np.ndarray:
    return pd.cut(horizonte, bins=BORDAS_HORIZONTE, labels=FAIXAS_HORIZONTE).to_numpy()


def treinar_xgboost(
    df: pd.DataFrame,
    colunas: Sequence[str] | None = None,
    quantis: Sequence[float] = (0.1, 0.5, 0.9),
    limiar_atraso_s: int = 300,
) -> dict:
    """
    Erro da previsao da API (chegada real menos prevista), por consulta.

    Linhas de base, todas so com o treino: a API como esta (erro zero), a
    correcao do vies medio (mediana do erro) e a correcao por faixa de
    antecedencia. A ultima e a que importa: o erro cresce com a antecedencia,
    e so corrigir isso ja tira boa parte do erro da API.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("instale: pip install xgboost") from exc

    colunas = [c for c in (colunas or COLUNAS_MODELO) if c in df.columns]
    treino, teste = divisao_temporal(df)
    treino, validacao = separar_validacao(treino, "t_chegada")
    Xtr, ytr = treino[colunas], treino["alvo_erro_s"].to_numpy("float32")
    Xva, yva = validacao[colunas], validacao["alvo_erro_s"].to_numpy("float32")
    Xte, yte = teste[colunas], teste["alvo_erro_s"].to_numpy("float32")

    mediana_treino = float(np.median(ytr))
    if "horizonte_s" in df.columns:
        por_faixa = pd.Series(ytr).groupby(_faixa_horizonte(treino["horizonte_s"])).median()
        base_horizonte = (
            pd.Series(_faixa_horizonte(teste["horizonte_s"]))
            .map(por_faixa)
            .astype("float64")
            .fillna(mediana_treino)
            .to_numpy("float32")
        )
    else:
        base_horizonte = np.full_like(yte, mediana_treino)
    del treino, validacao

    comum = dict(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    regressor = xgb.XGBRegressor(objective="reg:squarederror", **comum)
    regressor.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    previsto = regressor.predict(Xte)

    # quantis -> leitura probabilistica do atraso. O p50 e tambem o previsor de
    # menor erro absoluto: e ele que se compara com as medianas de base
    quantis_resumo: dict[str, dict] = {}
    modelos_q: dict[str, Any] = {}
    previsto_q: dict[float, np.ndarray] = {}
    for q in quantis:
        mq = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q, **comum
        )
        mq.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pq = mq.predict(Xte)
        previsto_q[q] = pq
        modelos_q[f"p{int(q * 100)}"] = mq
        quantis_resumo[f"p{int(q * 100)}"] = {
            "cobertura": float(np.mean(yte <= pq)),
            "mediana_s": float(np.median(pq)),
        }

    zeros = np.zeros_like(yte)
    resultado: dict[str, Any] = {
        "colunas": list(colunas),
        "baselines": {
            "api_erro_zero": _metricas(yte, zeros),
            "mediana_treino": _metricas(yte, np.full_like(yte, mediana_treino)),
            "mediana_por_horizonte_treino": _metricas(yte, base_horizonte),
        },
        "regressao": _metricas(yte, previsto),
    }
    previsoes = {"modelo": previsto, "api": zeros, "mediana_horizonte": base_horizonte}
    if 0.5 in previsto_q:
        resultado["regressao_p50"] = _metricas(yte, previsto_q[0.5])
        previsoes["p50"] = previsto_q[0.5]
    melhor_mae = min(
        resultado["regressao"]["mae_s"],
        resultado.get("regressao_p50", {}).get("mae_s", float("inf")),
    )
    base_mae = resultado["baselines"]["mediana_por_horizonte_treino"]["mae_s"]
    resultado["ganho_mae_sobre_mediana_por_horizonte"] = round(
        float(1 - melhor_mae / max(base_mae, 1e-9)), 4
    )
    resultado["importancia"] = dict(
        sorted(
            zip(colunas, regressor.feature_importances_.astype(float)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    resultado["por_horizonte"] = _erro_por_horizonte(teste, previsoes)
    resultado["quantis"] = quantis_resumo

    # probabilidade de atraso relevante
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", **comum
    )
    ctr = (ytr >= limiar_atraso_s).astype("int8")
    cva = (yva >= limiar_atraso_s).astype("int8")
    cte = (yte >= limiar_atraso_s).astype("int8")
    clf.fit(Xtr, ctr, eval_set=[(Xva, cva)], verbose=False)
    resultado["classificacao"] = _classificacao(
        ctr, cva, cte, clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    )

    return {
        "resultado": resultado,
        "modelos": {"regressor": regressor, "classificador": clf, **modelos_q},
    }


def _erro_por_horizonte(teste: pd.DataFrame, previsoes: dict[str, np.ndarray]) -> dict:
    """Erro absoluto medio por faixa de antecedencia, do modelo e das bases."""
    if "horizonte_s" not in teste:
        return {}
    faixas = _faixa_horizonte(teste["horizonte_s"])
    y = teste["alvo_erro_s"].to_numpy("float32")
    saida = {}
    for faixa in FAIXAS_HORIZONTE:
        m = faixas == faixa
        if not m.any():
            continue
        saida[faixa] = {
            "n": int(m.sum()),
            **{
                f"mae_{nome}_s": round(float(np.mean(np.abs(p[m] - y[m]))), 1)
                for nome, p in previsoes.items()
            },
        }
    return saida


# ---------------------------------------------------------------------- LSTM
def treinar_lstm(
    caminho_npz,
    epocas: int = 40,
    oculto: int = 64,
    camadas: int = 2,
    lote: int = 256,
    paciencia: int = 6,
) -> dict:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "instale: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    dados = np.load(caminho_npz, allow_pickle=True)
    X, y, bins = dados["X"], dados["y"], dados["bin"]

    ordem = np.argsort(bins)  # divisao temporal
    X, y = X[ordem], y[ordem]
    corte = int(len(X) * FRACAO_TREINO)
    # parada antecipada na fatia final do treino; o teste so e usado no fim
    corte_val = int(corte * 0.85)

    # linhas de base: o proximo bin repete o ultimo (persistencia) ou a media
    # dos ultimos bins da janela. A media da janela e a mais dificil de bater
    y_teste = y[corte:]
    baselines = {
        "persistencia": _metricas(y_teste, X[corte:, -1, 0]),
        "media_3_bins": _metricas(y_teste, X[corte:, -3:, 0].mean(1)),
        "media_janela": _metricas(y_teste, X[corte:, :, 0].mean(1)),
    }

    mu, sigma = X[:corte_val].mean((0, 1)), X[:corte_val].std((0, 1)) + 1e-6
    X = (X - mu) / sigma
    y_mu, y_sigma = float(y[:corte_val].mean()), float(y[:corte_val].std() + 1e-6)

    tX = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor((y - y_mu) / y_sigma, dtype=torch.float32).unsqueeze(1)

    class Rede(nn.Module):
        def __init__(self, entradas: int):
            super().__init__()
            self.lstm = nn.LSTM(entradas, oculto, camadas, batch_first=True, dropout=0.1)
            self.saida = nn.Linear(oculto, 1)

        def forward(self, x):
            h, _ = self.lstm(x)
            return self.saida(h[:, -1, :])

    modelo = Rede(tX.shape[-1])
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    perda_fn = nn.SmoothL1Loss()

    treino = torch.utils.data.TensorDataset(tX[:corte_val], ty[:corte_val])
    carregador = torch.utils.data.DataLoader(treino, batch_size=lote, shuffle=True)

    melhor, espera, melhor_estado = float("inf"), 0, None
    for epoca in range(epocas):
        modelo.train()
        for xb, yb in carregador:
            otimizador.zero_grad()
            perda = perda_fn(modelo(xb), yb)
            perda.backward()
            otimizador.step()

        modelo.eval()
        with torch.no_grad():
            val = perda_fn(modelo(tX[corte_val:corte]), ty[corte_val:corte]).item()
        if val < melhor - 1e-4:
            melhor, espera = val, 0
            melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            espera += 1
            if espera >= paciencia:
                break
        log.info("LSTM epoca %d: val=%.4f", epoca, val)

    if melhor_estado:
        modelo.load_state_dict(melhor_estado)
    modelo.eval()
    with torch.no_grad():
        previsto = modelo(tX[corte:]).squeeze(1).numpy() * y_sigma + y_mu

    metricas = _metricas(y_teste, previsto)
    melhor_base = min(b["mae_s"] for b in baselines.values())
    return {
        "resultado": {
            **metricas,
            "baselines": baselines,
            "ganho_mae_sobre_melhor_baseline": round(
                float(1 - metricas["mae_s"] / max(melhor_base, 1e-9)), 4
            ),
        },
        "modelo": modelo,
    }


# ----------------------------------------------------------------------- GNN
def treinar_gnn(
    caminho_npz,
    epocas: int = 300,
    oculto: int = 64,
    sementes: Sequence[int] = (42, 43, 44, 45, 46),
) -> dict:
    """
    Regressao de no sobre o grafo da rede: o erro mediano de cada parada no fim
    do periodo, a partir de atributos do inicio (`features.montar_grafo`).

    GCN em PyTorch puro (propagacao com adjacencia normalizada por grau), que
    nesta escala da o mesmo resultado do PyTorch Geometric sem a instalacao
    complicada no Windows.

    Formulacao residual: a rede preve a CORRECAO sobre a persistencia (o erro
    mediano que a parada ja tinha). Com saida zero ela reproduz a persistencia,
    que sozinha ja explica a maior parte do alvo; o que a rede acrescentar e o
    ganho. A mesma rede roda tambem sem arestas (so o laco proprio, um MLP por
    no), para separar o efeito do grafo do efeito de ter um modelo. Com ~350
    paradas, uma divisao so deixa ~110 no teste e o numero oscila: repete-se a
    divisao com varias sementes e reporta-se media e desvio.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("instale o torch para treinar a GNN") from exc

    dados = np.load(caminho_npz, allow_pickle=True)
    x_bruto = torch.tensor(dados["x"], dtype=torch.float32)
    y = torch.tensor(dados["y"], dtype=torch.float32)
    edge_index = torch.tensor(dados["edge_index"], dtype=torch.long)

    nomes = [str(a) for a in dados["atributos"]]
    persistencia = x_bruto[:, nomes.index("erro_mediano")].clone()
    x = (x_bruto - x_bruto.mean(0)) / (x_bruto.std(0) + 1e-6)
    residuo = y - persistencia

    # so nos com chegada no periodo do alvo entram em treino/validacao/teste;
    # os demais seguem no grafo, propagando informacao
    n = x.shape[0]
    com_alvo = torch.nonzero(~torch.isnan(y)).squeeze(1)
    if len(com_alvo) < 3:
        raise RuntimeError(f"so {len(com_alvo)} paradas com alvo — pouco para a GNN")

    def adjacencia(ei: torch.Tensor) -> torch.Tensor:
        # adjacencia normalizada com laco proprio (GCN de Kipf & Welling)
        lacos = torch.arange(n).repeat(2, 1)
        ei = torch.cat([ei, lacos], dim=1) if ei.numel() else lacos
        valores = torch.ones(ei.shape[1])
        grau = torch.zeros(n).index_add_(0, ei[0], valores).clamp(min=1)
        norm = grau[ei[0]].pow(-0.5) * grau[ei[1]].pow(-0.5)
        return torch.sparse_coo_tensor(ei, norm, (n, n)).coalesce()

    variantes = {
        "gcn": adjacencia(edge_index),
        # so o laco proprio: a mesma rede sem enxergar vizinho nenhum
        "sem_grafo": adjacencia(torch.zeros((2, 0), dtype=torch.long)),
    }

    class GCN(nn.Module):
        def __init__(self, entradas: int, A: torch.Tensor):
            super().__init__()
            self.A = A
            self.l1 = nn.Linear(entradas, oculto)
            self.l2 = nn.Linear(oculto, oculto)
            self.l3 = nn.Linear(oculto, 1)
            self.drop = nn.Dropout(0.2)
            # saida zero no inicio: o treino parte exatamente da persistencia
            nn.init.zeros_(self.l3.weight)
            nn.init.zeros_(self.l3.bias)

        def forward(self, x):
            h = torch.relu(torch.sparse.mm(self.A, self.l1(x)))
            h = self.drop(h)
            h = torch.relu(torch.sparse.mm(self.A, self.l2(h)))
            return self.l3(h).squeeze(1)

    perda_fn = nn.SmoothL1Loss()
    rodadas: dict[str, list[dict]] = {"gcn": [], "sem_grafo": [], "persistencia": []}
    modelo = None
    teste = com_alvo
    for semente in sementes:
        gerador = torch.Generator().manual_seed(semente)
        perm = com_alvo[torch.randperm(len(com_alvo), generator=gerador)]
        corte = int(len(perm) * FRACAO_TREINO)
        corte_val = max(int(corte * 0.85), 1)
        treino, validacao, teste = perm[:corte_val], perm[corte_val:corte], perm[corte:]
        if len(validacao) == 0:
            validacao = treino

        mu = residuo[treino].mean()
        sd = residuo[treino].std(unbiased=False)
        sd = sd if float(sd) > 1e-6 else torch.tensor(1.0)
        alvo = torch.nan_to_num((residuo - mu) / sd)

        y_teste = y[teste].numpy()
        base = persistencia[teste].numpy()
        rodadas["persistencia"].append(_metricas(y_teste, base))

        for nome, A in variantes.items():
            torch.manual_seed(semente)
            modelo = GCN(x.shape[1], A)
            otimizador = torch.optim.Adam(modelo.parameters(), lr=5e-3, weight_decay=1e-4)
            melhor, melhor_estado = float("inf"), None
            for epoca in range(epocas):
                modelo.train()
                otimizador.zero_grad()
                perda = perda_fn(modelo(x)[treino], alvo[treino])
                perda.backward()
                otimizador.step()

                if epoca % 10 == 0:
                    modelo.eval()
                    with torch.no_grad():
                        val = perda_fn(modelo(x)[validacao], alvo[validacao]).item()
                    if val < melhor:
                        melhor = val
                        melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}

            if melhor_estado:
                modelo.load_state_dict(melhor_estado)
            modelo.eval()
            with torch.no_grad():
                correcao = (modelo(x)[teste] * sd + mu).numpy()
            rodadas[nome].append(_metricas(y_teste, base + correcao))

        log.info(
            "GNN semente %d: MAE gcn %.1f | sem grafo %.1f | persistencia %.1f",
            semente, rodadas["gcn"][-1]["mae_s"], rodadas["sem_grafo"][-1]["mae_s"],
            rodadas["persistencia"][-1]["mae_s"],
        )

    def resumo(lista: list[dict]) -> dict:
        mae = np.array([r["mae_s"] for r in lista])
        r2 = np.array([r["r2"] for r in lista])
        return {
            "mae_s_media": float(mae.mean()),
            "mae_s_dp": float(mae.std()),
            "r2_media": float(r2.mean()),
            "mae_s_por_semente": [round(float(v), 2) for v in mae],
        }

    resumos = {nome: resumo(lista) for nome, lista in rodadas.items()}
    return {
        "resultado": {
            "sementes": len(sementes),
            "n_teste_por_semente": int(len(teste)),
            "mae_s": resumos["gcn"]["mae_s_media"],
            **resumos,
        },
        "modelo": modelo,
    }


# ------------------------------------------------- alvo do estudo: atraso
JANELA_RECENTE = pd.Timedelta("30min")


def _referencia_treino(df: pd.DataFrame, indice_treino: pd.Index) -> pd.Series:
    """Tempo mediano do trecho, sentido e faixa, so com o treino."""
    treino = df.loc[indice_treino]
    fina = treino.groupby(["cl", "ordem_origem", "faixa"])["tempo_percurso_s"].median()
    grossa = treino.groupby(["cl", "ordem_origem"])["tempo_percurso_s"].median()
    ref = df.join(fina.rename("_ref"), on=["cl", "ordem_origem", "faixa"])["_ref"]
    # faixa sem esse trecho no treino: a mediana do trecho em qualquer faixa
    reserva = df.join(grossa.rename("_ref"), on=["cl", "ordem_origem"])["_ref"]
    return ref.fillna(reserva)


def _desvio_recente(df: pd.DataFrame) -> pd.DataFrame:
    """
    O transito no trecho agora: desvio dos veiculos que COMPLETARAM o mesmo
    trecho antes de `t_origem`.

    Mediana e contagem numa janela de 30 min terminada na conclusao mais
    recente, e essa conclusao tem de ser anterior a `t_origem` (merge_asof
    estrito) e ter no maximo 30 min. Mais o desvio do ultimo veiculo e ha
    quanto tempo ele passou.
    """
    ev = (
        df[["cl", "ordem_origem", "t_chegada", "_desvio"]]
        .dropna()
        .astype({"t_chegada": NS})
        .sort_values(["cl", "ordem_origem", "t_chegada"])
        .reset_index(drop=True)
    )
    # groupby ordena as chaves como o sort acima: a saida vem na ordem de `ev`
    janela = (
        ev.set_index("t_chegada")
        .groupby(["cl", "ordem_origem"])["_desvio"]
        .rolling(JANELA_RECENTE)
    )
    ev["desvio_recente_trecho_s"] = janela.median().to_numpy()
    ev["n_recente_trecho"] = janela.count().to_numpy()
    ev = ev.rename(
        columns={"t_chegada": "t_evento", "_desvio": "desvio_ultimo_veiculo_s"}
    ).sort_values("t_evento")

    consulta = (
        df[["cl", "ordem_origem", "t_origem"]]
        .astype({"t_origem": NS})
        .reset_index()
        .sort_values("t_origem")
    )
    m = pd.merge_asof(
        consulta, ev, left_on="t_origem", right_on="t_evento",
        by=["cl", "ordem_origem"], allow_exact_matches=False,
        tolerance=JANELA_RECENTE,
    )
    m["desde_ultimo_veiculo_s"] = (m["t_origem"] - m["t_evento"]).dt.total_seconds()
    return m.set_index("index")[[
        "desvio_recente_trecho_s", "n_recente_trecho",
        "desvio_ultimo_veiculo_s", "desde_ultimo_veiculo_s",
    ]]


def _atributos_atraso(
    df: pd.DataFrame, indice_treino: pd.Index, limiar_percentual: float
) -> pd.DataFrame:
    """
    Alvo e atributos de atraso que dependem da divisao treino/teste.

    * `percurso_tipico_s`: mediana do trecho, sentido e faixa so no TREINO. A de
      `percursos` e a mediana da semana inteira, que inclui o teste: descreve o
      conjunto, mas como referencia do alvo daria ao modelo, e ao trivial,
      informacao do periodo avaliado.
    * `alvo_s`: tempo de percurso menos essa referencia; `alvo_atrasado`: acima
      do limiar percentual sobre ela. Prever zero e o trivial do metodo.
    * `atraso_trecho_anterior_s`: desvio do trecho anterior da MESMA viagem.
    * `desvio_recente_trecho_s` e afins: ver `_desvio_recente`.

    Tudo e conhecido no instante `t_origem`.
    """
    df = df.copy()
    df["percurso_tipico_s"] = _referencia_treino(df, indice_treino)
    df["alvo_s"] = df["tempo_percurso_s"] - df["percurso_tipico_s"]
    df["alvo_atrasado"] = (
        df["alvo_s"] >= limiar_percentual * df["percurso_tipico_s"]
    ).astype("int8")
    df["_desvio"] = df["alvo_s"]

    ordenado = df.sort_values(["viagem_id", "ordem_origem"])
    por_viagem = ordenado.groupby("viagem_id", sort=False)
    anterior = por_viagem["_desvio"].shift(1)
    consecutivo = ordenado["ordem_origem"] - por_viagem["ordem_origem"].shift(1) == 1
    df["atraso_trecho_anterior_s"] = anterior.where(consecutivo)

    df = df.join(_desvio_recente(df))
    return df.drop(columns="_desvio")


def preparar_atraso(df: pd.DataFrame, limiar_percentual: float = 0.20) -> dict:
    """
    Base comum dos modelos de atraso: o alvo contra a referencia do treino, os
    atributos conhecidos em `t_origem` e a mesma divisao temporal. XGBoost,
    LSTM e GNN usam exatamente estas linhas, para que a comparacao entre eles
    meca o modelo, e nao diferencas de dados.

    O alvo e o desvio contra o tempo mediano historico do trecho, calculado so
    com o treino — a definicao do estudo, com a referencia restrita ao periodo
    que o modelo pode conhecer, como o metodo manda fazer com toda normalizacao.
    A `atraso_s` de `percursos` usa a mediana da semana inteira: como alvo, ela
    daria aos modelos e ao trivial uma referencia vinda do teste. Ela fica
    guardada (`semana`) so para comparacao.
    """
    df = df.dropna(subset=["tempo_percurso_s"]).reset_index(drop=True)
    treino, teste = divisao_temporal(df, "t_origem")
    if len(teste) < 30:
        raise RuntimeError(
            f"conjunto de teste com apenas {len(teste)} trechos — colete mais dias"
        )
    treino, validacao = separar_validacao(treino, "t_origem")

    semana = df["percurso_tipico_s"]
    df = _atributos_atraso(df, treino.index, limiar_percentual)
    sem_referencia = int(df["alvo_s"].isna().sum())
    df = df.dropna(subset=["alvo_s"])
    return {
        "df": df,
        "treino": df.index.intersection(treino.index),
        "validacao": df.index.intersection(validacao.index),
        "teste": df.index.intersection(teste.index),
        "semana": semana,
        "sem_referencia": sem_referencia,
        "limiar_percentual": limiar_percentual,
    }


def baselines_atraso(base: dict) -> dict:
    """As linhas de base de todo modelo de atraso, no conjunto de teste."""
    teste = base["df"].loc[base["teste"]]
    yte = teste["alvo_s"].to_numpy("float32")
    zeros = np.zeros_like(yte)
    recente = (
        pd.to_numeric(teste["desvio_recente_trecho_s"], errors="coerce")
        .fillna(0).to_numpy("float32")
        if "desvio_recente_trecho_s" in teste else zeros
    )
    semana_prevista = (
        base["semana"].loc[teste.index] - teste["percurso_tipico_s"]
    ).to_numpy("float32")
    return {
        # o trivial do metodo: o tempo mediano historico do trecho
        "trivial_mediana_historica": _metricas(yte, zeros),
        # o transito agora: o desvio dos veiculos que acabaram de passar
        "desvio_recente_do_trecho": _metricas(yte, recente),
        # a mediana da semana inteira INCLUI o teste: so para comparacao
        "mediana_semana_inclui_teste": _metricas(yte, semana_prevista),
    }


def treinar_atraso(
    df: pd.DataFrame, limiar_percentual: float = 0.20, base: dict | None = None
) -> dict:
    """
    XGBoost sobre a variavel de atraso do estudo (desvio de tempo de percurso),
    na base de `preparar_atraso`.

    Duas formulacoes, como o metodo pede: regressao (por erro absoluto, a
    metrica principal) e classificacao, com limiar escolhido na validacao.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("instale: pip install xgboost") from exc

    from .atraso import COLUNAS_MODELO as COLUNAS_ATRASO
    from .vizinhanca import COLUNAS_VIZINHANCA

    base = base or preparar_atraso(df, limiar_percentual)
    df = base["df"].copy()
    # os atributos de vizinhanca entram quando a base os traz (vizinhanca.atributos)
    colunas = [c for c in COLUNAS_ATRASO + COLUNAS_VIZINHANCA if c in df.columns]
    for c in colunas:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    treino = df.loc[base["treino"]]
    validacao = df.loc[base["validacao"]]
    teste = df.loc[base["teste"]]
    sem_referencia = base["sem_referencia"]

    Xtr, Xva, Xte = treino[colunas], validacao[colunas], teste[colunas]
    ytr = treino["alvo_s"].to_numpy("float32")
    yva = validacao["alvo_s"].to_numpy("float32")
    yte = teste["alvo_s"].to_numpy("float32")

    # min_child_weight e learning_rate escolhidos na validacao
    # (scripts/experimento_atributos.py)
    comum = dict(
        n_estimators=1500, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=20, n_jobs=-1,
        early_stopping_rounds=50,
    )

    # erro absoluto: e a metrica principal, e com o alvo assimetrico (mediana
    # perto de zero, cauda longa a direita) o erro quadratico puxaria a previsao
    # para a media e perderia do trivial em MAE so por isso
    reg = xgb.XGBRegressor(objective="reg:absoluteerror", eval_metric="mae", **comum)
    reg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    previsto = reg.predict(Xte)

    baselines = baselines_atraso(base)
    trivial = baselines["trivial_mediana_historica"]
    metricas_reg = _metricas(yte, previsto)
    ganho = 1 - metricas_reg["mae_s"] / max(trivial["mae_s"], 1e-9)

    resultado: dict[str, Any] = {
        "colunas": colunas,
        "limiar_percentual": base["limiar_percentual"],
        "trechos_sem_referencia_no_treino": sem_referencia,
        "baselines": baselines,
        "regressao": metricas_reg,
        "ganho_sobre_trivial": round(float(ganho), 4),
        "superou_trivial": bool(metricas_reg["mae_s"] < trivial["mae_s"]),
        "importancia": dict(
            sorted(zip(colunas, reg.feature_importances_.astype(float)),
                   key=lambda kv: kv[1], reverse=True)
        ),
    }

    ctr = treino["alvo_atrasado"].to_numpy("int8")
    cva = validacao["alvo_atrasado"].to_numpy("int8")
    cte = teste["alvo_atrasado"].to_numpy("int8")
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", **comum
    )
    clf.fit(Xtr, ctr, eval_set=[(Xva, cva)], verbose=False)
    prob_va, prob_te = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    resultado["classificacao"] = _classificacao(ctr, cva, cte, prob_va, prob_te)

    return {
        "resultado": resultado,
        "modelo": reg,
        # previsoes no teste, na ordem de base["teste"]: para comparar e combinar
        "previsoes": {"p50": previsto, "prob_validacao": prob_va, "prob_teste": prob_te},
    }
