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
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .features import COLUNAS_MODELO

log = logging.getLogger("olhovivo.modelos")


def divisao_temporal(
    df: pd.DataFrame, coluna_tempo: str = "t_chegada", fracao_treino: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.sort_values(coluna_tempo)
    corte = int(len(d) * fracao_treino)
    return d.iloc[:corte], d.iloc[corte:]


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


def baseline_trivial(df: pd.DataFrame, coluna_tempo: str = "t_origem") -> dict:
    """
    Referencia minima do estudo: prever o tempo mediano historico do trecho.

    Em termos da variavel de atraso, isso equivale a prever desvio ZERO. Todo
    modelo precisa superar esta linha para que se caracterize aprendizado — um
    MAE bonito sozinho nao prova nada se o trivial chega perto.
    """
    _, teste = divisao_temporal(df, coluna_tempo)
    y = teste["atraso_s"].to_numpy("float32")
    resultado = _metricas(y, np.zeros_like(y))
    if "atrasado" in teste:
        alvo = teste["atrasado"].to_numpy()
        resultado["classificacao"] = metricas_classificacao(
            alvo, np.zeros_like(alvo)
        )
    return resultado


# ------------------------------------------------------------------ XGBoost
def treinar_xgboost(
    df: pd.DataFrame,
    colunas: Sequence[str] | None = None,
    quantis: Sequence[float] = (0.1, 0.5, 0.9),
    limiar_atraso_s: int = 300,
) -> dict:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("instale: pip install xgboost") from exc

    colunas = [c for c in (colunas or COLUNAS_MODELO) if c in df.columns]
    treino, teste = divisao_temporal(df)
    Xtr, ytr = treino[colunas], treino["alvo_erro_s"].to_numpy("float32")
    Xte, yte = teste[colunas], teste["alvo_erro_s"].to_numpy("float32")

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
    regressor.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    previsto = regressor.predict(Xte)

    resultado: dict[str, Any] = {
        "colunas": list(colunas),
        "regressao": _metricas(yte, previsto),
        "importancia": dict(
            sorted(
                zip(colunas, regressor.feature_importances_.astype(float)),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
        "por_horizonte": _erro_por_horizonte(teste, previsto),
    }

    # quantis -> leitura probabilistica do atraso
    modelos_q = {}
    for q in quantis:
        mq = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q, **comum
        )
        mq.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        pq = mq.predict(Xte)
        modelos_q[f"p{int(q * 100)}"] = {
            "cobertura": float(np.mean(yte <= pq)),
            "mediana_s": float(np.median(pq)),
        }
    resultado["quantis"] = modelos_q

    # probabilidade de atraso relevante
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", **comum
    )
    ctr = (ytr >= limiar_atraso_s).astype("int8")
    cte = (yte >= limiar_atraso_s).astype("int8")
    clf.fit(Xtr, ctr, eval_set=[(Xte, cte)], verbose=False)
    prob = clf.predict_proba(Xte)[:, 1]
    resultado["classificacao"] = {
        "taxa_base": float(cte.mean()),
        "brier": float(np.mean((prob - cte) ** 2)),
        "auc": _auc(cte, prob),
    }

    return {"resultado": resultado, "modelos": {"regressor": regressor, "classificador": clf, **{f"q{k}": v for k, v in modelos_q.items()}}}


def _erro_por_horizonte(teste: pd.DataFrame, previsto: np.ndarray) -> dict:
    if "horizonte_s" not in teste:
        return {}
    faixas = pd.cut(
        teste["horizonte_s"],
        bins=[0, 300, 600, 1200, 1800, 3600],
        labels=["0-5min", "5-10min", "10-20min", "20-30min", "30-60min"],
    )
    saida = {}
    for faixa, idx in teste.groupby(faixas, observed=True).groups.items():
        pos = teste.index.get_indexer(idx)
        y = teste.loc[idx, "alvo_erro_s"].to_numpy("float32")
        saida[str(faixa)] = _metricas(y, previsto[pos])
    return saida


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


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
    corte = int(len(X) * 0.7)

    mu, sigma = X[:corte].mean((0, 1)), X[:corte].std((0, 1)) + 1e-6
    X = (X - mu) / sigma
    y_mu, y_sigma = float(y[:corte].mean()), float(y[:corte].std() + 1e-6)

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

    treino = torch.utils.data.TensorDataset(tX[:corte], ty[:corte])
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
            val = perda_fn(modelo(tX[corte:]), ty[corte:]).item()
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

    return {"resultado": _metricas(y[corte:], previsto), "modelo": modelo}


# ----------------------------------------------------------------------- GNN
def treinar_gnn(caminho_npz, epocas: int = 300, oculto: int = 64) -> dict:
    """
    Regressao de no sobre o grafo da rede.

    Usa PyTorch Geometric se estiver instalado; senao roda uma GCN em PyTorch
    puro (propagacao com adjacencia normalizada por grau), que da o mesmo
    resultado nesta escala e evita a instalacao complicada do PyG no Windows.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("instale o torch para treinar a GNN") from exc

    dados = np.load(caminho_npz, allow_pickle=True)
    x = torch.tensor(dados["x"], dtype=torch.float32)
    y = torch.tensor(dados["y"], dtype=torch.float32).unsqueeze(1)
    edge_index = torch.tensor(dados["edge_index"], dtype=torch.long)

    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y_mu, y_sigma = y.mean(), y.std() + 1e-6
    y_norm = (y - y_mu) / y_sigma

    n = x.shape[0]
    gerador = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=gerador)
    treino = perm[: int(n * 0.7)]
    teste = perm[int(n * 0.7) :]

    # adjacencia normalizada com laco proprio (GCN de Kipf & Welling)
    lacos = torch.arange(n).repeat(2, 1)
    ei = torch.cat([edge_index, lacos], dim=1) if edge_index.numel() else lacos
    valores = torch.ones(ei.shape[1])
    grau = torch.zeros(n).index_add_(0, ei[0], valores).clamp(min=1)
    norm = grau[ei[0]].pow(-0.5) * grau[ei[1]].pow(-0.5)
    A = torch.sparse_coo_tensor(ei, norm, (n, n)).coalesce()

    class GCN(nn.Module):
        def __init__(self, entradas: int):
            super().__init__()
            self.l1 = nn.Linear(entradas, oculto)
            self.l2 = nn.Linear(oculto, oculto)
            self.l3 = nn.Linear(oculto, 1)
            self.drop = nn.Dropout(0.2)

        def forward(self, x):
            h = torch.relu(torch.sparse.mm(A, self.l1(x)))
            h = self.drop(h)
            h = torch.relu(torch.sparse.mm(A, self.l2(h)))
            return self.l3(h)

    modelo = GCN(x.shape[1])
    otimizador = torch.optim.Adam(modelo.parameters(), lr=5e-3, weight_decay=1e-4)
    perda_fn = nn.SmoothL1Loss()

    melhor, melhor_estado = float("inf"), None
    for epoca in range(epocas):
        modelo.train()
        otimizador.zero_grad()
        perda = perda_fn(modelo(x)[treino], y_norm[treino])
        perda.backward()
        otimizador.step()

        if epoca % 10 == 0:
            modelo.eval()
            with torch.no_grad():
                val = perda_fn(modelo(x)[teste], y_norm[teste]).item()
            if val < melhor:
                melhor = val
                melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            log.info("GNN epoca %d: val=%.4f", epoca, val)

    if melhor_estado:
        modelo.load_state_dict(melhor_estado)
    modelo.eval()
    with torch.no_grad():
        previsto = (modelo(x)[teste] * y_sigma + y_mu).squeeze(1).numpy()

    return {
        "resultado": _metricas(y[teste].squeeze(1).numpy(), previsto),
        "modelo": modelo,
    }


# ------------------------------------------------- alvo do estudo: atraso
def treinar_atraso(df: pd.DataFrame, limiar_percentual: float = 0.20) -> dict:
    """
    Treina sobre a variavel de atraso do estudo (desvio de tempo de percurso).

    Duas formulacoes, como o metodo pede: regressao sobre `atraso_s` e
    classificacao sobre `atrasado`. O resultado inclui o baseline trivial, sem
    o qual nenhuma metrica de regressao significa alguma coisa aqui — prever
    desvio zero ja acerta a mediana.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("instale: pip install xgboost") from exc

    from .atraso import COLUNAS_MODELO as COLUNAS_ATRASO

    colunas = [c for c in COLUNAS_ATRASO if c in df.columns]
    df = df.dropna(subset=["atraso_s"]).copy()
    for c in colunas:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    treino, teste = divisao_temporal(df, "t_origem")
    if len(teste) < 30:
        raise RuntimeError(
            f"conjunto de teste com apenas {len(teste)} trechos — colete mais dias"
        )

    Xtr, Xte = treino[colunas], teste[colunas]
    ytr = treino["atraso_s"].to_numpy("float32")
    yte = teste["atraso_s"].to_numpy("float32")

    comum = dict(
        n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, n_jobs=-1,
        early_stopping_rounds=50,
    )

    reg = xgb.XGBRegressor(objective="reg:squarederror", **comum)
    reg.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    previsto = reg.predict(Xte)

    trivial = baseline_trivial(df, "t_origem")
    metricas_reg = _metricas(yte, previsto)
    ganho = 1 - metricas_reg["mae_s"] / max(trivial["mae_s"], 1e-9)

    resultado: dict[str, Any] = {
        "colunas": colunas,
        "limiar_percentual": limiar_percentual,
        "baseline_trivial": trivial,
        "regressao": metricas_reg,
        "ganho_sobre_trivial": round(float(ganho), 4),
        "superou_trivial": bool(metricas_reg["mae_s"] < trivial["mae_s"]),
        "importancia": dict(
            sorted(zip(colunas, reg.feature_importances_.astype(float)),
                   key=lambda kv: kv[1], reverse=True)
        ),
    }

    if "atrasado" in df.columns:
        ctr = treino["atrasado"].to_numpy("int8")
        cte = teste["atrasado"].to_numpy("int8")
        clf = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", **comum
        )
        clf.fit(Xtr, ctr, eval_set=[(Xte, cte)], verbose=False)
        prob = clf.predict_proba(Xte)[:, 1]
        resultado["classificacao"] = {
            **metricas_classificacao(cte, (prob >= 0.5).astype("int8")),
            "auc": _auc(cte, prob),
            "brier": float(np.mean((prob - cte) ** 2)),
        }
        resultado["classificacao_trivial"] = trivial.get("classificacao")

    return {"resultado": resultado, "modelo": reg}
