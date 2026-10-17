"""
LSTM e GNN sobre a variavel de atraso do estudo (desvio de tempo de percurso
por trecho), na MESMA base do XGBoost (`modelos.preparar_atraso`): mesmo alvo,
mesmas linhas de treino, validacao e teste, mesmas linhas de base. Assim a
comparacao entre os tres mede o modelo, e nao diferencas de dados.

* **LSTM com duas memorias.** Uma le os ultimos trechos da propria viagem (o
  atraso se propagando ao longo dela, como o metodo propoe); a outra, os
  ultimos veiculos que completaram o mesmo trecho (o transito ali agora). Roda
  tambem com cada memoria sozinha, para saber qual delas carrega a informacao.
* **GNN espaco-temporal.** Os nos sao os trechos fisicos (par de paradas),
  comuns a todas as linhas que passam por eles; as arestas ligam cada trecho ao
  de cima, ao de baixo e aos vizinhos a menos de 300 m, com pesos separados
  para cada tipo. A cada 5 min, o estado de congestionamento da rede inteira
  passa por duas convolucoes em grafo, e o atraso do onibus que sai agora e
  previsto a partir do estado do seu trecho e dos vizinhos, junto com os
  atributos da propria linha. Roda tambem sem arestas (so o proprio trecho),
  o que isola o ganho do grafo.

As duas redes saem com tres quantis (p10, p50, p90) e a probabilidade de
atraso, treinados juntos (perda quantilica + entropia cruzada). O p50 e a
previsao pontual, comparavel ao XGBoost por erro absoluto. Redes neurais variam
com a inicializacao: cada configuracao roda com varias sementes, e o resultado
traz media e desvio, alem da media das previsoes das sementes (o conjunto).

Nada aqui usa informacao posterior a `t_origem`: os trechos da viagem
terminaram ate a parada de origem, os veiculos do trecho completaram-no antes
dela, e o estado da rede de cada janela de 5 min so ve conclusoes anteriores ao
inicio da janela.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .features import NS
from .modelos import _classificacao, _metricas, baselines_atraso, preparar_atraso
from .vizinhanca import COLUNAS_VIZINHANCA

log = logging.getLogger("olhovivo.modelos_atraso")

QUANTIS = (0.1, 0.5, 0.9)
EPOCA = pd.Timestamp("1970-01-01", tz="UTC")
JANELA_TRECHO = pd.Timedelta("60min")


def _segundos(serie: pd.Series) -> np.ndarray:
    return ((serie.astype(NS) - EPOCA) / pd.Timedelta("1s")).to_numpy("float64")


def _relativo(df: pd.DataFrame) -> pd.Series:
    """Desvio em proporcao do tempo tipico: comparavel entre trechos."""
    tipico = df["percurso_tipico_s"].where(df["percurso_tipico_s"] > 0)
    return (df["alvo_s"] / tipico).clip(-3, 10)


# ------------------------------------------------------------- atributos
def _contexto(df: pd.DataFrame, com_desvios: bool) -> np.ndarray:
    """
    Atributos da propria linha, conhecidos em `t_origem`. `com_desvios` inclui
    os desvios recentes que o XGBoost ve; a LSTM recebe essa informacao pelas
    sequencias e fica com calendario e trecho. Os atributos de vizinhanca
    (`vizinhanca.COLUNAS_VIZINHANCA`) entram nos dois casos, quando a base os traz.
    """
    hora = df["hora"].to_numpy("float64")
    dia = df["dia_semana"].to_numpy("float64")
    colunas: list[Any] = [
        np.sin(2 * np.pi * hora / 24), np.cos(2 * np.pi * hora / 24),
        np.sin(2 * np.pi * dia / 7), np.cos(2 * np.pi * dia / 7),
        df["fim_de_semana"], df["pico"], df["sentido"],
        df["percurso_tipico_s"], df["extensao_m"], df["ordem_origem"],
        df["lat"], df["lon"],
    ]
    extras = [c for c in COLUNAS_VIZINHANCA if c in df.columns]
    if com_desvios:
        extras = ["atraso_trecho_anterior_s", "desvio_recente_trecho_s",
                  "n_recente_trecho", "desvio_ultimo_veiculo_s",
                  "desde_ultimo_veiculo_s"] + extras
    for c in extras:
        v = pd.to_numeric(df[c], errors="coerce")
        colunas += [v.fillna(0), v.isna()]
    return np.column_stack(
        [np.asarray(pd.to_numeric(pd.Series(np.asarray(c)), errors="coerce"), "float64")
         for c in colunas]
    ).astype("float32")


def _padronizar(X: np.ndarray, pos_treino: np.ndarray) -> np.ndarray:
    eixos = tuple(range(X.ndim - 1))
    ref = X[pos_treino]
    mu = np.nanmean(ref, axis=eixos)
    sd = np.nanstd(ref, axis=eixos) + 1e-6
    return np.nan_to_num((X - mu) / sd).astype("float32")


def _seq_viagem(df: pd.DataFrame, passos: int) -> np.ndarray:
    """
    (n, passos, 5): os `passos` trechos anteriores da MESMA viagem, do mais
    antigo ao mais recente — desvio, desvio relativo, tempo tipico, distancia em
    trechos e presenca. Todos terminaram ate a parada de origem.
    """
    d = df[["viagem_id", "ordem_origem", "alvo_s", "percurso_tipico_s"]].assign(
        _rel=_relativo(df)
    ).sort_values(["viagem_id", "ordem_origem"])
    g = d.groupby("viagem_id", sort=False)
    blocos = []
    for defasagem in range(passos, 0, -1):
        desvio = g["alvo_s"].shift(defasagem)
        presente = desvio.notna().to_numpy("float64")
        blocos.append(np.column_stack([
            desvio.fillna(0).to_numpy() / 60,
            g["_rel"].shift(defasagem).fillna(0).to_numpy(),
            g["percurso_tipico_s"].shift(defasagem).fillna(0).to_numpy() / 100,
            (d["ordem_origem"] - g["ordem_origem"].shift(defasagem)).fillna(0).to_numpy()
            / passos,
            presente,
        ]))
    X = np.stack(blocos, axis=1).astype("float32")
    saida = np.empty_like(X)
    saida[df.index.get_indexer(d.index)] = X
    return saida


def _seq_trecho(df: pd.DataFrame, passos: int) -> np.ndarray:
    """
    (n, passos, 4): os ultimos veiculos que COMPLETARAM o mesmo trecho antes de
    `t_origem` (no maximo 60 min antes), do mais antigo ao mais recente —
    desvio, desvio relativo, idade da observacao e presenca.
    """
    ev = (
        df[["cl", "ordem_origem", "t_chegada", "alvo_s"]]
        .assign(_rel=_relativo(df))
        .dropna(subset=["alvo_s"])
        .astype({"t_chegada": NS})
        .sort_values(["cl", "ordem_origem", "t_chegada"])
        .reset_index(drop=True)
    )
    ev["pos"] = np.arange(len(ev), dtype="float64")
    grupo = ev.groupby(["cl", "ordem_origem"], sort=False).ngroup().to_numpy()
    t_ev = _segundos(ev["t_chegada"])
    desvio = ev["alvo_s"].to_numpy("float64")
    rel = ev["_rel"].fillna(0).to_numpy("float64")

    consulta = df[["cl", "ordem_origem", "t_origem"]].astype({"t_origem": NS}).reset_index()
    m = pd.merge_asof(
        consulta.sort_values("t_origem"),
        ev[["cl", "ordem_origem", "t_chegada", "pos"]].sort_values("t_chegada"),
        left_on="t_origem", right_on="t_chegada", by=["cl", "ordem_origem"],
        allow_exact_matches=False, tolerance=JANELA_TRECHO,
    )
    tem = m["pos"].notna().to_numpy()
    p = m["pos"].fillna(0).to_numpy("int64")
    P = p[:, None] - np.arange(passos - 1, -1, -1)[None, :]  # antigo -> recente
    Pc = np.clip(P, 0, None)
    idade_min = (_segundos(m["t_origem"])[:, None] - t_ev[Pc]) / 60
    valido = (
        tem[:, None] & (P >= 0) & (grupo[Pc] == grupo[p][:, None])
        & (idade_min <= JANELA_TRECHO / pd.Timedelta("1min"))
    )
    X = np.stack([
        np.where(valido, desvio[Pc] / 60, 0),
        np.where(valido, rel[Pc], 0),
        np.where(valido, idade_min / 60, 0),
        valido.astype("float64"),
    ], axis=2).astype("float32")
    saida = np.empty_like(X)
    saida[df.index.get_indexer(m["index"])] = X
    return saida


# ------------------------------------------------------------ treino comum
def _perda(saida, y, c):
    """Perda quantilica dos tres quantis + entropia cruzada da probabilidade."""
    import torch

    q = torch.tensor(QUANTIS, dtype=saida.dtype)
    erro = y[:, None] - saida[:, : len(QUANTIS)]
    quantilica = torch.mean(torch.maximum(q * erro, (q - 1) * erro))
    logit = saida[:, len(QUANTIS)]
    return quantilica + torch.nn.functional.binary_cross_entropy_with_logits(logit, c)


def _desempacotar(saida: np.ndarray, escala: float) -> tuple[np.ndarray, np.ndarray]:
    """Quantis em segundos (ordenados, sem cruzamento) e probabilidade."""
    quantis = np.sort(saida[:, : len(QUANTIS)], axis=1) * escala
    prob = 1 / (1 + np.exp(-saida[:, len(QUANTIS)]))
    return quantis, prob


def _resumir(base: dict, rodadas: dict[str, list[dict]]) -> dict:
    """
    Metricas no teste, por configuracao: media e desvio entre sementes e o
    conjunto (media das previsoes das sementes), com as mesmas linhas de base.
    """
    df = base["df"]
    yte = df.loc[base["teste"], "alvo_s"].to_numpy("float32")
    ctr = df.loc[base["treino"], "alvo_atrasado"].to_numpy("int8")
    cva = df.loc[base["validacao"], "alvo_atrasado"].to_numpy("int8")
    cte = df.loc[base["teste"], "alvo_atrasado"].to_numpy("int8")

    def avaliar(q: np.ndarray, prob_va: np.ndarray, prob_te: np.ndarray) -> dict:
        return {
            "regressao": _metricas(yte, q[:, 1]),
            "cobertura_p10": float(np.mean(yte <= q[:, 0])),
            "cobertura_p90": float(np.mean(yte <= q[:, 2])),
            "classificacao": _classificacao(ctr, cva, cte, prob_va, prob_te),
        }

    saida: dict[str, Any] = {}
    for nome, lista in rodadas.items():
        avaliacoes = [avaliar(r["q"], r["prob_va"], r["prob_te"]) for r in lista]
        mae = np.array([a["regressao"]["mae_s"] for a in avaliacoes])
        resumo = {
            "sementes": len(lista),
            "mae_s_media": float(mae.mean()),
            "mae_s_dp": float(mae.std()),
            "mae_s_por_semente": [round(float(v), 2) for v in mae],
            "epocas": [r["epocas"] for r in lista],
        }
        conjunto = avaliar(
            np.mean([r["q"] for r in lista], axis=0),
            np.mean([r["prob_va"] for r in lista], axis=0),
            np.mean([r["prob_te"] for r in lista], axis=0),
        )
        saida[nome] = {**resumo, "conjunto": conjunto}
    return saida


# ---------------------------------------------------------------------- LSTM
def treinar_lstm_atraso(
    df: pd.DataFrame | None = None,
    limiar_percentual: float = 0.20,
    base: dict | None = None,
    passos: int = 8,
    oculto: int = 64,
    epocas: int = 25,
    paciencia: int = 4,
    lote: int = 512,
    sementes: Sequence[int] = (1, 2, 3),
    variantes: Sequence[str] = ("completa", "so_viagem", "so_trecho"),
) -> dict:
    """LSTM com a memoria da viagem e a do trecho. As variantes `so_*` zeram a
    outra memoria e rodam com a primeira semente apenas."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "instale: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    base = base or preparar_atraso(df, limiar_percentual)
    d = base["df"]
    pos = {k: d.index.get_indexer(base[k]) for k in ("treino", "validacao", "teste")}
    Xv = _seq_viagem(d, passos)
    Xt = _seq_trecho(d, passos)
    Xc = _padronizar(_contexto(d, com_desvios=False), pos["treino"])
    y = d["alvo_s"].to_numpy("float32")
    c = d["alvo_atrasado"].to_numpy("float32")
    escala = float(np.std(y[pos["treino"]])) or 1.0
    log.info(
        "LSTM atraso: %d trechos | memoria da viagem presente em %.0f%% | do trecho em %.0f%%",
        len(d), 100 * Xv[:, -1, -1].mean(), 100 * Xt[:, -1, -1].mean(),
    )

    tXv, tXt, tXc = (torch.from_numpy(a) for a in (Xv, Xt, Xc))
    ty, tc = torch.from_numpy(y / escala), torch.from_numpy(c)

    class Rede(nn.Module):
        def __init__(self):
            super().__init__()
            self.viagem = nn.LSTM(Xv.shape[2], oculto, batch_first=True)
            self.trecho = nn.LSTM(Xt.shape[2], oculto, batch_first=True)
            self.contexto = nn.Sequential(nn.Linear(Xc.shape[1], oculto), nn.ReLU())
            self.cabeca = nn.Sequential(
                nn.Linear(3 * oculto, oculto), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(oculto, len(QUANTIS) + 1),
            )

        def forward(self, xv, xt, xc):
            _, (hv, _) = self.viagem(xv)
            _, (ht, _) = self.trecho(xt)
            return self.cabeca(torch.cat([hv[-1], ht[-1], self.contexto(xc)], 1))

    def entradas(idx: np.ndarray, variante: str):
        idx = torch.as_tensor(idx)
        xv, xt = tXv[idx], tXt[idx]
        if variante == "so_trecho":
            xv = torch.zeros_like(xv)
        elif variante == "so_viagem":
            xt = torch.zeros_like(xt)
        return xv, xt, tXc[idx]

    def prever(rede, idx: np.ndarray, variante: str) -> np.ndarray:
        rede.eval()
        partes = []
        with torch.no_grad():
            for i in range(0, len(idx), 8192):
                partes.append(rede(*entradas(idx[i : i + 8192], variante)).numpy())
        return np.concatenate(partes)

    rodadas: dict[str, list[dict]] = {}
    for variante in variantes:
        for semente in (sementes if variante == "completa" else sementes[:1]):
            torch.manual_seed(semente)
            rng = np.random.default_rng(semente)
            rede = Rede()
            otimizador = torch.optim.Adam(rede.parameters(), lr=1e-3)
            melhor, espera, melhor_estado, feitas = float("inf"), 0, None, 0
            for epoca in range(epocas):
                rede.train()
                ordem = rng.permutation(pos["treino"])
                for i in range(0, len(ordem), lote):
                    b = ordem[i : i + lote]
                    otimizador.zero_grad()
                    bt = torch.as_tensor(b)
                    _perda(rede(*entradas(b, variante)), ty[bt], tc[bt]).backward()
                    otimizador.step()
                feitas = epoca + 1
                q_va, _ = _desempacotar(prever(rede, pos["validacao"], variante), escala)
                val = float(np.mean(np.abs(q_va[:, 1] - y[pos["validacao"]])))
                log.info("LSTM %s semente %d epoca %d: MAE validacao %.2f s",
                         variante, semente, epoca, val)
                if val < melhor - 1e-3:
                    melhor, espera = val, 0
                    melhor_estado = {k: v.clone() for k, v in rede.state_dict().items()}
                else:
                    espera += 1
                    if espera >= paciencia:
                        break
            if melhor_estado is None:
                raise RuntimeError(f"LSTM {variante} semente {semente} divergiu")
            rede.load_state_dict(melhor_estado)
            _, prob_va = _desempacotar(prever(rede, pos["validacao"], variante), escala)
            q_te, prob_te = _desempacotar(prever(rede, pos["teste"], variante), escala)
            rodadas.setdefault(variante, []).append(
                {"q": q_te, "prob_va": prob_va, "prob_te": prob_te, "epocas": feitas}
            )

    return {
        "resultado": {
            "baselines": baselines_atraso(base),
            "passos": passos,
            **_resumir(base, rodadas),
        },
        "rodadas": rodadas,
    }


# ----------------------------------------------------------------------- GNN
def _grafo_trechos(d: pd.DataFrame, raio_m: float):
    """
    Nos = trechos fisicos (par de paradas). Tres relacoes, normalizadas pela
    media dos vizinhos: o trecho de cima (de onde vem o onibus), o de baixo
    (para onde ele vai) e os vizinhos a menos de `raio_m` que nao sejam
    sequenciais (corredores paralelos e cruzamentos).
    """
    import torch

    from . import geo

    chave = pd.MultiIndex.from_arrays([d["cp_origem"], d["cp_destino"]])
    seg, unicos = pd.factorize(chave)
    n = len(unicos)

    s = pd.DataFrame({
        "viagem_id": d["viagem_id"].to_numpy(), "ordem": d["ordem_origem"].to_numpy(),
        "seg": seg, "cp_o": d["cp_origem"].to_numpy(), "cp_d": d["cp_destino"].to_numpy(),
    }).sort_values(["viagem_id", "ordem"])
    g = s.groupby("viagem_id", sort=False)
    seguinte = g["seg"].shift(-1)
    continua = s["cp_d"].to_numpy() == g["cp_o"].shift(-1).to_numpy()
    pares = (
        pd.DataFrame({"a": s["seg"][continua], "b": seguinte[continua]})
        .dropna().astype("int64").drop_duplicates()
    )
    pares = pares[pares["a"] != pares["b"]]

    pontos = pd.DataFrame({"seg": seg, "lat": d["lat"].to_numpy(), "lon": d["lon"].to_numpy()})
    medio = pontos.groupby("seg")[["lat", "lon"]].mean().reindex(range(n))
    x, y = geo.obter_projecao().para_xy(medio["lat"].to_numpy(), medio["lon"].to_numpy())
    sequenciais = set(map(tuple, pares[["a", "b"]].to_numpy())) | set(
        map(tuple, pares[["b", "a"]].to_numpy())
    )
    viz: list[tuple[int, int]] = []
    try:
        from scipy.spatial import cKDTree

        for a, b in cKDTree(np.column_stack([x, y])).query_pairs(r=raio_m):
            if (a, b) not in sequenciais:
                viz += [(a, b), (b, a)]
    except ImportError:
        log.warning("scipy ausente: GNN sem a relacao de vizinhanca espacial")

    def adjacencia(origem: np.ndarray, destino: np.ndarray):
        # linha i agrega as colunas j (media): A[i, j] = 1 / grau(i)
        if len(origem) == 0:
            return None
        grau = np.bincount(origem, minlength=n).astype("float32")
        valores = 1.0 / grau[origem]
        return torch.sparse_coo_tensor(
            np.vstack([origem, destino]), valores, (n, n)
        ).coalesce()

    a, b = pares["a"].to_numpy(), pares["b"].to_numpy()
    relacoes = [
        adjacencia(b, a),  # de cima: b recebe de a, que vem antes
        adjacencia(a, b),  # de baixo: a recebe de b, para onde o onibus vai
    ]
    if viz:
        va = np.array(viz, dtype="int64")
        relacoes.append(adjacencia(va[:, 0], va[:, 1]))
    relacoes = [r for r in relacoes if r is not None]
    info = {"nos": n, "arestas_sequenciais": int(len(pares)), "arestas_vizinhanca": len(viz) // 2}
    return seg, relacoes, info


def _estado_rede(d: pd.DataFrame, seg: np.ndarray, n: int, bin_min: int):
    """
    (janelas, nos, 6): estado de cada trecho fisico no INICIO de cada janela de
    `bin_min` minutos, so com conclusoes anteriores a ele — desvio relativo
    mediano em 15 e 60 min, desvio mediano em 15 min, veiculos em 15 min, idade
    da ultima observacao e presenca em 60 min. As janelas moveis terminam na
    conclusao mais recente; a idade diz o quanto ela ja ficou para tras.
    """
    ev = (
        pd.DataFrame({
            "seg": seg,
            "t": d["t_chegada"].astype(NS).reset_index(drop=True),
            "rel": _relativo(d).reset_index(drop=True),
            "desvio": d["alvo_s"].reset_index(drop=True),
        })
        .dropna(subset=["desvio"])
        .sort_values(["seg", "t"])
        .reset_index(drop=True)
    )
    por_seg = ev.set_index("t").groupby("seg")
    ev["rel15"] = por_seg["rel"].rolling("15min").median().to_numpy()
    ev["rel60"] = por_seg["rel"].rolling("60min").median().to_numpy()
    ev["desvio15"] = por_seg["desvio"].rolling("15min").median().to_numpy()
    ev["n15"] = por_seg["desvio"].rolling("15min").count().to_numpy()

    t0 = d["t_origem"].astype(NS).min().floor(f"{bin_min}min")
    passo = pd.Timedelta(f"{bin_min}min")
    janela_de = ((d["t_origem"].astype(NS) - t0) // passo).to_numpy("int64")
    n_janelas = int(janela_de.max()) + 1
    inicio = t0 + passo * np.arange(n_janelas)

    grade = pd.DataFrame({
        "k": np.repeat(np.arange(n_janelas), n),
        "seg": np.tile(np.arange(n), n_janelas),
        "inicio": inicio.repeat(n),
    }).astype({"inicio": NS})
    m = pd.merge_asof(
        grade.sort_values("inicio"), ev.sort_values("t"),
        left_on="inicio", right_on="t", by="seg",
        allow_exact_matches=False, tolerance=pd.Timedelta("60min"),
    ).sort_values(["k", "seg"])
    idade = ((m["inicio"] - m["t"]) / pd.Timedelta("1min")).to_numpy("float64")
    em15 = idade <= 15
    em60 = idade <= 60
    X = np.column_stack([
        np.where(em15, m["rel15"], 0),
        np.where(em60, m["rel60"], 0),
        np.where(em15, m["desvio15"] / 60, 0),
        np.where(em15, np.log1p(m["n15"]), 0),
        np.where(em60, np.log1p(idade) / np.log1p(60), 1),
        em60.astype("float64"),
    ])
    X = np.nan_to_num(X).reshape(n_janelas, n, -1).astype("float32")
    return X, janela_de


def treinar_gnn_atraso(
    df: pd.DataFrame | None = None,
    limiar_percentual: float = 0.20,
    base: dict | None = None,
    bin_min: int = 5,
    raio_m: float = 300.0,
    oculto: int = 32,
    epocas: int = 40,
    paciencia: int = 5,
    janelas_por_lote: int = 32,
    sementes: Sequence[int] = (1, 2, 3),
    variantes: Sequence[str] = ("gcn", "sem_grafo"),
) -> dict:
    """GNN espaco-temporal sobre os trechos fisicos, e a mesma rede sem grafo."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("instale o torch para treinar a GNN") from exc

    base = base or preparar_atraso(df, limiar_percentual)
    d = base["df"]
    pos = {k: d.index.get_indexer(base[k]) for k in ("treino", "validacao", "teste")}
    seg, relacoes, info = _grafo_trechos(d, raio_m)
    X, janela_de = _estado_rede(d, seg, info["nos"], bin_min)

    # padroniza o estado com as janelas do treino apenas
    ultima_treino = int(janela_de[pos["treino"]].max())
    ref = X[: ultima_treino + 1].reshape(-1, X.shape[2])
    X = ((X - ref.mean(0)) / (ref.std(0) + 1e-6)).astype("float32")
    linha = _padronizar(_contexto(d, com_desvios=True), pos["treino"])
    y = d["alvo_s"].to_numpy("float32")
    c = d["alvo_atrasado"].to_numpy("float32")
    escala = float(np.std(y[pos["treino"]])) or 1.0
    log.info("GNN atraso: %s | %d janelas de %d min", info, X.shape[0], bin_min)

    tX = torch.from_numpy(X)
    tlinha = torch.from_numpy(linha)
    tseg = torch.from_numpy(seg.astype("int64"))
    ty, tc = torch.from_numpy(y / escala), torch.from_numpy(c)

    class Camada(nn.Module):
        def __init__(self, ent: int, sai: int, n_rel: int):
            super().__init__()
            self.proprio = nn.Linear(ent, sai)
            self.vizinhos = nn.ModuleList(
                nn.Linear(ent, sai, bias=False) for _ in range(n_rel)
            )

        def forward(self, H, adjs):
            k, n, f = H.shape
            saida = self.proprio(H)
            plano = H.permute(1, 0, 2).reshape(n, k * f)
            for A, linear in zip(adjs, self.vizinhos):
                agregado = torch.sparse.mm(A, plano).reshape(n, k, f).permute(1, 0, 2)
                saida = saida + linear(agregado)
            return torch.relu(saida)

    class Rede(nn.Module):
        def __init__(self, n_rel: int):
            super().__init__()
            self.g1 = Camada(X.shape[2], oculto, n_rel)
            self.g2 = Camada(oculto, oculto, n_rel)
            self.cabeca = nn.Sequential(
                nn.Linear(oculto + linha.shape[1], 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, len(QUANTIS) + 1),
            )

        def forward(self, janelas, adjs, posicao, trecho, xl):
            H = self.g2(self.g1(tX[janelas], adjs), adjs)
            return self.cabeca(torch.cat([H[posicao, trecho], xl], 1))

    def agrupar(idx: np.ndarray):
        """Linhas agrupadas por janela: (janelas unicas, inicio e fim de cada uma)."""
        ordem = idx[np.argsort(janela_de[idx], kind="stable")]
        k = janela_de[ordem]
        unicas = np.unique(k)
        return ordem, k, unicas, np.searchsorted(k, unicas, "left"), np.searchsorted(k, unicas, "right")

    def lote(rede, adjs, grupo, escolhidas: np.ndarray):
        ordem, k, unicas, ini, fim = grupo
        linhas = np.concatenate([ordem[ini[i]:fim[i]] for i in escolhidas])
        janelas = unicas[escolhidas]
        posicao = np.searchsorted(janelas, janela_de[linhas])
        saida = rede(
            torch.from_numpy(janelas), adjs, torch.from_numpy(posicao),
            tseg[torch.as_tensor(linhas)], tlinha[torch.as_tensor(linhas)],
        )
        return saida, linhas

    def prever(rede, adjs, idx: np.ndarray) -> np.ndarray:
        grupo = agrupar(idx)
        rede.eval()
        saida = np.empty((len(idx), len(QUANTIS) + 1), dtype="float32")
        onde = {int(v): i for i, v in enumerate(idx)}
        with torch.no_grad():
            for i in range(0, len(grupo[2]), 64):
                out, linhas = lote(rede, adjs, grupo, np.arange(i, min(i + 64, len(grupo[2]))))
                saida[[onde[int(r)] for r in linhas]] = out.numpy()
        return saida

    grupo_treino = agrupar(pos["treino"])
    grafos = {"gcn": relacoes, "sem_grafo": []}
    rodadas: dict[str, list[dict]] = {}
    for nome in variantes:
        adjs = grafos[nome]
        for semente in sementes:
            torch.manual_seed(semente)
            rng = np.random.default_rng(semente)
            rede = Rede(len(adjs))
            otimizador = torch.optim.Adam(rede.parameters(), lr=2e-3, weight_decay=1e-5)
            melhor, espera, melhor_estado, feitas = float("inf"), 0, None, 0
            n_janelas = len(grupo_treino[2])
            for epoca in range(epocas):
                rede.train()
                ordem = rng.permutation(n_janelas)
                for i in range(0, n_janelas, janelas_por_lote):
                    otimizador.zero_grad()
                    out, linhas = lote(rede, adjs, grupo_treino, np.sort(ordem[i : i + janelas_por_lote]))
                    lt = torch.as_tensor(linhas)
                    _perda(out, ty[lt], tc[lt]).backward()
                    otimizador.step()
                feitas = epoca + 1
                q_va, _ = _desempacotar(prever(rede, adjs, pos["validacao"]), escala)
                val = float(np.mean(np.abs(q_va[:, 1] - y[pos["validacao"]])))
                log.info("GNN %s semente %d epoca %d: MAE validacao %.2f s",
                         nome, semente, epoca, val)
                if val < melhor - 1e-3:
                    melhor, espera = val, 0
                    melhor_estado = {k: v.clone() for k, v in rede.state_dict().items()}
                else:
                    espera += 1
                    if espera >= paciencia:
                        break
            if melhor_estado is None:
                raise RuntimeError(f"GNN {nome} semente {semente} divergiu")
            rede.load_state_dict(melhor_estado)
            _, prob_va = _desempacotar(prever(rede, adjs, pos["validacao"]), escala)
            q_te, prob_te = _desempacotar(prever(rede, adjs, pos["teste"]), escala)
            rodadas.setdefault(nome, []).append(
                {"q": q_te, "prob_va": prob_va, "prob_te": prob_te, "epocas": feitas}
            )

    return {
        "resultado": {
            "baselines": baselines_atraso(base),
            "grafo": info,
            "janela_min": bin_min,
            **_resumir(base, rodadas),
        },
        "rodadas": rodadas,
    }
