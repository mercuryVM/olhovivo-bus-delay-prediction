"""
A frota inteira como sensor de transito nos trechos das linhas estudadas.

Cada par de paradas consecutivas das linhas estudadas e percorrido, em mediana,
por 11 linhas-sentido (ate 34), mas so as estudadas entram nos atributos de
vizinhanca. Aqui toda leitura de GPS de qualquer onibus que caia sobre o
trecho, no sentido dele, vira medida de velocidade.

Por dia:

1. **Geometria.** O traçado de cada linha estudada (derivado/tracados/{cl}.json)
   e a abscissa das paradas (catalogo linha_parada) dao a polilinha de cada par
   (cp_origem, cp_destino), densificada a cada 10 m, com o rumo local.
2. **Leituras.** Posicoes de todos os veiculos, sem duplicatas por (prefixo,
   ta), em ordem por veiculo; o rumo de cada leitura e o do ultimo deslocamento
   de ao menos 15 m do proprio veiculo.
3. **Casamento.** Leitura a menos de 25 m de um ponto do par e com rumo
   alinhado (cosseno > 0,5) recebe o par e a abscissa local. Vias sobrepostas
   (dois pares no mesmo leito) recebem os dois.
4. **Medida.** Duas leituras consecutivas do mesmo veiculo no mesmo par, com 10
   a 120 s entre elas, dao a velocidade no par, datada pela segunda leitura.

Os atributos consultam as medidas estritamente antes de `t_origem`, como em
`vizinhanca.py`: mediana da velocidade em 10 e 30 min, contagens, idade da
medida mais recente e o tempo de percurso implicito (extensao / velocidade).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import geo
from .features import NS

log = logging.getLogger("olhovivo.frota")
TZ = ZoneInfo("America/Sao_Paulo")

PASSO_M = 10.0
RAIO_M = 25.0
COS_MIN = 0.5
DT_MIN_S, DT_MAX_S = 10.0, 120.0
MOVIMENTO_MIN_M = 15.0
VIZINHOS = 6

COLUNAS_FROTA = [
    "frota_vel10_kmh", "frota_n10", "frota_vel30_kmh", "frota_n30",
    "frota_idade_s", "frota_tempo_s",
]


def _projecao(cfg):
    return geo.obter_projecao(
        int(cfg.get("geo.epsg_metrico", 31983)),
        float(cfg.get("geo.ancora_lat", -23.5505)),
        float(cfg.get("geo.ancora_lon", -46.6333)),
    )


def geometria(arm, cfg) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Pares de paradas consecutivas das linhas estudadas e seus pontos densificados."""
    proj = _projecao(cfg)
    vinc = arm.ler_catalogo("linha_parada").dropna(subset=["s_m"])
    pares, pts = [], {k: [] for k in ("x", "y", "par", "s", "ux", "uy")}
    vistos: set[tuple[int, int]] = set()
    for cl, paradas in vinc.groupby("cl"):
        arq = arm.raiz / "derivado" / "tracados" / f"{int(cl)}.json"
        if not arq.exists():
            continue
        t = json.loads(arq.read_text(encoding="utf-8"))
        x, y = proj.para_xy(np.asarray(t["lat"]), np.asarray(t["lon"]))
        x, y = np.asarray(x), np.asarray(y)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        p = paradas.sort_values("ordem")
        for a, b in zip(p.itertuples(), list(p.itertuples())[1:]):
            chave = (int(a.cp), int(b.cp))
            s0, s1 = float(a.s_m), float(b.s_m)
            if chave in vistos or not (0 < s1 - s0 < 8000):
                continue
            vistos.add(chave)
            sk = np.append(np.arange(s0, s1, PASSO_M), s1)
            px, py = np.interp(sk, s, x), np.interp(sk, s, y)
            dx = np.interp(sk + 5, s, x) - np.interp(sk - 5, s, x)
            dy = np.interp(sk + 5, s, y) - np.interp(sk - 5, s, y)
            norma = np.hypot(dx, dy)
            norma[norma == 0] = 1.0
            pid = len(pares)
            pares.append({"par": pid, "cp_origem": chave[0], "cp_destino": chave[1],
                          "cl": int(cl), "extensao_m": s1 - s0})
            for k, v in (("x", px), ("y", py), ("s", sk - s0), ("ux", dx / norma), ("uy", dy / norma)):
                pts[k].append(v)
            pts["par"].append(np.full(len(sk), pid))
    pontos = {k: np.concatenate(v) for k, v in pts.items()}
    log.info("geometria: %d pares, %d pontos densificados", len(pares), len(pontos["x"]))
    return pd.DataFrame(pares), pontos


def medidas_do_dia(arm, cfg, dia: date, pares: pd.DataFrame, pontos: dict) -> pd.DataFrame:
    """Velocidades de toda a frota sobre os pares, num dia local."""
    proj = _projecao(cfg)
    inicio = datetime.combine(dia, time(0), TZ).astimezone(ZoneInfo("UTC"))
    pos = arm.ler_bruto("posicoes", inicio=inicio, fim=inicio + timedelta(days=1),
                        colunas=["cl", "prefixo", "ta", "lat", "lon"])
    if pos.empty:
        return pd.DataFrame()
    pos = pos.dropna(subset=["ta", "lat", "lon"])
    pos["ta"] = pd.to_datetime(pos["ta"], utc=True).astype(NS)
    pos["prefixo"] = pos["prefixo"].astype(str)
    pos = pos.drop_duplicates(subset=["prefixo", "ta"]).sort_values(["prefixo", "ta"])
    x, y = proj.para_xy(pos["lat"].to_numpy(), pos["lon"].to_numpy())
    x, y = np.asarray(x), np.asarray(y)
    pref = pos["prefixo"].to_numpy()
    ta = pos["ta"].astype("int64").to_numpy() / 1e9
    mesmo = np.r_[False, pref[1:] == pref[:-1]]
    dt = np.r_[np.nan, np.diff(ta)]

    # rumo: ultimo deslocamento de ao menos 15 m do proprio veiculo
    dx, dy = np.r_[0.0, np.diff(x)], np.r_[0.0, np.diff(y)]
    d = np.hypot(dx, dy)
    ok = mesmo & (d >= MOVIMENTO_MIN_M) & (dt <= 300)
    rumo = pd.DataFrame({"g": np.cumsum(~mesmo), "ux": np.where(ok, dx / np.where(d > 0, d, 1), np.nan),
                         "uy": np.where(ok, dy / np.where(d > 0, d, 1), np.nan)})
    rumo[["ux", "uy"]] = rumo.groupby("g")[["ux", "uy"]].ffill(limit=10)
    ux, uy = rumo["ux"].to_numpy(), rumo["uy"].to_numpy()

    # so leituras perto de algum par interessam
    arvore = cKDTree(np.c_[pontos["x"], pontos["y"]])
    dist, j = arvore.query(np.c_[x, y], k=VIZINHOS, distance_upper_bound=RAIO_M, workers=-1)
    n_pts = len(pontos["x"])
    linhas, pares_l, s_l = [], [], []
    for k in range(VIZINHOS):
        jk = j[:, k]
        valido = jk < n_pts
        jv = np.where(valido, jk, 0)
        cos = ux * pontos["ux"][jv] + uy * pontos["uy"][jv]
        m = valido & (cos > COS_MIN)
        linhas.append(np.flatnonzero(m))
        pares_l.append(pontos["par"][jv[m]])
        s_l.append(pontos["s"][jv[m]])
    L = pd.DataFrame({"i": np.concatenate(linhas), "par": np.concatenate(pares_l),
                      "s": np.concatenate(s_l)})
    # varios pontos do mesmo par entre os vizinhos: fica a abscissa do mais proximo (o primeiro)
    L = L.drop_duplicates(subset=["i", "par"], keep="first")

    # pares de leituras consecutivas do mesmo veiculo no mesmo par
    anterior = L.rename(columns={"i": "i0", "s": "s0"})
    anterior["i"] = anterior["i0"] + 1
    M = L.merge(anterior, on=["i", "par"])
    M = M[mesmo[M["i"].to_numpy()]]
    dtm = dt[M["i"].to_numpy()]
    ds = M["s"].to_numpy() - M["s0"].to_numpy()
    ok = (dtm >= DT_MIN_S) & (dtm <= DT_MAX_S) & (ds >= -5)
    M = M[ok]
    vel = np.maximum(ds[ok], 0) / dtm[ok]
    saida = pd.DataFrame({
        "par": M["par"].to_numpy(),
        "t": pd.to_datetime(ta[M["i"].to_numpy()], unit="s", utc=True),
        "vel_ms": vel,
        "cl_medidor": pos["cl"].to_numpy()[M["i"].to_numpy()],
        "prefixo": pref[M["i"].to_numpy()],
    }).merge(pares[["par", "cp_origem", "cp_destino"]], on="par")
    log.info("frota %s: %d leituras, %d casadas, %d medidas em %d pares", dia, len(pos),
             L["i"].nunique(), len(saida), saida["par"].nunique())
    return saida.drop(columns="par")


def gerar(arm, cfg, dias: list[date], forcar: bool = False) -> Path:
    """Grava as medidas por dia em derivado/frota/dt=AAAA-MM-DD.parquet."""
    destino = arm.raiz / "derivado" / "frota"
    destino.mkdir(parents=True, exist_ok=True)
    pares, pontos = geometria(arm, cfg)
    for dia in dias:
        arq = destino / f"dt={dia:%Y-%m-%d}.parquet"
        if arq.exists() and not forcar:
            continue
        m = medidas_do_dia(arm, cfg, dia, pares, pontos)
        if len(m):
            m.to_parquet(arq, index=False)
    return destino


def ler(raiz: Path) -> pd.DataFrame:
    pasta = Path(raiz) / "derivado" / "frota"
    arquivos = sorted(pasta.glob("dt=*.parquet"))
    if not arquivos:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(a) for a in arquivos], ignore_index=True)


def _janela(ev: pd.DataFrame, consulta: pd.DataFrame, janela: str, nome: str) -> pd.DataFrame:
    chave = ["cp_origem", "cp_destino"]
    ev = ev.sort_values(chave + ["t"]).reset_index(drop=True)
    rol = ev.set_index("t").groupby(chave)["vel_ms"].rolling(janela)
    ev[f"{nome}_med"] = rol.median().to_numpy()
    ev[f"{nome}_n"] = rol.count().to_numpy()
    m = pd.merge_asof(
        consulta.sort_values("t"), ev.rename(columns={"t": "t_ev"})[chave + ["t_ev", f"{nome}_med", f"{nome}_n"]]
        .sort_values("t_ev"), left_on="t", right_on="t_ev", by=chave,
        allow_exact_matches=False, tolerance=pd.Timedelta(janela),
    )
    return m.set_index("i").sort_index()


def atributos(df: pd.DataFrame, medidas: pd.DataFrame) -> pd.DataFrame:
    """Colunas de `COLUNAS_FROTA` no indice de `df` (percursos com cp e t_origem)."""
    saida = pd.DataFrame(np.nan, index=df.index, columns=COLUNAS_FROTA)
    if medidas is None or medidas.empty:
        return saida
    ev = medidas[["cp_origem", "cp_destino", "t", "vel_ms"]].copy()
    ev["t"] = pd.to_datetime(ev["t"], utc=True).astype(NS)
    consulta = pd.DataFrame({
        "i": np.arange(len(df)),
        "cp_origem": df["cp_origem"].astype("int64").to_numpy(),
        "cp_destino": df["cp_destino"].astype("int64").to_numpy(),
        "t": pd.to_datetime(df["t_origem"], utc=True).astype(NS).to_numpy(),
    })
    ev = ev.astype({"cp_origem": "int64", "cp_destino": "int64"})
    j10 = _janela(ev, consulta, "10min", "v10")
    j30 = _janela(ev, consulta, "30min", "v30")
    saida["frota_vel10_kmh"] = j10["v10_med"].to_numpy() * 3.6
    saida["frota_n10"] = j10["v10_n"].to_numpy()
    saida["frota_vel30_kmh"] = j30["v30_med"].to_numpy() * 3.6
    saida["frota_n30"] = j30["v30_n"].to_numpy()
    saida["frota_idade_s"] = (j30["t"] - j30["t_ev"]).dt.total_seconds().to_numpy()
    vel = saida["frota_vel10_kmh"].fillna(saida["frota_vel30_kmh"]) / 3.6
    saida["frota_tempo_s"] = df["extensao_m"].to_numpy("float64") / vel.where(vel > 0.5)
    return saida.astype("float64")
