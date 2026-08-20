"""
Autoteste com dados sinteticos.

Nao chama a API. Constroi uma linha de onibus artificial com horarios de
chegada CONHECIDOS, roda o pipeline inteiro em cima dela e confere se o que
sai bate com a verdade. E o jeito de saber que a deteccao de chegada e o
casamento previsao x realizado estao corretos antes de gastar uma semana
coletando.

    python scripts/autoteste.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from olhovivo import casamento, chegadas, coleta, config, geo, regioes  # noqa: E402
from olhovivo.storage import (  # noqa: E402
    ESQUEMA_LINHA_PARADA,
    ESQUEMA_LINHAS,
    ESQUEMA_PARADAS,
    ESQUEMA_PREVISOES,
    Armazenamento,
    EscritorParquet,
    ESQUEMA_POSICOES,
)

FALHAS: list[str] = []


def checar(condicao: bool, descricao: str, detalhe: str = "") -> None:
    marca = "OK  " if condicao else "FALHA"
    print(f"  [{marca}] {descricao}" + (f"  ({detalhe})" if detalhe else ""))
    if not condicao:
        FALHAS.append(descricao)


# ---------------------------------------------------------------- 1. geo
def teste_geo() -> None:
    print("\n1) geo")
    # Praca da Se -> Estadio do Morumbi ~ 9,6 km em linha reta
    d = float(geo.haversine_m(-23.5505, -46.6333, -23.6000, -46.7200))
    checar(9000 < d < 10500, "haversine em escala urbana", f"{d:.0f} m")

    proj = geo.ProjecaoLocal()
    x, y = proj.para_xy(-23.6, -46.7)
    lat, lon = proj.para_latlon(x, y)
    checar(
        abs(lat + 23.6) < 1e-9 and abs(lon + 46.7) < 1e-9,
        "projecao local ida e volta",
    )

    # a projecao local precisa concordar com a haversine
    x0, y0 = proj.para_xy(-23.5505, -46.6333)
    x1, y1 = proj.para_xy(-23.6000, -46.7200)
    d_proj = float(np.hypot(x1 - x0, y1 - y0))
    checar(abs(d_proj - d) / d < 0.01, "erro da projecao < 1%", f"{abs(d_proj-d):.1f} m")

    # referenciamento linear numa rota que vai e volta pela MESMA rua:
    # a deteccao por raio confundiria ida com volta; a abscissa nao confunde.
    ida_lon = np.linspace(-46.70, -46.60, 200)
    volta_lon = ida_lon[::-1]
    lons = np.concatenate([ida_lon, volta_lon])
    lats = np.full(len(lons), -23.55)
    xs, ys = proj.para_xy(lats, lons)
    ref = geo.ReferenciadorLinear(xs, ys)

    xp, yp = proj.para_xy(-23.55, -46.65)  # meio do caminho: 2 candidatos
    s_ida = ref.projetar(float(xp), float(yp), janela=(0, ref.comprimento / 2)).s
    s_volta = ref.projetar(
        float(xp), float(yp), janela=(ref.comprimento / 2, ref.comprimento)
    ).s
    checar(
        s_volta > s_ida + 5000,
        "referenciamento separa ida de volta na mesma rua",
        f"s_ida={s_ida:.0f} m, s_volta={s_volta:.0f} m",
    )

    celula = geo.celula(-23.5505, -46.6333, 9)
    checar(isinstance(celula, str) and len(celula) > 3, "celula gerada", celula)


# ------------------------------------------------------- 2. parsing da API
def teste_parsing() -> None:
    print("\n2) parsing da API")
    ts = datetime(2026, 8, 20, 17, 0, tzinfo=timezone.utc)  # 14:00 em SP

    t = coleta.resolver_horario("14:05", ts)
    checar(
        t is not None and (t - ts).total_seconds() == 300,
        "HH:MM vira instante absoluto",
        str(t),
    )

    ts_noite = datetime(2026, 8, 21, 2, 50, tzinfo=timezone.utc)  # 23:50 em SP
    t2 = coleta.resolver_horario("00:05", ts_noite)
    checar(
        t2 is not None and 0 < (t2 - ts_noite).total_seconds() <= 1200,
        "virada da meia-noite soma um dia",
        str(t2),
    )

    payload_pos = {
        "hr": "14:00",
        "l": [
            {
                "c": "8000-10",
                "cl": 1273,
                "sl": 1,
                "lt0": "TERM. LAPA",
                "lt1": "TERM. PINHEIROS",
                "qv": 2,
                "vs": [
                    {"p": "11433", "a": True, "ta": "2026-08-20T17:00:37Z", "py": -23.55, "px": -46.63},
                    {"p": "11434", "a": False, "ta": "2026-08-20T17:00:41Z", "py": -23.56, "px": -46.64},
                ],
            }
        ],
    }
    linhas = coleta.achatar_posicoes(payload_pos, ts)
    checar(len(linhas) == 2, "achatar /Posicao", f"{len(linhas)} veiculos")
    checar(linhas[0]["letreiro"] == "8000" and linhas[0]["cl"] == 1273, "campos da posicao")
    checar(linhas[0]["ta"].tzinfo is not None, "timestamp do GPS com fuso")

    payload_prev = {
        "hr": "14:00",
        "ps": [
            {
                "cp": 4200953,
                "np": "PARADA TESTE",
                "py": -23.55,
                "px": -46.63,
                "vs": [
                    {"p": "11433", "t": "14:07", "a": True, "ta": "2026-08-20T17:00:37Z", "py": -23.554, "px": -46.636}
                ],
            }
        ],
    }
    prev = coleta.achatar_previsao_linha(payload_prev, 1273, ts, "8000", 1)
    checar(len(prev) == 1, "achatar /Previsao/Linha")
    checar(prev[0]["horizonte_s"] == 420, "horizonte calculado", f"{prev[0]['horizonte_s']} s")

    janela_ok = coleta.dentro_da_janela(ts, coleta._hora_config("04:00", None), coleta._hora_config("01:00", None))
    checar(janela_ok, "janela diaria com virada de dia (14h dentro de 04h-01h)")

    # /Posicao/Linha tem formato DIFERENTE: `vs` vem na raiz, sem info de linha
    payload_linha = {
        "hr": "14:00",
        "vs": [{"p": "11433", "a": True, "ta": "2026-08-20T17:00:37Z", "py": -23.55, "px": -46.63}],
    }
    por_linha = coleta.achatar_posicoes(
        payload_linha, ts, cl_padrao=1273, letreiro_padrao="8000", sentido_padrao=1
    )
    checar(len(por_linha) == 1, "achatar /Posicao/Linha (vs na raiz)")
    checar(
        por_linha and por_linha[0]["cl"] == 1273 and por_linha[0]["letreiro"] == "8000",
        "linha preenchida a partir do parametro",
    )

    # o fuso do campo `t` nao e documentado -> calibracao empirica
    class ClienteFalso:
        """Devolve previsoes cujo `t` esta declaradamente em UTC."""

        def __init__(self, em_utc: bool):
            self.em_utc = em_utc

        def previsao_linha(self, cl):
            agora = datetime.now(timezone.utc)
            alvo = agora + timedelta(minutes=7)
            fuso = timezone.utc if self.em_utc else coleta.TZ_SP
            return {
                "hr": agora.astimezone(coleta.TZ_SP).strftime("%H:%M"),
                "ps": [
                    {
                        "cp": 1,
                        "np": "P",
                        "py": -23.5,
                        "px": -46.6,
                        "vs": [
                            {"p": "1", "t": alvo.astimezone(fuso).strftime("%H:%M")}
                            for _ in range(30)
                        ],
                    }
                ],
            }

    decisao_utc, _ = coleta.calibrar_fuso_previsao(ClienteFalso(True), [1])
    decisao_local, _ = coleta.calibrar_fuso_previsao(ClienteFalso(False), [1])
    checar(decisao_utc == "utc", "calibracao detecta `t` em UTC", decisao_utc)
    checar(decisao_local == "local", "calibracao detecta `t` em horario local", decisao_local)


# ------------------------------------------------- 3. cenario ponta a ponta
def montar_cenario(raiz: Path) -> dict:
    """Uma linha reta com 11 paradas e 3 viagens de horario conhecido."""
    proj = geo.ProjecaoLocal()
    lat = -23.55
    lon_ini, lon_fim = -46.70, -46.60
    cl, letreiro = 9999, "TESTE"

    lons_parada = np.linspace(lon_ini, lon_fim, 11)
    paradas = [
        {
            "cp": 900000 + i,
            "nome": f"PARADA {i}",
            "endereco": "AV TESTE",
            "lat": lat,
            "lon": float(lo),
            "celula": geo.celula(lat, float(lo), 9),
            "atualizado_em": datetime.now(timezone.utc),
        }
        for i, lo in enumerate(lons_parada)
    ]

    velocidade_ms = 20 / 3.6
    dt = 30.0
    comprimento = float(geo.haversine_m(lat, lon_ini, lat, lon_fim))
    n_amostras = int(comprimento / (velocidade_ms * dt)) + 1

    rng = np.random.default_rng(7)
    posicoes: list[dict] = []
    verdade: list[dict] = []

    # 10 partidas espalhadas pelo dia, cada uma com um erro de previsao
    # diferente — assim o dataset final tem variacao para agrupar e modelar
    partidas = [
        datetime(2026, 8, 20, 6, 0, tzinfo=timezone.utc) + timedelta(hours=1.5 * v)
        for v in range(10)
    ]
    erro_por_viagem = [60 + 70 * v for v in range(10)]  # 60 s a 690 s

    for v, partida in enumerate(partidas):
        prefixo = f"1000{v}"
        for k in range(n_amostras):
            avanco = velocidade_ms * dt * k
            frac = min(avanco / comprimento, 1.0)
            lo = lon_ini + frac * (lon_fim - lon_ini)
            # ruido de GPS urbano: ~8 m de desvio
            ruido_lat = rng.normal(0, 8) / 111_000
            ruido_lon = rng.normal(0, 8) / 102_000
            ta = partida + timedelta(seconds=dt * k)
            posicoes.append(
                {
                    "ts_coleta": ta,
                    "hr_api": ta.strftime("%H:%M"),
                    "cl": cl,
                    "letreiro": letreiro,
                    "sentido": 1,
                    "destino": "FIM",
                    "prefixo": prefixo,
                    "acessivel": True,
                    "ta": ta,
                    "lat": lat + ruido_lat,
                    "lon": float(lo) + ruido_lon,
                }
            )
        for i, lo in enumerate(lons_parada):
            dist = float(geo.haversine_m(lat, lon_ini, lat, float(lo)))
            verdade.append(
                {
                    "cl": cl,
                    "cp": 900000 + i,
                    "prefixo": prefixo,
                    "t_real": partida + timedelta(seconds=dist / velocidade_ms),
                }
            )

    # previsoes: a API "promete" 120 s ANTES do que vai acontecer,
    # consultadas 10 minutos antes da chegada
    previsoes: list[dict] = []
    erros_injetados: list[float] = []
    for v in verdade:
        indice = int(str(v["prefixo"])[-1])
        t_prev = (
            v["t_real"] - timedelta(seconds=erro_por_viagem[indice])
        ).replace(second=0, microsecond=0)
        ts_coleta = t_prev - timedelta(seconds=600)
        erros_injetados.append((v["t_real"] - t_prev).total_seconds())
        previsoes.append(
            {
                "ts_coleta": ts_coleta,
                "hr_api": ts_coleta.strftime("%H:%M"),
                "cl": cl,
                "letreiro": letreiro,
                "sentido": 1,
                "cp": v["cp"],
                "parada_nome": f"PARADA {v['cp'] - 900000}",
                "parada_lat": lat,
                "parada_lon": float(lons_parada[v["cp"] - 900000]),
                "prefixo": v["prefixo"],
                "acessivel": True,
                "t_previsto_str": t_prev.strftime("%H:%M"),
                "t_previsto": t_prev,
                "horizonte_s": int((t_prev - ts_coleta).total_seconds()),
                "ta": ts_coleta,
                "lat": lat,
                "lon": float(lons_parada[v["cp"] - 900000]) - 0.005,
            }
        )

    linhas_cat = [
        {
            "cl": cl,
            "letreiro": letreiro,
            "letreiro_completo": "TESTE-10",
            "tl": 10,
            "sentido": 1,
            "circular": False,
            "terminal_principal": "INICIO",
            "terminal_secundario": "FIM",
            "atualizado_em": datetime.now(timezone.utc),
        }
    ]
    vinculos = [
        {
            "cl": cl,
            "cp": p["cp"],
            "ordem_api": i,
            "ordem": None,
            "s_m": None,
            "dist_tracado_m": None,
            "atualizado_em": datetime.now(timezone.utc),
        }
        for i, p in enumerate(paradas)
    ]

    return {
        "paradas": paradas,
        "posicoes": posicoes,
        "previsoes": previsoes,
        "verdade": pd.DataFrame(verdade),
        "linhas": linhas_cat,
        "vinculos": vinculos,
        "cl": cl,
        "erro_mediano_injetado": float(np.median(erros_injetados)),
    }


def teste_pipeline(raiz: Path) -> None:
    print("\n3) pipeline ponta a ponta (dados sinteticos)")
    os.environ["OLHOVIVO_ARMAZENAMENTO__RAIZ"] = str(raiz)
    cfg = config.carregar()
    arm = Armazenamento(cfg, backends=["parquet"])

    cenario = montar_cenario(raiz)

    arm.salvar_catalogo("linhas", cenario["linhas"], ESQUEMA_LINHAS)
    arm.salvar_catalogo("paradas", cenario["paradas"], ESQUEMA_PARADAS)
    arm.salvar_catalogo("linha_parada", cenario["vinculos"], ESQUEMA_LINHA_PARADA)

    esc_pos = EscritorParquet(raiz, "posicoes", ESQUEMA_POSICOES)
    esc_pos.adicionar(cenario["posicoes"])
    esc_pos.fechar()

    esc_prev = EscritorParquet(raiz, "previsoes", ESQUEMA_PREVISOES)
    esc_prev.adicionar(cenario["previsoes"])
    esc_prev.fechar()

    lidas = arm.ler_bruto("posicoes")
    checar(
        len(lidas) == len(cenario["posicoes"]),
        "Parquet gravado e relido",
        f"{len(lidas)} linhas",
    )

    resumo_ch = chegadas.processar(cfg, arm, conf_minima=0.0)
    checar(resumo_ch["chegadas"] > 0, "chegadas detectadas", str(resumo_ch["chegadas"]))
    checar(
        resumo_ch["cobertura_media_tracado"] > 0.9,
        "tracado cobre as paradas",
        f"{resumo_ch['cobertura_media_tracado']:.2f}",
    )

    detectadas = arm.ler_derivado("chegadas")
    detectadas["t_chegada"] = pd.to_datetime(detectadas["t_chegada"], utc=True)
    verdade = cenario["verdade"]
    verdade["t_real"] = pd.to_datetime(verdade["t_real"], utc=True)

    conferencia = detectadas.merge(verdade, on=["cl", "cp", "prefixo"], how="inner")
    erro_deteccao = (
        (conferencia["t_chegada"] - conferencia["t_real"]).dt.total_seconds().abs()
    )
    checar(
        len(conferencia) >= len(verdade) * 0.9,
        "quase toda chegada verdadeira foi encontrada",
        f"{len(conferencia)}/{len(verdade)}",
    )
    checar(
        float(erro_deteccao.median()) < 15,
        "erro mediano da deteccao < 15 s",
        f"{erro_deteccao.median():.1f} s",
    )
    checar(
        float(erro_deteccao.quantile(0.9)) < 40,
        "p90 do erro de deteccao < 40 s",
        f"{erro_deteccao.quantile(0.9):.1f} s",
    )

    resumo_ca = casamento.executar(cfg, arm, conf_minima=0.0)
    checar(resumo_ca["validas"] > 0, "pares previsao x chegada", str(resumo_ca["validas"]))
    checar(
        resumo_ca["taxa_casamento"] > 0.9,
        "taxa de casamento alta",
        f"{resumo_ca['taxa_casamento']:.2f}",
    )
    esperado = cenario["erro_mediano_injetado"]
    checar(
        abs(resumo_ca["erro_mediano_s"] - esperado) <= 5,
        "erro reconstruido bate com o erro injetado",
        f"medido {resumo_ca['erro_mediano_s']:.0f} s vs injetado {esperado:.0f} s",
    )

    df_final = arm.ler_derivado("previsao_realizado")
    checar(
        set(["celula", "faixa", "horizonte_s", "erro_s"]).issubset(df_final.columns),
        "dataset final tem as colunas de analise",
    )
    validas = {"madrugada", "pico_manha", "entrepico", "pico_tarde", "noite"}
    checar(
        set(df_final["faixa"].unique()).issubset(validas)
        and df_final["faixa"].nunique() >= 2,
        "faixas horarias atribuidas",
        ", ".join(sorted(df_final["faixa"].unique())),
    )

    teste_analise(cfg, arm)
    arm.fechar()


def teste_analise(cfg, arm) -> None:
    """Atributos para os modelos e agrupamento das regioes."""
    print("\n3b) atributos e regioes")
    from olhovivo import features

    percursos = features.tabela_percursos(arm)
    checar(
        len(percursos) > 0 and percursos["tempo_percurso_s"].median() > 0,
        "tempo de percurso entre paradas consecutivas",
        f"{len(percursos)} trechos, mediana {percursos['tempo_percurso_s'].median():.0f} s"
        if len(percursos)
        else "vazio",
    )

    tabela = features.montar_tabela(arm, min_confianca=0.0)
    faltando = [c for c in features.COLUNAS_MODELO if c not in tabela.columns]
    checar(not faltando, "todas as colunas do modelo presentes", ", ".join(faltando))
    checar(
        tabela["alvo_erro_s"].notna().all()
        and tabela["alvo_atrasado"].isin([0, 1]).all(),
        "alvos de regressao e classificacao montados",
        f"{int(tabela['alvo_atrasado'].sum())} de {len(tabela)} acima de 5 min",
    )

    seq = features.montar_sequencias(arm, bin_min=60, passos=3, min_pontos=1)
    checar(isinstance(seq, dict) and "amostras" in seq, "sequencias LSTM geradas", str(seq))

    grafo = features.montar_grafo(arm)
    checar(
        grafo["nos"] > 0 and grafo["arestas"] > 0,
        "grafo da rede montado",
        f"{grafo['nos']} nos, {grafo['arestas']} arestas",
    )

    resultado = regioes.executar(
        cfg,
        arm,
        algoritmos=("kmeans", "dbscan", "hdbscan", "st-dbscan"),
        modo="eventos",
        limiar_atraso_s=300,
    )
    com_erro = [a for a, r in resultado.items() if isinstance(r, dict) and "erro" in r]
    checar(not com_erro, "os 4 algoritmos rodaram sem erro", ", ".join(com_erro))
    checar(
        resultado.get("kmeans", {}).get("clusters", 0) > 0,
        "K-means produziu regioes",
        str(resultado.get("kmeans", {}).get("clusters")),
    )
    checar(
        (arm.raiz / "derivado" / "regioes.geojson").exists(),
        "GeoJSON das regioes exportado",
    )

    teste_modelos(arm)


def teste_modelos(arm) -> None:
    """
    Fumaca dos tres modelos. Pula o que nao estiver instalado.

    Aqui NAO se avalia qualidade: o cenario sintetico e deterministico e
    qualquer modelo acerta. O que se verifica e que o codigo roda ponta a
    ponta — divisao temporal, treino, metricas — sem estourar.
    """
    print("\n3c) modelos (fumaca)")
    import pandas as pd

    from olhovivo import modelos

    derivado = arm.raiz / "derivado"

    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("  [pulado] xgboost nao instalado")
    else:
        saida = modelos.treinar_xgboost(pd.read_parquet(derivado / "modelagem.parquet"))
        r = saida["resultado"]
        checar(
            r["regressao"]["n"] > 0 and "importancia" in r,
            "XGBoost treinou e mediu",
            f"MAE {r['regressao']['mae_s']:.0f} s, {len(r['quantis'])} quantis",
        )
        checar(
            "por_horizonte" in r and "classificacao" in r,
            "avaliacao estratificada por horizonte + probabilidade de atraso",
        )

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  [pulado] torch nao instalado (LSTM e GNN)")
        return

    seq = np.load(derivado / "sequencias_lstm.npz", allow_pickle=True)
    if seq["X"].shape[0] >= 20:
        r = modelos.treinar_lstm(derivado / "sequencias_lstm.npz", epocas=3, paciencia=2)
        checar("mae_s" in r["resultado"], "LSTM treinou", f"MAE {r['resultado']['mae_s']:.0f} s")
    else:
        print("  [pulado] sequencias insuficientes para a LSTM")

    r = modelos.treinar_gnn(derivado / "grafo.npz", epocas=40)
    checar("mae_s" in r["resultado"], "GNN treinou", f"MAE {r['resultado']['mae_s']:.0f} s")


# ------------------------------------------------------------ 4. ST-DBSCAN
def teste_stdbscan() -> None:
    print("\n4) ST-DBSCAN")
    rng = np.random.default_rng(3)
    base = datetime(2026, 8, 20, tzinfo=timezone.utc)

    # MESMO lugar, dois horarios diferentes: um DBSCAN espacial veria 1 cluster;
    # o ST-DBSCAN tem que ver 2.
    registros = []
    for hora, rotulo in ((8, "manha"), (18, "tarde")):
        for _ in range(60):
            registros.append(
                {
                    "lat": -23.55 + rng.normal(0, 0.0008),
                    "lon": -46.63 + rng.normal(0, 0.0008),
                    "t": base + timedelta(hours=hora, seconds=float(rng.normal(0, 300))),
                    "erro_s": 600.0,
                    "peso": 1.0,
                    "grupo": rotulo,
                }
            )
    df = pd.DataFrame(registros)
    proj = geo.ProjecaoLocal()
    x, y = proj.para_xy(df["lat"].to_numpy(), df["lon"].to_numpy())
    df["x_m"], df["y_m"] = np.asarray(x), np.asarray(y)

    rotulos = regioes.rodar_st_dbscan(df, 400.0, 1800.0, 10)
    n_clusters = len({r for r in rotulos if r >= 0})
    checar(n_clusters == 2, "separa manha de tarde no mesmo ponto", f"{n_clusters} clusters")

    espacial = regioes.rodar_dbscan(df, 400.0, 10)
    checar(
        len({r for r in espacial if r >= 0}) == 1,
        "DBSCAN puramente espacial funde os dois (comportamento esperado)",
    )

    clusters = regioes.resumir_clusters(df, rotulos, "st-dbscan", proj)
    checar(len(clusters) == 2 and clusters[0]["envoltoria"] is not None, "resumo com geometria")


def teste_gtfs_real(raiz: Path) -> None:
    """
    Integracao com a geometria REAL de uma linha da SPTrans.

    Roda so se o GTFS ja tiver sido baixado (`python -m olhovivo gtfs`). Aqui a
    rota tem curvas, alcas e retornos de verdade — e um teste bem mais duro que
    a linha reta sintetica.
    """
    print("\n5) traçado real do GTFS")
    origem = Path(__file__).resolve().parent.parent / "dados" / "gtfs" / "sptrans-gtfs.zip"
    if not origem.exists():
        print("  [pulado] GTFS nao baixado — rode: python -m olhovivo gtfs")
        return

    from olhovivo.gtfs import GTFS

    destino = raiz / "gtfs"
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origem, destino / "sptrans-gtfs.zip")

    feed = GTFS(destino / "sptrans-gtfs.zip")
    alvo = None
    for route_id in ["8000-10", "875A-10", "702U-10"]:
        dados = feed.tracado_de(route_id, 1)
        if dados is not None and len(dados["paradas"]) >= 10:
            alvo = (route_id, dados)
            break
    if alvo is None:
        print("  [pulado] nenhuma rota de referencia encontrada no feed")
        return
    route_id, dados = alvo

    proj = geo.ProjecaoLocal()
    xs, ys = proj.para_xy(dados["lat"], dados["lon"])
    xs, ys = np.asarray(xs), np.asarray(ys)

    # amostra o traçado a cada 5 m — serve tanto para simular o onibus quanto
    # para calcular a verdade por forca bruta, sem usar o codigo sob teste
    fx, fy = geo.densificar(xs, ys, passo_m=5.0)
    passo = np.hypot(np.diff(fx), np.diff(fy))
    arco = np.concatenate([[0.0], np.cumsum(passo)])
    comprimento = float(arco[-1])
    flat, flon = proj.para_latlon(fx, fy)

    paradas_gtfs = dados["paradas"].dropna(subset=["stop_lat", "stop_lon"])
    verdade_s = {}
    for r in paradas_gtfs.itertuples():
        d = geo.haversine_m(flat, flon, r.stop_lat, r.stop_lon)
        i = int(np.argmin(d))
        if float(d[i]) <= 120:  # parada realmente sobre o traçado
            verdade_s[int(r.stop_id)] = (float(arco[i]), float(d[i]))

    checar(
        len(verdade_s) >= 8,
        "paradas reais sobre o traçado",
        f"{len(verdade_s)}/{len(paradas_gtfs)} em {route_id}",
    )

    cl = 8888
    velocidade = 22 / 3.6
    dt = 30.0
    rng = np.random.default_rng(11)
    posicoes: list[dict] = []
    verdade: list[dict] = []

    for v, partida in enumerate(
        [
            datetime(2026, 8, 20, 8, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 20, 17, 30, tzinfo=timezone.utc),
        ]
    ):
        prefixo = f"2000{v}"
        n = int(comprimento / (velocidade * dt)) + 1
        for k in range(n):
            avanco = min(velocidade * dt * k, comprimento)
            i = int(np.searchsorted(arco, avanco))
            i = min(i, len(flat) - 1)
            ta = partida + timedelta(seconds=dt * k)
            posicoes.append(
                {
                    "ts_coleta": ta,
                    "hr_api": ta.strftime("%H:%M"),
                    "cl": cl,
                    "letreiro": route_id.split("-")[0],
                    "sentido": 1,
                    "destino": "FIM",
                    "prefixo": prefixo,
                    "acessivel": True,
                    "ta": ta,
                    "lat": float(flat[i]) + rng.normal(0, 8) / 111_000,
                    "lon": float(flon[i]) + rng.normal(0, 8) / 102_000,
                }
            )
        for stop_id, (s_k, _) in verdade_s.items():
            verdade.append(
                {
                    "cl": cl,
                    "cp": stop_id,
                    "prefixo": prefixo,
                    "t_real": partida + timedelta(seconds=s_k / velocidade),
                }
            )

    os.environ["OLHOVIVO_ARMAZENAMENTO__RAIZ"] = str(raiz)
    cfg = config.carregar()
    arm = Armazenamento(cfg, backends=["parquet"])

    arm.salvar_catalogo(
        "linhas",
        [
            {
                "cl": cl,
                "letreiro": route_id.split("-")[0],
                "letreiro_completo": route_id,
                "tl": None,
                "sentido": 1,
                "circular": False,
                "terminal_principal": None,
                "terminal_secundario": None,
                "atualizado_em": datetime.now(timezone.utc),
            }
        ],
        ESQUEMA_LINHAS,
    )
    arm.salvar_catalogo(
        "paradas",
        [
            {
                "cp": int(r.stop_id),
                "nome": str(r.stop_name),
                "endereco": None,
                "lat": float(r.stop_lat),
                "lon": float(r.stop_lon),
                "celula": geo.celula(float(r.stop_lat), float(r.stop_lon), 9),
                "atualizado_em": datetime.now(timezone.utc),
            }
            for r in paradas_gtfs.itertuples()
        ],
        ESQUEMA_PARADAS,
    )
    arm.salvar_catalogo(
        "linha_parada",
        [
            {
                "cl": cl,
                "cp": int(r.stop_id),
                "ordem_api": int(r.stop_sequence),
                "ordem": None,
                "s_m": None,
                "dist_tracado_m": None,
                "atualizado_em": datetime.now(timezone.utc),
            }
            for r in paradas_gtfs.itertuples()
        ],
        ESQUEMA_LINHA_PARADA,
    )

    esc = EscritorParquet(raiz, "posicoes", ESQUEMA_POSICOES)
    esc.adicionar(posicoes)
    esc.fechar()

    resumo = chegadas.processar(cfg, arm, conf_minima=0.0)
    checar(
        resumo["tracado_gtfs"] == 1,
        "traçado veio do GTFS (nao da inferencia)",
        f"gtfs={resumo['tracado_gtfs']} observado={resumo['tracado_observado']}",
    )

    detectadas = arm.ler_derivado("chegadas")
    detectadas["t_chegada"] = pd.to_datetime(detectadas["t_chegada"], utc=True)
    vdf = pd.DataFrame(verdade)
    vdf["t_real"] = pd.to_datetime(vdf["t_real"], utc=True)
    conf = detectadas.merge(vdf, on=["cl", "cp", "prefixo"], how="inner")

    checar(
        len(conf) >= len(vdf) * 0.8,
        "cobertura das chegadas em geometria real",
        f"{len(conf)}/{len(vdf)}",
    )
    if len(conf):
        erro = (conf["t_chegada"] - conf["t_real"]).dt.total_seconds().abs()
        checar(
            float(erro.median()) < 20,
            "erro mediano em geometria real < 20 s",
            f"{erro.median():.1f} s",
        )
        checar(
            float(erro.quantile(0.9)) < 60,
            "p90 em geometria real < 60 s",
            f"{erro.quantile(0.9):.1f} s",
        )
    arm.fechar()


def main() -> int:
    print("=" * 62)
    print("AUTOTESTE — pipeline Olho Vivo (sem chamar a API)")
    print("=" * 62)
    tmp = Path(tempfile.mkdtemp(prefix="olhovivo_teste_"))
    try:
        teste_geo()
        teste_parsing()
        teste_pipeline(tmp)
        teste_stdbscan()
        teste_gtfs_real(Path(tempfile.mkdtemp(prefix="olhovivo_gtfs_", dir=tmp)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("OLHOVIVO_ARMAZENAMENTO__RAIZ", None)

    print("\n" + "=" * 62)
    if FALHAS:
        print(f"{len(FALHAS)} FALHA(S):")
        for f in FALHAS:
            print(f"  - {f}")
        return 1
    print("todos os testes passaram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
