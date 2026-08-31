"""
Deteccao da chegada REAL do onibus em cada parada.

A API nao publica horario realizado — so previsao. O horario real precisa ser
inferido das posicoes GPS. O procedimento, por linha (`cl`, que ja e por
sentido):

1. **Tracado de referencia.** Escolhe, entre as viagens observadas, aquela cuja
   polilinha melhor cobre as paradas da linha (maior fracao de paradas a menos
   de `dist_max_parada_tracado_m` do traco, menor distancia media). Nao depende
   de GTFS nem do KMZ: o proprio dado coletado define o itinerario real.

2. **Abscissa curvilinea.** Projeta paradas e posicoes sobre esse tracado. Cada
   parada vira um numero `s_k` (metros desde o inicio) e cada posicao vira
   `s(t)`. Isso resolve o problema de ida e volta na mesma avenida, que quebra
   qualquer deteccao por raio.

3. **Segmentacao em viagens.** Nova viagem quando o veiculo some por mais de
   `gap_viagem_s` ou quando `s` retrocede mais de `retrocesso_viagem_m` (chegou
   ao fim e voltou para o inicio).

4. **Evento de chegada.** Dentro de uma viagem, a chegada na parada k e o
   instante em que `s(t)` cruza `s_k`, interpolado linearmente entre as duas
   amostras que cercam o cruzamento. Quando nao ha cruzamento limpo (falha de
   GPS), cai para aproximacao maxima, com confianca menor.

5. **Confianca.** Cada chegada recebe um score de 0 a 1 que cai conforme o salto
   entre as amostras usadas cresce (em metros e em segundos) e conforme a
   distancia minima ate a parada aumenta. A analise depois filtra por isso —
   melhor descartar do que tratar interpolacao de 900 m como se fosse medida.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import geo
from .storage import ESQUEMA_LINHA_PARADA, Armazenamento, agora_utc

log = logging.getLogger("olhovivo.chegadas")


# ------------------------------------------------------------------ tracado
def _viagens_candidatas(
    df: pd.DataFrame, gap_s: float, min_pontos: int = 25
) -> list[pd.DataFrame]:
    """Quebra as posicoes de cada veiculo em trechos continuos no tempo."""
    saida: list[pd.DataFrame] = []
    for _, grupo in df.groupby("prefixo", sort=False):
        g = grupo.sort_values("ta")
        if len(g) < min_pontos:
            continue
        delta = g["ta"].diff().dt.total_seconds().fillna(0.0)
        bloco = (delta > gap_s).cumsum()
        for _, trecho in g.groupby(bloco, sort=False):
            if len(trecho) >= min_pontos:
                saida.append(trecho)
    return saida


def construir_tracado(
    posicoes: pd.DataFrame,
    paradas: pd.DataFrame,
    proj,
    dist_max_m: float = 120.0,
    gap_s: float = 1200.0,
    max_candidatos: int = 60,
) -> tuple[geo.ReferenciadorLinear, np.ndarray, np.ndarray, dict] | None:
    """
    Escolhe a melhor polilinha de referencia para a linha.

    Retorna (referenciador, lats, lons, diagnostico) ou None se nao der.
    """
    candidatos = _viagens_candidatas(posicoes, gap_s)
    if not candidatos:
        return None

    # candidatos mais longos primeiro: uma viagem completa cobre mais paradas
    candidatos.sort(key=len, reverse=True)
    candidatos = candidatos[:max_candidatos]

    px, py = proj.para_xy(paradas["lat"].to_numpy(), paradas["lon"].to_numpy())
    melhor: tuple[float, Any] | None = None

    for trecho in candidatos:
        xs, ys = proj.para_xy(trecho["lat"].to_numpy(), trecho["lon"].to_numpy())
        xs, ys = geo.simplificar(xs, ys, tolerancia_m=10.0)
        if len(xs) < 5:
            continue
        try:
            ref = geo.ReferenciadorLinear(xs, ys)
        except ValueError:
            continue
        if ref.comprimento < 1500:  # trecho curto demais para ser itinerario
            continue

        distancias = np.array(
            [ref.projetar(float(x), float(y)).distancia for x, y in zip(px, py)]
        )
        cobertura = float((distancias <= dist_max_m).mean())
        dist_media = float(np.median(distancias))
        # cobertura pesa muito mais que proximidade media
        score = cobertura * 100.0 - dist_media / 50.0
        if melhor is None or score > melhor[0]:
            lat_t, lon_t = proj.para_latlon(xs, ys)
            melhor = (
                score,
                (
                    ref,
                    np.asarray(lat_t),
                    np.asarray(lon_t),
                    {
                        "cobertura": round(cobertura, 4),
                        "dist_mediana_m": round(dist_media, 1),
                        "comprimento_m": round(ref.comprimento, 1),
                        "n_pontos": int(len(xs)),
                        "prefixo_origem": str(trecho["prefixo"].iloc[0]),
                        "candidatos_avaliados": len(candidatos),
                    },
                ),
            )

    return melhor[1] if melhor else None


def tracado_do_gtfs(
    gtfs,
    route_id: str,
    sentido: int,
    paradas: pd.DataFrame,
    proj,
    dist_max_m: float = 120.0,
) -> tuple[geo.ReferenciadorLinear, np.ndarray, np.ndarray, dict] | None:
    """
    Traçado oficial do `shapes.txt`, quando a linha casa com o GTFS.

    Preferivel a inferir o itinerario das trajetorias: e o percurso cadastrado,
    nao uma viagem qualquer que pode ter desvio de obra ou terminar cedo. Mesmo
    assim a cobertura das paradas e conferida — feed desatualizado existe, e
    nesse caso o chamador cai para o traçado observado.
    """
    dados = gtfs.tracado_de(route_id, sentido)
    if dados is None:
        return None

    xs, ys = proj.para_xy(dados["lat"], dados["lon"])
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) < 5:
        return None
    try:
        ref = geo.ReferenciadorLinear(xs, ys)
    except ValueError:
        return None

    px, py = proj.para_xy(paradas["lat"].to_numpy(), paradas["lon"].to_numpy())
    distancias = np.array(
        [ref.projetar(float(a), float(b)).distancia for a, b in zip(px, py)]
    )
    cobertura = float((distancias <= dist_max_m).mean())

    return (
        ref,
        np.asarray(dados["lat"]),
        np.asarray(dados["lon"]),
        {
            "origem": "gtfs",
            "route_id": dados["route_id"],
            "trip_id": str(dados["trip_id"]),
            "shape_id": str(dados["shape_id"]),
            "cobertura": round(cobertura, 4),
            "dist_mediana_m": round(float(np.median(distancias)), 1),
            "comprimento_m": round(ref.comprimento, 1),
            "n_pontos": int(len(xs)),
            "paradas_gtfs": int(len(dados["paradas"])),
        },
    )


def ordenar_paradas(
    ref: geo.ReferenciadorLinear, paradas: pd.DataFrame, proj
) -> pd.DataFrame:
    """Projeta as paradas no tracado e devolve a ordem real do itinerario."""
    xs, ys = proj.para_xy(paradas["lat"].to_numpy(), paradas["lon"].to_numpy())
    resultados = [ref.projetar(float(x), float(y)) for x, y in zip(xs, ys)]
    out = paradas.copy()
    out["s_m"] = [r.s for r in resultados]
    out["dist_tracado_m"] = [r.distancia for r in resultados]
    out = out.sort_values("s_m").reset_index(drop=True)
    out["ordem"] = np.arange(len(out), dtype="int32")
    return out


# ----------------------------------------------------------------- chegadas
def _segmentar_viagens(
    ta: np.ndarray, s: np.ndarray, gap_s: float, retrocesso_m: float
) -> np.ndarray:
    """Rotula cada amostra com o indice da viagem a que pertence."""
    if len(ta) == 0:
        return np.array([], dtype="int64")
    dt = np.diff(ta.astype("datetime64[s]").astype("int64"), prepend=ta[0].astype("datetime64[s]").astype("int64"))
    ds = np.diff(s, prepend=s[0])
    corte = (dt > gap_s) | (ds < -retrocesso_m)
    return np.cumsum(corte)


def _chegadas_da_viagem(
    ta: np.ndarray,
    s: np.ndarray,
    dist_tracado: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    paradas: pd.DataFrame,
    cfg_ch: dict,
) -> list[dict]:
    """Cruzamentos de `s` com as abscissas das paradas dentro de uma viagem."""
    if len(s) < 2:
        return []

    raio = float(cfg_ch["raio_chegada_m"])
    salto_max_m = float(cfg_ch["salto_max_interp_m"])
    salto_max_s = float(cfg_ch["salto_max_interp_s"])
    vel_max = float(cfg_ch["velocidade_max_kmh"])

    ta_s = ta.astype("datetime64[ms]").astype("int64") / 1000.0
    s_paradas = paradas["s_m"].to_numpy()
    lat_p = paradas["lat"].to_numpy()
    lon_p = paradas["lon"].to_numpy()

    s_min, s_max = float(np.min(s)), float(np.max(s))
    eventos: list[dict] = []

    for k in range(len(paradas)):
        s_k = float(s_paradas[k])
        if not (s_min - raio <= s_k <= s_max + raio):
            continue

        # primeiro cruzamento ascendente de s_k
        antes = s[:-1] <= s_k
        depois = s[1:] >= s_k
        idx = np.flatnonzero(antes & depois)

        if len(idx):
            i = int(idx[0])
            ds = s[i + 1] - s[i]
            dt = ta_s[i + 1] - ta_s[i]
            f = 0.0 if ds <= 0 else float((s_k - s[i]) / ds)
            f = min(max(f, 0.0), 1.0)
            t_chegada = ta_s[i] + f * dt
            lat_i = lat[i] + f * (lat[i + 1] - lat[i])
            lon_i = lon[i] + f * (lon[i + 1] - lon[i])
            dist_min = float(geo.haversine_m(lat_i, lon_i, lat_p[k], lon_p[k]))
            salto_m = float(abs(ds))
            salto_s = float(dt)
            vel = (salto_m / salto_s * 3.6) if salto_s > 0 else 0.0
            metodo = "cruzamento"
        else:
            # sem cruzamento: usa a amostra mais proxima da parada
            d = geo.haversine_m(lat, lon, lat_p[k], lon_p[k])
            i = int(np.argmin(d))
            dist_min = float(d[i])
            if dist_min > raio:
                continue
            t_chegada = float(ta_s[i])
            salto_m = float(abs(s[min(i + 1, len(s) - 1)] - s[max(i - 1, 0)]))
            salto_s = float(ta_s[min(i + 1, len(s) - 1)] - ta_s[max(i - 1, 0)])
            vel = (salto_m / salto_s * 3.6) if salto_s > 0 else 0.0
            lat_i, lon_i = float(lat[i]), float(lon[i])
            metodo = "aproximacao"

        if vel > vel_max:  # salto de GPS, nao movimento real
            continue
        if dist_min > raio * 3:
            continue
        # interpolar por cima de um buraco enorme e inventar dado, nao medir
        if salto_m > float(cfg_ch["salto_hard_m"]) or salto_s > float(cfg_ch["salto_hard_s"]):
            continue

        c1 = 1.0 if salto_m <= salto_max_m else salto_max_m / salto_m
        c2 = 1.0 if salto_s <= salto_max_s else salto_max_s / salto_s
        c3 = 1.0 if dist_min <= raio else raio / dist_min
        conf = c1 * c2 * c3 * (1.0 if metodo == "cruzamento" else 0.7)

        eventos.append(
            {
                "cp": int(paradas["cp"].iloc[k]),
                "ordem": int(paradas["ordem"].iloc[k]),
                "t_chegada": pd.Timestamp(t_chegada, unit="s", tz="UTC"),
                "metodo": metodo,
                "dist_min_m": round(dist_min, 1),
                "salto_m": round(salto_m, 1),
                "salto_s": round(salto_s, 1),
                "velocidade_kmh": round(vel, 1),
                "confianca": round(float(conf), 3),
                "lat": float(lat_i),
                "lon": float(lon_i),
            }
        )

    return eventos


def detectar_para_linha(
    cl: int,
    posicoes: pd.DataFrame,
    paradas_ord: pd.DataFrame,
    ref: geo.ReferenciadorLinear,
    proj,
    cfg_ch: dict,
    letreiro: str | None,
    sentido: int | None,
) -> list[dict]:
    """Roda a deteccao para todos os veiculos de uma linha."""
    saida: list[dict] = []
    gap_s = float(cfg_ch["gap_viagem_s"])
    retrocesso = float(cfg_ch["retrocesso_viagem_m"])
    dist_max_amostra = float(cfg_ch["dist_max_amostra_m"])
    descartadas = 0

    for prefixo, grupo in posicoes.groupby("prefixo", sort=False):
        g = grupo.dropna(subset=["ta"]).sort_values("ta")
        if len(g) < 3:
            continue

        xs, ys = proj.para_xy(g["lat"].to_numpy(), g["lon"].to_numpy())
        s, dist_tracado = ref.projetar_sequencia(xs, ys)

        # amostra longe do traçado nao e movimento na linha: e garagem, AVL
        # com sentido errado ou coordenada corrompida. Sai antes de segmentar,
        # senao a projecao dela cria cruzamentos que nunca aconteceram.
        no_itinerario = dist_tracado <= dist_max_amostra
        descartadas += int((~no_itinerario).sum())
        if no_itinerario.sum() < 3:
            continue
        s = s[no_itinerario]
        dist_tracado = dist_tracado[no_itinerario]
        g = g.iloc[np.flatnonzero(no_itinerario)]

        ta = g["ta"].to_numpy(dtype="datetime64[ms]")
        lat = g["lat"].to_numpy()
        lon = g["lon"].to_numpy()

        viagens = _segmentar_viagens(ta, s, gap_s, retrocesso)
        for viagem in np.unique(viagens):
            m = viagens == viagem
            if m.sum() < 3:
                continue
            inicio = pd.Timestamp(ta[m][0]).tz_localize("UTC")
            viagem_id = f"{cl}-{prefixo}-{inicio:%Y%m%dT%H%M%S}"
            for evento in _chegadas_da_viagem(
                ta[m], s[m], dist_tracado[m], lat[m], lon[m], paradas_ord, cfg_ch
            ):
                saida.append(
                    {
                        "cl": int(cl),
                        "letreiro": letreiro,
                        "sentido": int(sentido or 0),
                        "prefixo": str(prefixo),
                        "viagem_id": viagem_id,
                        **evento,
                    }
                )

    if descartadas:
        log.info(
            "linha %s: %d amostras fora do itinerario descartadas (>%.0f m do tracado)",
            cl,
            descartadas,
            dist_max_amostra,
        )
    return saida


# ---------------------------------------------------------------- orquestra
def processar(
    cfg,
    arm: Armazenamento,
    inicio: datetime | None = None,
    fim: datetime | None = None,
    codigos: Sequence[int] | None = None,
    conf_minima: float = 0.0,
) -> dict:
    """Le as posicoes coletadas, detecta chegadas e grava a tabela derivada."""
    cfg_ch = {
        "raio_chegada_m": cfg.get("chegadas.raio_chegada_m", 90),
        "salto_max_interp_m": cfg.get("chegadas.salto_max_interp_m", 500),
        "salto_max_interp_s": cfg.get("chegadas.salto_max_interp_s", 120),
        "salto_hard_m": cfg.get("chegadas.salto_hard_m", 1200),
        "salto_hard_s": cfg.get("chegadas.salto_hard_s", 300),
        "dist_max_amostra_m": cfg.get("chegadas.dist_max_amostra_m", 150),
        "gap_viagem_s": cfg.get("chegadas.gap_viagem_s", 900),
        "retrocesso_viagem_m": cfg.get("chegadas.retrocesso_viagem_m", 1500),
        "velocidade_max_kmh": cfg.get("chegadas.velocidade_max_kmh", 80),
    }
    dist_max_tracado = float(cfg.get("chegadas.dist_max_parada_tracado_m", 120))
    proj = geo.obter_projecao(
        int(cfg.get("geo.epsg_metrico", 31983)),
        float(cfg.get("geo.ancora_lat", -23.5505)),
        float(cfg.get("geo.ancora_lon", -46.6333)),
    )

    paradas_cat = arm.ler_catalogo("paradas")
    vinculos = arm.ler_catalogo("linha_parada")
    linhas_cat = arm.ler_catalogo("linhas")
    if paradas_cat.empty or vinculos.empty:
        raise RuntimeError(
            "catalogo vazio: rode `python -m olhovivo catalogo` antes de detectar chegadas"
        )

    meta_linha = {
        int(r.cl): {
            "letreiro": r.letreiro,
            "sentido": r.sentido,
            "route_id": getattr(r, "letreiro_completo", None),
        }
        for r in linhas_cat.itertuples()
    }

    # traçado oficial do GTFS, quando disponivel
    fonte = str(cfg.get("chegadas.fonte_tracado", "gtfs")).lower()
    feed = None
    if fonte == "gtfs":
        zip_gtfs = arm.raiz / "gtfs" / "sptrans-gtfs.zip"
        if zip_gtfs.exists():
            from . import gtfs as mod_gtfs

            try:
                feed = mod_gtfs.GTFS(zip_gtfs)
                feed.rotas  # forca a leitura para falhar cedo se estiver corrompido
                log.info("usando o GTFS da SPTrans como traçado de referencia")
            except Exception as exc:
                log.warning("GTFS ilegivel (%s); caindo para traçado observado", exc)
                feed = None
        else:
            log.info(
                "GTFS ausente (rode `python -m olhovivo gtfs`); "
                "o traçado sera inferido das trajetorias"
            )

    alvos = (
        [int(c) for c in codigos]
        if codigos
        else sorted({int(c) for c in vinculos["cl"].unique()})
    )

    dir_tracados = arm.raiz / "derivado" / "tracados"
    dir_tracados.mkdir(parents=True, exist_ok=True)

    todas: list[dict] = []
    vinculos_atualizados: list[dict] = []
    diagnostico: dict[int, Any] = {}

    for n, cl in enumerate(alvos, 1):
        posicoes = arm.ler_bruto(
            "posicoes",
            inicio=inicio,
            fim=fim,
            colunas=["ts_coleta", "cl", "prefixo", "ta", "lat", "lon"],
            filtro_extra=f"cl = {cl}",
        )
        if posicoes.empty:
            continue
        posicoes = posicoes.dropna(subset=["ta", "lat", "lon"])
        posicoes["ta"] = pd.to_datetime(posicoes["ta"], utc=True)
        posicoes = posicoes.drop_duplicates(subset=["prefixo", "ta"])

        cps = vinculos.loc[vinculos["cl"] == cl, "cp"].unique()
        paradas = paradas_cat[paradas_cat["cp"].isin(cps)].dropna(subset=["lat", "lon"])

        # O conjunto de paradas cresce com a coleta: uma foto de /Previsao/Linha
        # so mostra os pontos com veiculo se aproximando naquele instante, mas a
        # uniao de uma semana de snapshots cobre a linha inteira. Sem isto,
        # perde-se chegada em parada que o catalogo nao pegou no dia da sincronia.
        observadas = arm.ler_bruto(
            "previsoes",
            inicio=inicio,
            fim=fim,
            colunas=["cp", "parada_lat", "parada_lon"],
            filtro_extra=f"cl = {cl}",
        )
        if len(observadas):
            observadas = (
                observadas.dropna(subset=["parada_lat", "parada_lon"])
                .drop_duplicates(subset=["cp"])
                .rename(columns={"parada_lat": "lat", "parada_lon": "lon"})
            )
            novas = observadas[~observadas["cp"].isin(paradas["cp"])]
            if len(novas):
                novas = novas.assign(nome=None, endereco=None, celula=None)
                paradas = pd.concat(
                    [paradas, novas[paradas.columns.intersection(novas.columns)]],
                    ignore_index=True,
                )
                log.info(
                    "linha %s: +%d paradas vistas so nas previsoes coletadas",
                    cl,
                    len(novas),
                )
        if len(paradas) < 3 or len(posicoes) < 50:
            continue

        meta = meta_linha.get(cl, {})
        construido = None
        if feed is not None and meta.get("route_id"):
            try:
                construido = tracado_do_gtfs(
                    feed,
                    str(meta["route_id"]),
                    int(meta.get("sentido") or 1),
                    paradas,
                    proj,
                    dist_max_tracado,
                )
            except Exception as exc:
                log.warning("GTFS falhou para a linha %s: %s", cl, exc)
            # feed desatualizado ou shape errado: melhor o dado observado
            if construido and construido[3]["cobertura"] < 0.5:
                log.info(
                    "linha %s: shape do GTFS cobre so %.0f%% das paradas; "
                    "usando traçado observado",
                    cl,
                    construido[3]["cobertura"] * 100,
                )
                construido = None

        if construido is None:
            construido = construir_tracado(
                posicoes,
                paradas,
                proj,
                dist_max_m=dist_max_tracado,
                gap_s=float(cfg_ch["gap_viagem_s"]),
            )
            if construido is not None:
                construido[3]["origem"] = "observado"

        if construido is None:
            log.warning("linha %s: nao consegui montar tracado de referencia", cl)
            continue
        ref, lat_t, lon_t, diag = construido
        diagnostico[cl] = diag

        (dir_tracados / f"{cl}.json").write_text(
            json.dumps(
                {"cl": cl, "lat": lat_t.tolist(), "lon": lon_t.tolist(), **diag},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if arm.mongo:
            arm.mongo.upsert_tracado(cl, lat_t, lon_t, diag)

        paradas_ord = ordenar_paradas(ref, paradas, proj)
        paradas_ord = paradas_ord[paradas_ord["dist_tracado_m"] <= dist_max_tracado]
        if len(paradas_ord) < 3:
            log.warning("linha %s: paradas longe demais do tracado; pulando", cl)
            continue
        paradas_ord = paradas_ord.reset_index(drop=True)
        paradas_ord["ordem"] = np.arange(len(paradas_ord), dtype="int32")

        for r in paradas_ord.itertuples():
            vinculos_atualizados.append(
                {
                    "cl": int(cl),
                    "cp": int(r.cp),
                    "ordem_api": None,
                    "ordem": int(r.ordem),
                    "s_m": float(r.s_m),
                    "dist_tracado_m": float(r.dist_tracado_m),
                    "atualizado_em": agora_utc(),
                }
            )

        eventos = detectar_para_linha(
            cl,
            posicoes,
            paradas_ord,
            ref,
            proj,
            cfg_ch,
            meta.get("letreiro"),
            meta.get("sentido"),
        )
        todas.extend(e for e in eventos if e["confianca"] >= conf_minima)
        log.info(
            "[%d/%d] linha %s: %d chegadas (tracado %s, cobre %.0f%% das paradas)",
            n,
            len(alvos),
            cl,
            len(eventos),
            diag.get("origem", "observado"),
            diag["cobertura"] * 100,
        )

    if vinculos_atualizados:
        # reescreve o vinculo com a ordem corrigida pelo tracado
        antigo = vinculos.drop_duplicates(subset=["cl", "cp"]).set_index(["cl", "cp"])
        for v in vinculos_atualizados:
            chave = (v["cl"], v["cp"])
            if chave not in antigo.index:
                continue
            # `ordem_api` e nulo de proposito nas paradas que so apareceram em
            # /Previsao/Linha — aquele endpoint nao devolve ordem nenhuma
            valor = antigo.loc[chave, "ordem_api"]
            v["ordem_api"] = int(valor) if pd.notna(valor) else None
        arm.salvar_catalogo("linha_parada", vinculos_atualizados, ESQUEMA_LINHA_PARADA)
        if arm.mongo:
            arm.mongo.upsert_linha_parada(vinculos_atualizados)

    if todas:
        todas.sort(key=lambda e: (e["cl"], e["t_chegada"]))
        arm.salvar_tabela("chegadas", todas)
        if arm.mongo:
            # o Mongo guarda datetime em milissegundos; arredonda antes de
            # converter para nao emitir aviso de nanossegundo descartado
            docs = [
                {
                    **e,
                    "loc": geo.ponto_geojson(e["lat"], e["lon"]),
                    "t_chegada": e["t_chegada"].floor("ms").to_pydatetime(),
                }
                for e in todas
            ]
            arm.mongo.substituir_colecao("chegadas", docs)

    origens = [d.get("origem", "observado") for d in diagnostico.values()]
    resumo = {
        "linhas_processadas": len(diagnostico),
        "chegadas": len(todas),
        "tracado_gtfs": origens.count("gtfs"),
        "tracado_observado": origens.count("observado"),
        "cobertura_media_tracado": round(
            float(np.mean([d["cobertura"] for d in diagnostico.values()]))
            if diagnostico
            else 0.0,
            3,
        ),
    }
    arm.eventos.registrar("chegadas_detectadas", **resumo)
    log.info("chegadas: %s", resumo)
    return resumo
