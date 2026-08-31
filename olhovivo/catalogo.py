"""
Mapeamento estatico da rede: linhas, paradas, vinculo linha-parada, corredores
e empresas.

Enumeracao das linhas
---------------------
`/Linha/Buscar` e um endpoint de BUSCA, nao de listagem: nao existe "me devolva
todas as linhas". Duas estrategias, usadas juntas:

1. **Por posicao (primaria).** Um unico `GET /Posicao` devolve todas as linhas
   com veiculo em circulacao naquele instante, ja com `cl`, letreiro e sentido.
   Rodando em horario de pico isso cobre praticamente a rede inteira com UMA
   requisicao. E o jeito mais confiavel e mais barato.

2. **Por varredura de termos (complementar).** Consulta `/Linha/Buscar` com
   cada prefixo de 1 caractere ("0".."9", "A".."Z") e, no modo profundo, com os
   prefixos de 2 digitos. Pega linhas que nao estavam circulando na hora do
   snapshot (madrugada, linhas de reforco, atendimento de fim de semana).

Ordem das paradas
-----------------
`/Parada/BuscarParadasPorLinha` devolve as paradas da linha, mas a ordem do
array NAO e garantida como a ordem do itinerario. Aqui ela e guardada como
`ordem_api` e a ordem real e recalculada depois, em `chegadas.py`, projetando
cada parada sobre o tracado observado da linha. Nunca confie em `ordem_api`
para calcular tempo de percurso.
"""

from __future__ import annotations

import logging
import string
from datetime import datetime, timezone
from typing import Iterable, Sequence

from . import geo
from .api import ClienteOlhoVivo
from .storage import (
    ESQUEMA_LINHA_PARADA,
    ESQUEMA_LINHAS,
    ESQUEMA_PARADAS,
    Armazenamento,
    agora_utc,
)

log = logging.getLogger("olhovivo.catalogo")

TERMOS_BASE = list(string.digits) + list(string.ascii_uppercase)


# ------------------------------------------------------------------- linhas
def linhas_de_posicao(cli: ClienteOlhoVivo) -> dict[int, dict]:
    """Extrai o catalogo de linhas de um snapshot de /Posicao."""
    payload = cli.posicoes()
    achadas: dict[int, dict] = {}
    for linha in payload.get("l") or []:
        cl = linha.get("cl")
        if cl is None:
            continue
        letreiro_completo = linha.get("c") or ""
        # atencao: lt0 e o letreiro de DESTINO e lt1 o de ORIGEM. Aqui eles
        # entram como um preenchimento provisorio de terminal; /Linha/Buscar
        # sobrescreve depois com os campos oficiais tp/ts.
        achadas[int(cl)] = {
            "cl": int(cl),
            "letreiro": letreiro_completo.split("-")[0] or None,
            "letreiro_completo": letreiro_completo or None,
            "tl": None,
            "sentido": int(linha.get("sl") or 0),
            "circular": None,
            "terminal_principal": linha.get("lt1"),
            "terminal_secundario": linha.get("lt0"),
            "atualizado_em": agora_utc(),
        }
    log.info("snapshot /Posicao: %d linhas em circulacao", len(achadas))
    return achadas


def _normalizar_linha(item: dict) -> dict:
    letreiro = str(item.get("lt") or "").strip()
    tl = item.get("tl")
    completo = f"{letreiro}-{tl}" if letreiro and tl is not None else letreiro or None
    return {
        "cl": int(item["cl"]),
        "letreiro": letreiro or None,
        "letreiro_completo": completo,
        "tl": int(tl) if tl is not None else None,
        "sentido": int(item.get("sl") or 0),
        "circular": bool(item.get("lc")) if item.get("lc") is not None else None,
        "terminal_principal": item.get("tp"),
        "terminal_secundario": item.get("ts"),
        "atualizado_em": agora_utc(),
    }


def linhas_por_varredura(cli: ClienteOlhoVivo, profundo: bool = False) -> dict[int, dict]:
    """Varre `/Linha/Buscar` com prefixos ate cobrir o que a busca alcanca."""
    termos: list[str] = list(TERMOS_BASE)
    if profundo:
        termos += [f"{a}{b}" for a in string.digits for b in string.digits]

    achadas: dict[int, dict] = {}
    for i, termo in enumerate(termos, 1):
        try:
            for item in cli.buscar_linhas(termo):
                if item.get("cl") is None:
                    continue
                achadas[int(item["cl"])] = _normalizar_linha(item)
        except Exception as exc:
            log.warning("varredura falhou no termo %r: %s", termo, exc)
        if i % 10 == 0:
            log.info("varredura %d/%d termos, %d linhas", i, len(termos), len(achadas))
    return achadas


def resolver_letreiros(cli: ClienteOlhoVivo, letreiros: Iterable[str]) -> list[dict]:
    """Resolve letreiros ("8000") para os codigos `cl` dos DOIS sentidos."""
    resultado: dict[int, dict] = {}
    for letreiro in letreiros:
        letreiro = str(letreiro).strip()
        if not letreiro:
            continue
        try:
            achados = cli.buscar_linhas(letreiro)
        except Exception as exc:
            log.error("nao consegui resolver o letreiro %r: %s", letreiro, exc)
            continue
        casados = [
            it
            for it in achados
            if str(it.get("lt") or "").strip().upper() == letreiro.upper()
        ]
        if not casados:
            log.warning("letreiro %r nao encontrado na API", letreiro)
            continue
        for item in casados:
            resultado[int(item["cl"])] = _normalizar_linha(item)
    return list(resultado.values())


# ------------------------------------------------------------------ paradas
def _normalizar_parada(item: dict, resolucao_h3: int) -> dict | None:
    lat, lon = item.get("py"), item.get("px")
    if lat is None or lon is None:
        return None
    return {
        "cp": int(item["cp"]),
        "nome": item.get("np"),
        "endereco": item.get("ed"),
        "lat": float(lat),
        "lon": float(lon),
        "celula": geo.celula(float(lat), float(lon), resolucao_h3),
        "atualizado_em": agora_utc(),
    }


def paradas_das_linhas(
    cli: ClienteOlhoVivo,
    codigos: Sequence[int],
    resolucao_h3: int = 9,
    usar_previsao: bool = True,
) -> tuple[dict[int, dict], list[dict]]:
    """
    Baixa as paradas de cada linha. Devolve (paradas, vinculos linha-parada).

    Duas fontes, e a segunda importa mais do que parece:

    * `/Parada/BuscarParadasPorLinha` — a fonte "obvia", mas **incompleta**.
      Medido sobre as 150 linhas de maior frota: **105 delas (70%)
      devolveram ZERO paradas**, e o resultado se repete quando se reconsulta
      devagar, entao nao e throttling — e o cadastro mesmo.

    * `/Previsao/Linha` — devolve os pontos onde a linha TEM previsao, com
      coordenada, e cobre muito mais. A linha 3459-10 devolve 0 paradas no
      endpoint acima e **75 pontos** aqui.

    Como o estudo mede erro de previsao, o universo que interessa e justamente
    o das paradas com previsao. Por isso as duas fontes sao unidas, com a
    previsao mandando. O nome vem vazio em `/Previsao/Linha`, entao fica nulo e
    e preenchido depois pelo GTFS (o casamento cp <-> stop_id deu 100%).
    """
    paradas: dict[int, dict] = {}
    vinculos: dict[tuple[int, int], dict] = {}
    total = len(codigos)
    de_previsao = 0

    for i, cl in enumerate(codigos, 1):
        encontradas: list[tuple[dict, int | None]] = []

        try:
            for ordem, item in enumerate(cli.paradas_por_linha(cl)):
                p = _normalizar_parada(item, resolucao_h3)
                if p:
                    encontradas.append((p, ordem))
        except Exception as exc:
            log.warning("paradas da linha %s falharam: %s", cl, exc)

        if usar_previsao:
            try:
                payload = cli.previsao_linha(cl)
                for ponto in payload.get("ps") or []:
                    p = _normalizar_parada(ponto, resolucao_h3)
                    if p and p["cp"] not in {q["cp"] for q, _ in encontradas}:
                        encontradas.append((p, None))
                        de_previsao += 1
            except Exception as exc:
                log.warning("previsao da linha %s falhou: %s", cl, exc)

        for parada, ordem in encontradas:
            # o registro com nome preenchido (vindo de /Parada) tem prioridade
            existente = paradas.get(parada["cp"])
            if existente is None or (not existente.get("nome") and parada.get("nome")):
                paradas[parada["cp"]] = parada
            vinculos[(int(cl), parada["cp"])] = {
                "cl": int(cl),
                "cp": parada["cp"],
                "ordem_api": ordem,
                "ordem": None,
                "s_m": None,
                "dist_tracado_m": None,
                "atualizado_em": agora_utc(),
            }

        if i % 25 == 0 or i == total:
            log.info(
                "paradas: %d/%d linhas, %d paradas unicas (%d so da previsao)",
                i,
                total,
                len(paradas),
                de_previsao,
            )
    return paradas, list(vinculos.values())


def completar_nomes_pelo_gtfs(paradas: dict[int, dict], raiz) -> int:
    """
    `/Previsao/Linha` devolve o nome da parada VAZIO. Preenche pelo GTFS.

    O casamento `cp` <-> `stop_id` foi medido em 100% por igualdade de codigo
    no feed de 2026, entao aqui e um join direto — sem heuristica de distancia.
    """
    from pathlib import Path

    zip_gtfs = Path(raiz) / "gtfs" / "sptrans-gtfs.zip"
    if not zip_gtfs.exists():
        return 0

    try:
        from .gtfs import GTFS

        nomes = GTFS(zip_gtfs).paradas.set_index("stop_id")["stop_name"].to_dict()
    except Exception as exc:
        log.warning("nao consegui ler nomes do GTFS: %s", exc)
        return 0

    preenchidos = 0
    for cp, parada in paradas.items():
        if not parada.get("nome"):
            nome = nomes.get(cp)
            if nome:
                parada["nome"] = str(nome)
                preenchidos += 1
    if preenchidos:
        log.info("nomes preenchidos pelo GTFS: %d paradas", preenchidos)
    return preenchidos


# --------------------------------------------------------------- sincronizar
def sincronizar(
    cfg,
    cli: ClienteOlhoVivo,
    arm: Armazenamento,
    letreiros: Sequence[str] | None = None,
    todas: bool = False,
    varredura: bool = True,
    profundo: bool = False,
) -> dict:
    """
    Monta o catalogo e grava em Parquet + MongoDB.

    `letreiros`  -> restringe as paradas a essas linhas (coleta monitorada)
    `todas`      -> baixa as paradas de todas as linhas descobertas
    """
    resolucao_h3 = int(cfg.get("geo.h3_resolucao", 9))

    linhas: dict[int, dict] = {}
    linhas.update(linhas_de_posicao(cli))
    if varredura:
        for cl, dados in linhas_por_varredura(cli, profundo=profundo).items():
            # os dados de /Linha/Buscar sao mais completos: sobrescrevem
            linhas[cl] = dados
    log.info("catalogo de linhas: %d codigos (linha x sentido)", len(linhas))

    if todas:
        alvo = sorted(linhas)
    elif letreiros:
        monitoradas = resolver_letreiros(cli, letreiros)
        for m in monitoradas:
            linhas.setdefault(m["cl"], m)
            linhas[m["cl"]].update({k: v for k, v in m.items() if v is not None})
        alvo = sorted({m["cl"] for m in monitoradas})
    else:
        alvo = []

    paradas, vinculos = ({}, [])
    if alvo:
        paradas, vinculos = paradas_das_linhas(cli, alvo, resolucao_h3)
        completar_nomes_pelo_gtfs(paradas, arm.raiz)

    # corredores e empresas (contexto para a analise por regiao)
    try:
        corredores = [
            {"cc": int(c["cc"]), "nome": c.get("nc")}
            for c in cli.corredores()
            if c.get("cc") is not None
        ]
    except Exception as exc:
        log.warning("corredores indisponiveis: %s", exc)
        corredores = []

    empresas = _achatar_empresas(cli)

    # ---- persistencia
    lista_linhas = [linhas[cl] for cl in sorted(linhas)]
    arm.salvar_catalogo("linhas", lista_linhas, ESQUEMA_LINHAS)
    if paradas:
        arm.salvar_catalogo(
            "paradas", [paradas[cp] for cp in sorted(paradas)], ESQUEMA_PARADAS
        )
        arm.salvar_catalogo("linha_parada", vinculos, ESQUEMA_LINHA_PARADA)
    if corredores:
        arm.salvar_catalogo("corredores", corredores, None)
    if empresas:
        arm.salvar_catalogo("empresas", empresas, None)

    if arm.mongo:
        arm.mongo.upsert_linhas(lista_linhas)
        if paradas:
            arm.mongo.upsert_paradas(paradas.values())
            arm.mongo.upsert_linha_parada(vinculos)

    resumo = {
        "linhas": len(linhas),
        "linhas_com_paradas": len(alvo),
        "paradas": len(paradas),
        "vinculos": len(vinculos),
        "corredores": len(corredores),
        "empresas": len(empresas),
    }
    arm.eventos.registrar("catalogo_sincronizado", **resumo)
    log.info("catalogo sincronizado: %s", resumo)
    return resumo


def _achatar_empresas(cli: ClienteOlhoVivo) -> list[dict]:
    """`/Empresa` vem aninhado por area de operacao; achata para uma tabela."""
    try:
        payload = cli.empresas()
    except Exception as exc:
        log.warning("empresas indisponiveis: %s", exc)
        return []

    saida: list[dict] = []
    for area in payload.get("e") or []:
        codigo_area = area.get("a")
        for emp in area.get("e") or []:
            saida.append(
                {
                    "area": int(codigo_area) if codigo_area is not None else None,
                    "codigo": int(emp["c"]) if emp.get("c") is not None else None,
                    "nome": emp.get("n"),
                }
            )
    return saida


def codigos_monitorados(cfg, cli: ClienteOlhoVivo, arm: Armazenamento) -> list[int]:
    """Resolve, a partir da configuracao, quais `cl` a coleta vai acompanhar."""
    if cfg.get("coleta.linhas.incluir_todas", False):
        catalogo = arm.ler_catalogo("linhas")
        if len(catalogo):
            return sorted(int(x) for x in catalogo["cl"].tolist())
        return sorted(linhas_de_posicao(cli))

    letreiros = list(cfg.get("coleta.linhas.letreiros", []) or [])
    codigos = {m["cl"] for m in resolver_letreiros(cli, letreiros)}

    for cc in cfg.get("coleta.linhas.corredores", []) or []:
        try:
            for parada in cli.paradas_por_corredor(int(cc)):
                for linha in parada.get("l") or []:
                    if linha.get("cl") is not None:
                        codigos.add(int(linha["cl"]))
        except Exception as exc:
            log.warning("corredor %s falhou: %s", cc, exc)

    return sorted(codigos)
