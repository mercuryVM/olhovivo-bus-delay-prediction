"""Interface de linha de comando: python -m olhovivo <comando>."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import __version__, config


def _log(verboso: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verboso else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _instante(texto: str | None) -> datetime | None:
    if not texto:
        return None
    try:
        dt = datetime.fromisoformat(texto)
    except ValueError:
        raise SystemExit(f"data invalida: {texto!r} (use AAAA-MM-DD ou AAAA-MM-DDTHH:MM)")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _mostrar(titulo: str, dados) -> None:
    print(f"\n== {titulo} ==")
    print(json.dumps(dados, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------- comandos
def _conferir_token(token: str) -> None:
    """Checagem local, antes de gastar uma chamada: formato e sujeira."""
    from .api import mascarar

    print(f"token: {mascarar(token)}")
    sujeira = []
    if token != token.strip():
        sujeira.append("espaco/quebra de linha nas pontas")
    if token.startswith(("'", '"')) or token.endswith(("'", '"')):
        sujeira.append("aspas em volta do valor")
    estranhos = sorted({c for c in token if not (c.isalnum() or c in "-_")})
    if estranhos:
        sujeira.append(f"caracteres inesperados: {estranhos}")
    if sujeira:
        print("  ATENCAO no .env -> " + "; ".join(sujeira))
    else:
        print("  formato local: OK (sem aspas, sem espaco, so alfanumerico)")


def cmd_testar(args, cfg) -> int:
    from .api import cliente_de_config

    cli = cliente_de_config(cfg)
    if args.token:
        cli.token = args.token.strip()
        print("(usando o token passado por --token, ignorando o .env)")
    _conferir_token(cli.token)

    cli.autenticar()
    print("autenticacao: OK")

    posicoes = cli.posicoes()
    linhas = posicoes.get("l") or []
    veiculos = sum(len(l.get("vs") or []) for l in linhas)
    print(f"/Posicao: {len(linhas)} linhas, {veiculos} veiculos, hora {posicoes.get('hr')}")

    if linhas:
        cl = linhas[0]["cl"]
        prev = cli.previsao_linha(cl)
        pontos = prev.get("ps") or []
        print(f"/Previsao/Linha({cl}): {len(pontos)} paradas")
        if pontos and (pontos[0].get("vs")):
            v = pontos[0]["vs"][0]
            print(f"  exemplo -> parada {pontos[0].get('np')}: veiculo {v.get('p')} as {v.get('t')}")

    if cfg.mongo_uri:
        try:
            from .mongo import MongoStore

            store = MongoStore(cfg)
            store.preparar()
            print(f"MongoDB: OK ({cfg.mongo_db})")
            store.fechar()
        except Exception as exc:
            print(f"MongoDB: FALHOU -> {exc}")
    else:
        print("MongoDB: nao configurado (MONGO_URI vazio)")
    return 0


def cmd_catalogo(args, cfg) -> int:
    from . import catalogo
    from .api import cliente_de_config
    from .storage import Armazenamento

    cli = cliente_de_config(cfg)
    arm = Armazenamento(cfg)
    try:
        letreiros = (
            args.linhas.split(",") if args.linhas else cfg.get("coleta.linhas.letreiros", [])
        )
        resumo = catalogo.sincronizar(
            cfg,
            cli,
            arm,
            letreiros=letreiros,
            todas=args.todas or cfg.get("coleta.linhas.incluir_todas", False),
            varredura=not args.sem_varredura,
            profundo=args.profundo,
        )
        _mostrar("catalogo", resumo)
    finally:
        arm.fechar()
        cli.fechar()
    return 0


def cmd_gtfs(args, cfg) -> int:
    from . import gtfs as mod
    from .storage import Armazenamento

    arm = Armazenamento(cfg, backends=["parquet"])
    try:
        destino = arm.raiz / "gtfs"
        zip_path = mod.baixar(destino, cfg.get("gtfs.url", mod.URL_GTFS), args.forcar)
        feed = mod.GTFS(zip_path)
        relatorio = {"arquivo": str(zip_path), "verificacao": mod.verificar(feed)}

        catalogo = arm.ler_catalogo("linhas")
        if len(catalogo):
            ponte = mod.tabela_ponte(feed, catalogo)
            if len(ponte):
                ponte.to_parquet(arm.raiz / "catalogo" / "ponte_gtfs.parquet", index=False)
                relatorio["ponte"] = {
                    "linhas_casadas": int(len(ponte)),
                    "linhas_no_catalogo": int(len(catalogo)),
                    "taxa": round(len(ponte) / len(catalogo), 3),
                }

            paradas = arm.ler_catalogo("paradas")
            if len(paradas):
                casadas = mod.casar_paradas(
                    paradas,
                    feed.paradas,
                    float(cfg.get("gtfs.raio_casamento_parada_m", 80)),
                )
                if len(casadas):
                    relatorio["paradas"] = {
                        "taxa_casamento": round(
                            float(casadas["stop_id"].notna().mean()), 3
                        ),
                        "por_id": int((casadas["metodo"] == "id").sum()),
                        "por_proximidade": int(
                            (casadas["metodo"] == "proximidade").sum()
                        ),
                    }
        else:
            relatorio["aviso"] = "catalogo vazio: rode `catalogo` para montar a ponte"

        _mostrar("gtfs", relatorio)

        vinculos = arm.ler_catalogo("linha_parada")
        if len(catalogo) and len(vinculos):
            cob = mod.cobertura_de_paradas(feed, catalogo, vinculos)
            if len(cob):
                cob.to_parquet(
                    arm.raiz / "catalogo" / "cobertura_paradas.parquet", index=False
                )
                print(
                    "\n== paradas que a API conhece, por linha ==\n"
                    "So da para medir erro de previsao onde a API preve.\n"
                )
                print(cob.to_string(index=False))
                ruins = cob[cob["cobertura"].notna() & (cob["cobertura"] < 0.6)]
                if len(ruins):
                    print(
                        f"\n{len(ruins)} linha(s) abaixo de 60% de cobertura — "
                        "considere trocar por outras no config/coleta.yaml"
                    )
    finally:
        arm.fechar()
    return 0


def cmd_candidatas(args, cfg) -> int:
    """Ranqueia linhas por quanto elas rendem de dado, antes de coletar."""
    from . import gtfs as mod
    from .api import cliente_de_config
    from .storage import Armazenamento

    cli = cliente_de_config(cfg)
    arm = Armazenamento(cfg, backends=["parquet"])
    try:
        zip_path = arm.raiz / "gtfs" / "sptrans-gtfs.zip"
        if not zip_path.exists():
            print("GTFS ausente. Rode antes: python -m olhovivo gtfs")
            return 2
        feed = mod.GTFS(zip_path)

        cli.autenticar()
        payload = cli.posicoes()
        circulando = [
            {
                "cl": int(l["cl"]),
                "route_id": (l.get("c") or ""),
                "sentido": int(l.get("sl") or 0),
                "veiculos": int(l.get("qv") or 0),
            }
            for l in (payload.get("l") or [])
            if l.get("cl") is not None
        ]
        circulando.sort(key=lambda x: x["veiculos"], reverse=True)
        amostra = circulando[: args.examinar]
        print(
            f"{len(circulando)} linhas circulando agora; "
            f"examinando as {len(amostra)} de maior frota..."
        )

        linhas: list[dict] = []
        for i, item in enumerate(amostra, 1):
            # /Parada/BuscarParadasPorLinha e incompleto (70% das linhas
            # devolvem zero); o universo que interessa e o das paradas COM
            # previsao, entao a conta usa a uniao das duas fontes.
            try:
                cps = {p["cp"] for p in cli.paradas_por_linha(item["cl"]) if p.get("cp")}
            except Exception:
                cps = set()
            n_parada = len(cps)
            try:
                pontos = cli.previsao_linha(item["cl"]).get("ps") or []
                cps |= {p["cp"] for p in pontos if p.get("cp")}
            except Exception:
                pass

            dados = feed.tracado_de(item["route_id"], item["sentido"])
            n_gtfs = len(dados["paradas"]) if dados else 0
            n_api = len(cps)
            linhas.append(
                {
                    **item,
                    "paradas_api": n_api,
                    "so_endpoint_parada": n_parada,
                    "paradas_gtfs": n_gtfs,
                    "cobertura": round(n_api / n_gtfs, 3) if n_gtfs else None,
                    # eventos esperados por ciclo: frota x paradas com previsao
                    "rendimento": item["veiculos"] * n_api,
                }
            )
            if i % 25 == 0:
                print(f"  {i}/{len(amostra)}...")

        import pandas as pd

        df = pd.DataFrame(linhas)
        if df.empty:
            print("nenhuma linha avaliada")
            return 1

        bons = df[
            (df["cobertura"].notna())
            & (df["cobertura"] >= args.min_cobertura)
            & (df["paradas_api"] >= args.min_paradas)
            & (df["veiculos"] >= args.min_veiculos)
        ].sort_values("rendimento", ascending=False)

        df.to_parquet(arm.raiz / "catalogo" / "linhas_candidatas.parquet", index=False)

        print(
            f"\n== linhas que atendem aos criterios "
            f"(cobertura >= {args.min_cobertura}, "
            f">= {args.min_paradas} paradas, >= {args.min_veiculos} veiculos) ==\n"
        )
        if bons.empty:
            print("nenhuma — afrouxe os criterios")
            print("\ndistribuicao da cobertura na amostra:")
            print(df["cobertura"].describe().to_string())
            return 1

        print(
            bons.head(args.limite)[
                ["route_id", "sentido", "veiculos", "paradas_api", "paradas_gtfs",
                 "cobertura", "rendimento"]
            ].to_string(index=False)
        )

        letreiros = []
        for r in bons.itertuples():
            base = str(r.route_id).split("-")[0]
            if base and base not in letreiros:
                letreiros.append(base)
            if len(letreiros) >= args.limite:
                break

        print("\nCole em config/coleta.yaml, em coleta.linhas.letreiros:\n")
        for lt in letreiros:
            print(f'      - "{lt}"')
        return 0
    finally:
        arm.fechar()
        cli.fechar()


def cmd_mapa(args, cfg) -> int:
    """Exporta o itinerario de uma linha em GeoJSON, direto do GTFS (sem token)."""
    import json as _json

    from . import gtfs as mod
    from .storage import Armazenamento

    arm = Armazenamento(cfg, backends=["parquet"])
    try:
        zip_path = arm.raiz / "gtfs" / "sptrans-gtfs.zip"
        if not zip_path.exists():
            print("GTFS ausente. Rode antes: python -m olhovivo gtfs")
            return 2
        feed = mod.GTFS(zip_path)

        if args.letreiro:
            rotas = feed.rotas
            alvos = sorted(
                rotas.loc[
                    rotas["route_id"].astype(str).str.split("-").str[0]
                    == args.letreiro.upper(),
                    "route_id",
                ].astype(str).unique()
            )
            if not alvos:
                print(f"nenhuma rota com letreiro {args.letreiro!r} no feed")
                return 1
        else:
            alvos = [args.linha]

        sentidos = [args.sentido] if args.sentido else [1, 2]
        destino_dir = arm.raiz / "derivado" / "mapas"
        destino_dir.mkdir(parents=True, exist_ok=True)

        gerados = 0
        for route_id in alvos:
            for sentido in sentidos:
                dados = mod.geojson_da_linha(feed, route_id, sentido)
                if dados is None:
                    print(f"  {route_id} sentido {sentido}: sem traçado no feed")
                    continue

                destino = destino_dir / f"{route_id}_s{sentido}.geojson"
                destino.write_text(
                    _json.dumps(dados, ensure_ascii=False), encoding="utf-8"
                )
                gerados += 1

                tracado = dados["features"][0]["properties"]
                paradas = [f["properties"] for f in dados["features"][1:]]
                espacos = [
                    p["espacamento_m"] for p in paradas if p["espacamento_m"] is not None
                ]
                afast = [p["afastamento_do_tracado_m"] for p in paradas]
                print(
                    f"\n{route_id} sentido {sentido} "
                    f"({tracado['comprimento_m']/1000:.1f} km, {len(paradas)} paradas)"
                )
                if espacos:
                    espacos_ord = sorted(espacos)
                    print(
                        f"  espacamento entre paradas: mediana "
                        f"{espacos_ord[len(espacos_ord)//2]:.0f} m, "
                        f"min {min(espacos):.0f} m, max {max(espacos):.0f} m"
                    )
                if afast:
                    longe = sum(1 for a in afast if a > 120)
                    print(
                        f"  afastamento do traçado: mediana "
                        f"{sorted(afast)[len(afast)//2]:.0f} m, "
                        f"{longe} parada(s) acima de 120 m"
                    )
                if args.detalhar:
                    for p in paradas:
                        print(
                            f"    {p['stop_sequence']:>3}. {p['stop_name'][:44]:<44} "
                            f"{p['dist_no_tracado_m']/1000:>6.2f} km"
                        )
                print(f"  -> {destino}")

        if gerados:
            print(
                f"\n{gerados} arquivo(s) em {destino_dir}\n"
                "Abra em geojson.io, QGIS ou kepler.gl para conferir no mapa."
            )
        return 0 if gerados else 1
    finally:
        arm.fechar()


def cmd_coletar(args, cfg) -> int:
    from . import catalogo
    from .api import cliente_de_config
    from .coleta import executar_coleta
    from .storage import Armazenamento

    cli = cliente_de_config(cfg)
    arm = Armazenamento(cfg)
    try:
        if args.linhas:
            monitoradas = catalogo.resolver_letreiros(cli, args.linhas.split(","))
            codigos = sorted(m["cl"] for m in monitoradas)
            meta = {m["cl"]: m for m in monitoradas}
        else:
            codigos = catalogo.codigos_monitorados(cfg, cli, arm)
            cat = arm.ler_catalogo("linhas")
            meta = (
                {
                    int(r.cl): {"letreiro": r.letreiro, "sentido": int(r.sentido)}
                    for r in cat.itertuples()
                }
                if len(cat)
                else {}
            )

        if not codigos:
            print(
                "nenhuma linha monitorada. Preencha coleta.linhas.letreiros no "
                "config/coleta.yaml ou use --linhas 8000,875A"
            )
            return 2

        print(f"linhas monitoradas ({len(codigos)}): {codigos[:20]}{'...' if len(codigos) > 20 else ''}")
        resultado = executar_coleta(
            cfg,
            cli,
            arm,
            codigos,
            catalogo_linhas=meta,
            duracao_horas=args.horas,
            reiniciar=args.reiniciar,
        )
        _mostrar("coleta encerrada", resultado)
    finally:
        arm.fechar()
        cli.fechar()
    return 0


def cmd_status(args, cfg) -> int:
    from .storage import Armazenamento

    arm = Armazenamento(cfg, backends=["parquet"] if args.sem_mongo else None)
    try:
        relatorio: dict = {"raiz": str(arm.raiz)}

        estado = arm.raiz / "estado_coleta.json"
        if estado.exists():
            relatorio["estado"] = json.loads(estado.read_text(encoding="utf-8"))

        for nome in ("posicoes", "previsoes"):
            base = arm.raiz / "bruto" / nome
            manifesto = base / "_manifesto.jsonl"
            if not manifesto.exists():
                relatorio[nome] = "sem dados"
                continue
            registros = [
                json.loads(l)
                for l in manifesto.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            janelas = sorted(
                (r["ts_min"], r["ts_max"]) for r in registros if r.get("ts_min")
            )
            lacunas = []
            for anterior, atual in zip(janelas, janelas[1:]):
                fim = datetime.fromisoformat(anterior[1])
                inicio = datetime.fromisoformat(atual[0])
                minutos = (inicio - fim).total_seconds() / 60
                if minutos > 10:
                    lacunas.append(
                        {"de": anterior[1], "ate": atual[0], "minutos": round(minutos, 1)}
                    )
            relatorio[nome] = {
                "arquivos": len(registros),
                "linhas": sum(r.get("linhas", 0) for r in registros),
                "mb": round(
                    sum(f.stat().st_size for f in base.rglob("*.parquet")) / 1e6, 1
                ),
                "inicio": janelas[0][0] if janelas else None,
                "fim": janelas[-1][1] if janelas else None,
                "lacunas_acima_10min": lacunas[-10:],
                "total_lacunas": len(lacunas),
            }

        for nome in ("chegadas", "previsao_realizado"):
            df = arm.ler_derivado(nome)
            relatorio[nome] = {"linhas": int(len(df))} if len(df) else "ainda nao gerado"

        if arm.mongo:
            relatorio["mongo"] = arm.mongo.cobertura_coleta()

        _mostrar("status", relatorio)
    finally:
        arm.fechar()
    return 0


def cmd_chegadas(args, cfg) -> int:
    from . import chegadas
    from .storage import Armazenamento

    arm = Armazenamento(cfg)
    try:
        resumo = chegadas.processar(
            cfg,
            arm,
            inicio=_instante(args.inicio),
            fim=_instante(args.fim),
            codigos=[int(c) for c in args.cl.split(",")] if args.cl else None,
            conf_minima=args.conf_min,
        )
        _mostrar("chegadas detectadas", resumo)
    finally:
        arm.fechar()
    return 0


def cmd_atraso(args, cfg) -> int:
    from . import atraso
    from .storage import Armazenamento

    arm = Armazenamento(cfg)
    try:
        resumo = atraso.construir(cfg, arm, min_confianca=args.conf_min)
        _mostrar("atraso por trecho", resumo)

        fluxos = atraso.por_regiao(arm, min_amostras=args.min_amostras)
        if len(fluxos):
            print("\n== fluxos (origem -> destino) com maior taxa de atraso ==")
            print(fluxos.head(20).to_string(index=False))
    finally:
        arm.fechar()
    return 0


def cmd_casar(args, cfg) -> int:
    from . import casamento
    from .storage import Armazenamento

    arm = Armazenamento(cfg)
    try:
        resumo = casamento.executar(
            cfg,
            arm,
            inicio=_instante(args.inicio),
            fim=_instante(args.fim),
            conf_minima=args.conf_min,
        )
        _mostrar("previsao x realizado", resumo)

        ranking = casamento.resumo_por_linha(arm)
        if len(ranking):
            print("\n== linhas com maior probabilidade de atraso (>= 5 min) ==")
            print(ranking.head(20).to_string(index=False))
    finally:
        arm.fechar()
    return 0


def cmd_regioes(args, cfg) -> int:
    from . import regioes
    from .storage import Armazenamento

    arm = Armazenamento(cfg)
    try:
        resultado = regioes.executar(
            cfg,
            arm,
            algoritmos=args.algoritmos.split(","),
            modo=args.modo,
            limiar_atraso_s=args.limiar,
        )
        _mostrar("regioes", resultado)
        print(f"\nGeoJSON: {arm.raiz / 'derivado' / 'regioes.geojson'}")
    finally:
        arm.fechar()
    return 0


def cmd_features(args, cfg) -> int:
    from . import features
    from .storage import Armazenamento

    arm = Armazenamento(cfg, backends=["parquet"])
    try:
        saida = {}
        if args.tudo or args.tabela:
            df = features.montar_tabela(arm)
            saida["tabela"] = {"linhas": int(len(df)), "colunas": int(df.shape[1])}
        if args.tudo or args.sequencias:
            saida["sequencias"] = features.montar_sequencias(arm)
        if args.tudo or args.grafo:
            saida["grafo"] = features.montar_grafo(arm)
        if not saida:
            print("escolha o que gerar: --tudo, --tabela, --sequencias ou --grafo")
            return 2
        _mostrar("atributos", saida)
    finally:
        arm.fechar()
    return 0


def cmd_treinar(args, cfg) -> int:
    import pandas as pd

    from . import modelos
    from .storage import Armazenamento

    arm = Armazenamento(cfg, backends=["parquet"])
    try:
        derivado = arm.raiz / "derivado"
        if args.modelo == "xgboost" and args.alvo == "atraso":
            caminho = derivado / "percursos.parquet"
            if not caminho.exists():
                print("rode antes: python -m olhovivo atraso")
                return 2
            saida = modelos.treinar_atraso(
                pd.read_parquet(caminho),
                float(cfg.get("atraso.limiar_percentual", 0.20)),
            )["resultado"]
        elif args.modelo == "xgboost":
            caminho = derivado / "modelagem.parquet"
            if not caminho.exists():
                print("rode antes: python -m olhovivo features --tabela")
                return 2
            saida = modelos.treinar_xgboost(pd.read_parquet(caminho))["resultado"]
        elif args.modelo == "lstm":
            saida = modelos.treinar_lstm(derivado / "sequencias_lstm.npz")["resultado"]
        elif args.modelo == "gnn":
            saida = modelos.treinar_gnn(derivado / "grafo.npz")["resultado"]
        else:
            print(f"modelo desconhecido: {args.modelo}")
            return 2
        _mostrar(f"modelo {args.modelo}", saida)
    finally:
        arm.fechar()
    return 0


def cmd_mongo_init(args, cfg) -> int:
    from .mongo import MongoStore

    store = MongoStore(cfg)
    store.preparar()
    _mostrar("mongo", store.cobertura_coleta())
    store.fechar()
    return 0


# ------------------------------------------------------------------- parser
def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m olhovivo",
        description="Coleta e analise de atraso de onibus de Sao Paulo (API Olho Vivo)",
    )
    p.add_argument("--config", help="caminho do YAML de configuracao")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"olhovivo {__version__}")
    sub = p.add_subparsers(dest="comando", required=True)

    s = sub.add_parser("testar", help="valida token, API e MongoDB")
    s.add_argument("--token", help="testa este token em vez do que esta no .env")
    s.set_defaults(func=cmd_testar)

    s = sub.add_parser("catalogo", help="mapeia linhas, paradas, corredores e empresas")
    s.add_argument("--linhas", help="letreiros separados por virgula (ex: 8000,875A)")
    s.add_argument("--todas", action="store_true", help="baixa paradas de TODAS as linhas")
    s.add_argument("--sem-varredura", action="store_true", help="so usa o snapshot /Posicao")
    s.add_argument("--profundo", action="store_true", help="varredura com prefixos de 2 digitos")
    s.set_defaults(func=cmd_catalogo)

    s = sub.add_parser(
        "gtfs", help="baixa o GTFS estatico da SPTrans (tracado e ordem das paradas)"
    )
    s.add_argument("--forcar", action="store_true", help="rebaixa mesmo se ja existir")
    s.set_defaults(func=cmd_gtfs)

    s = sub.add_parser(
        "linhas-candidatas",
        help="ranqueia linhas por cobertura de previsao e frota, antes de coletar",
    )
    s.add_argument("--examinar", type=int, default=120, help="quantas linhas sondar")
    s.add_argument("--min-cobertura", type=float, default=0.6)
    s.add_argument("--min-paradas", type=int, default=12)
    s.add_argument("--min-veiculos", type=int, default=6)
    s.add_argument("--limite", type=int, default=15)
    s.set_defaults(func=cmd_candidatas)

    s = sub.add_parser(
        "mapa", help="exporta o itinerario de uma linha em GeoJSON (so GTFS, sem token)"
    )
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--linha", help="route_id completo, ex: 8000-10")
    g.add_argument("--letreiro", help="so o letreiro, ex: 8000 (pega todas as variantes)")
    s.add_argument("--sentido", type=int, choices=[1, 2], help="padrao: os dois")
    s.add_argument("--detalhar", action="store_true", help="lista as paradas na ordem")
    s.set_defaults(func=cmd_mapa)

    s = sub.add_parser("coletar", help="coleta continua (padrao: 7 dias)")
    s.add_argument("--horas", type=float, help="duracao em horas (sobrepoe o config)")
    s.add_argument("--linhas", help="letreiros separados por virgula")
    s.add_argument("--reiniciar", action="store_true", help="comeca uma nova janela de coleta")
    s.set_defaults(func=cmd_coletar)

    s = sub.add_parser("status", help="volume coletado, janela e lacunas")
    s.add_argument("--sem-mongo", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("chegadas", help="detecta a chegada real a partir das trajetorias")
    s.add_argument("--inicio")
    s.add_argument("--fim")
    s.add_argument("--cl", help="codigos de linha separados por virgula")
    s.add_argument("--conf-min", type=float, default=0.0)
    s.set_defaults(func=cmd_chegadas)

    s = sub.add_parser(
        "atraso", help="constroi a variavel de atraso por trecho (definicao do estudo)"
    )
    s.add_argument("--conf-min", type=float, default=0.4)
    s.add_argument("--min-amostras", type=int, default=30, help="minimo por fluxo")
    s.set_defaults(func=cmd_atraso)

    s = sub.add_parser("casar", help="cruza previsao com chegada real")
    s.add_argument("--inicio")
    s.add_argument("--fim")
    s.add_argument("--conf-min", type=float, default=0.35)
    s.set_defaults(func=cmd_casar)

    s = sub.add_parser("regioes", help="agrupa as regioes mais impactadas")
    s.add_argument(
        "--algoritmos", default="kmeans,dbscan,hdbscan,st-dbscan"
    )
    s.add_argument("--modo", choices=["eventos", "agregado"], default="eventos")
    s.add_argument("--limiar", type=int, default=300, help="atraso minimo, em segundos")
    s.set_defaults(func=cmd_regioes)

    s = sub.add_parser("features", help="gera os atributos para os modelos")
    s.add_argument("--tudo", action="store_true")
    s.add_argument("--tabela", action="store_true")
    s.add_argument("--sequencias", action="store_true")
    s.add_argument("--grafo", action="store_true")
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("treinar", help="treina um modelo preditivo")
    s.add_argument("modelo", choices=["xgboost", "lstm", "gnn"])
    s.add_argument(
        "--alvo", choices=["atraso", "previsao"], default="atraso",
        help="atraso = desvio de tempo de percurso (definicao do estudo); "
             "previsao = erro da previsao publicada pela API",
    )
    s.set_defaults(func=cmd_treinar)

    s = sub.add_parser("mongo-init", help="cria colecoes e indices geoespaciais")
    s.set_defaults(func=cmd_mongo_init)

    return p


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    _log(args.verbose)
    cfg = config.carregar(args.config)
    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:
        print("\ninterrompido")
        return 130
    except RuntimeError as exc:
        texto = str(exc)
        # mensagens multilinha (ex.: login recusado) ja vem formatadas
        print(("\n" + texto) if "\n" in texto else f"\nerro: {texto}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
