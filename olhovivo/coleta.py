"""
Coleta continua de 1 semana.

Dois laços independentes, em threads separadas:

* **Posicoes** — `GET /Posicao` (uma chamada, todos os veiculos da cidade) ou
  `GET /Posicao/Linha` por linha monitorada. Reconstroi a trajetoria de cada
  veiculo, que e o que permite descobrir a hora em que ele DE FATO chegou.

* **Previsoes** — `GET /Previsao/Linha` por linha monitorada. Uma chamada
  devolve, de uma vez, a previsao de chegada de todos os veiculos daquela
  linha em todas as paradas dela. E a "promessa" da API, que depois vai ser
  confrontada com a chegada real.

Detalhes que importam
---------------------
* O campo `ta` do veiculo so muda a cada ~20-60 s (e o relogio do GPS embarcado,
  nao o da consulta). Fazer poll mais rapido que isso gera duplicata pura, entao
  a dedup acontece na ingestao pela chave (prefixo, linha, ta).
* O campo `t` da previsao e "HH:MM" sem data e **sem fuso declarado**. A
  resolucao para instante absoluto acontece aqui, na coleta, onde ainda se sabe
  o instante da consulta — e o fuso e medido antes de comecar, por
  `calibrar_fuso_previsao`, em vez de assumido. Perto da meia-noite "00:05"
  significa amanha.
* A coleta e retomavel: o instante de inicio da semana fica em
  `dados/estado_coleta.json`. Se o processo morrer e voltar, ele continua a
  MESMA semana em vez de comecar outra, e a lacuna fica registrada no log de
  eventos.
* So um coletor por vez (`dados/coleta.lock`): duas sessoes com o mesmo token
  podem derrubar a autenticacao uma da outra na API da SPTrans.
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from zoneinfo import ZoneInfo

    TZ_SP = ZoneInfo("America/Sao_Paulo")
except Exception:  # pragma: no cover - Windows sem tzdata
    TZ_SP = timezone(timedelta(hours=-3))

from .api import ClienteOlhoVivo
from .storage import Armazenamento, agora_utc

log = logging.getLogger("olhovivo.coleta")


# ------------------------------------------------------------------ parsing
def parse_ta(valor: Any) -> datetime | None:
    """Converte o timestamp UTC do GPS ('2026-08-20T14:30:37Z')."""
    if not valor:
        return None
    if isinstance(valor, datetime):
        return valor if valor.tzinfo else valor.replace(tzinfo=timezone.utc)
    texto = str(valor).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(texto)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def resolver_horario(
    t_str: str | None, ts_coleta: datetime, fuso: str = "local"
) -> datetime | None:
    """
    'HH:MM' (sem data) -> instante absoluto em UTC.

    Ancora na data do momento da consulta, no fuso indicado. Se o horario
    resultante ficar muito no passado, e previsao para depois da meia-noite:
    soma um dia.

    `fuso` = "local" (America/Sao_Paulo) ou "utc". A documentacao da SPTrans
    NAO declara o fuso do campo `t`, e os exemplos oficiais sao ambiguos — nos
    tres exemplos publicados o `t` bate com `ta` (UTC), nao com `hr` (local).
    Por isso a escolha nao e chutada: `calibrar_fuso_previsao` mede em dado
    real qual interpretacao produz horizontes plausiveis.
    """
    if not t_str:
        return None
    partes = str(t_str).strip().split(":")
    if len(partes) < 2:
        return None
    try:
        hora, minuto = int(partes[0]), int(partes[1])
        segundo = int(partes[2]) if len(partes) > 2 else 0
    except ValueError:
        return None
    if not (0 <= hora <= 23 and 0 <= minuto <= 59):
        return None

    tz = timezone.utc if fuso == "utc" else TZ_SP
    ancora = ts_coleta.astimezone(tz)
    alvo = ancora.replace(hour=hora, minute=minuto, second=segundo, microsecond=0)
    # 90 min de folga para tras: como o horizonte util nunca passa de ~60 min,
    # qualquer coisa mais antiga que isso e virada de meia-noite, nao atraso
    if alvo < ancora - timedelta(minutes=90):
        alvo += timedelta(days=1)
    elif alvo > ancora + timedelta(hours=12):
        alvo -= timedelta(days=1)
    return alvo.astimezone(timezone.utc)


def calibrar_fuso_previsao(
    cli: ClienteOlhoVivo, codigos: Sequence[int], max_linhas: int = 4
) -> tuple[str, dict]:
    """
    Descobre EMPIRICAMENTE o fuso do campo `t` das previsoes.

    Errar isso desloca a analise inteira em 3 horas. Em vez de assumir, faz
    algumas chamadas reais e testa as duas interpretacoes: fica com aquela em
    que a maior parte dos horizontes (previsto menos instante da consulta) cai
    na faixa plausivel de -2 a 60 minutos. Previsao de chegada nunca aponta
    para 3 horas no futuro nem para o passado distante.
    """
    amostras: list[tuple[str, datetime]] = []
    for cl in list(codigos)[:max_linhas]:
        ts = agora_utc()
        try:
            payload = cli.previsao_linha(int(cl))
        except Exception as exc:
            log.warning("calibracao: linha %s falhou (%s)", cl, exc)
            continue
        for ponto in payload.get("ps") or []:
            for veiculo in ponto.get("vs") or []:
                if veiculo.get("t"):
                    amostras.append((str(veiculo["t"]), ts))
        if len(amostras) >= 60:
            break

    diagnostico: dict[str, Any] = {"amostras": len(amostras)}
    if not amostras:
        log.warning("calibracao de fuso sem amostras; assumindo horario local")
        return "local", diagnostico | {"decisao": "local", "motivo": "sem amostras"}

    for candidato in ("local", "utc"):
        horizontes = []
        for t_str, ts in amostras:
            t = resolver_horario(t_str, ts, candidato)
            if t:
                horizontes.append((t - ts).total_seconds())
        plausiveis = [h for h in horizontes if -120 <= h <= 3600]
        diagnostico[candidato] = {
            "fracao_plausivel": round(len(plausiveis) / max(len(horizontes), 1), 3),
            "horizonte_mediano_s": (
                round(sorted(horizontes)[len(horizontes) // 2]) if horizontes else None
            ),
        }

    escolhido = max(
        ("local", "utc"), key=lambda c: diagnostico[c]["fracao_plausivel"]
    )
    diagnostico["decisao"] = escolhido
    log.info(
        "fuso do campo `t` calibrado como %s (local=%.0f%% plausivel, utc=%.0f%%)",
        escolhido.upper(),
        diagnostico["local"]["fracao_plausivel"] * 100,
        diagnostico["utc"]["fracao_plausivel"] * 100,
    )
    return escolhido, diagnostico


def _hora_config(texto: str, padrao: dtime) -> dtime:
    try:
        h, m = str(texto).split(":")[:2]
        return dtime(int(h), int(m))
    except Exception:
        return padrao


def dentro_da_janela(agora: datetime, inicio: dtime, fim: dtime) -> bool:
    """Janela diaria em horario de Sao Paulo, com suporte a virada de dia."""
    hhmm = agora.astimezone(TZ_SP).time()
    if inicio == fim:
        return True
    if inicio < fim:
        return inicio <= hhmm < fim
    return hhmm >= inicio or hhmm < fim


# ------------------------------------------------------------ achatamento
def achatar_posicoes(
    payload: dict,
    ts_coleta: datetime,
    cl_padrao: int | None = None,
    letreiro_padrao: str | None = None,
    sentido_padrao: int | None = None,
) -> list[dict]:
    """
    Payload de /Posicao ou /Posicao/Linha -> linhas tabulares.

    As duas respostas tem formatos DIFERENTES: /Posicao agrupa os veiculos por
    linha no array `l`, enquanto /Posicao/Linha ja vem filtrado e devolve `vs`
    na RAIZ, sem nenhuma informacao de linha — por isso os parametros
    `*_padrao`, que preenchem o que a resposta nao traz.
    """
    hr = payload.get("hr")
    blocos = payload.get("l")
    if not blocos and payload.get("vs") is not None:
        blocos = [
            {
                "cl": cl_padrao,
                "c": letreiro_padrao,
                "sl": sentido_padrao,
                "lt0": None,
                "vs": payload.get("vs"),
            }
        ]

    saida: list[dict] = []
    for linha in blocos or []:
        cl = linha.get("cl", cl_padrao)
        if cl is None:
            continue
        letreiro_completo = linha.get("c") or letreiro_padrao or ""
        sentido = int(linha.get("sl") or sentido_padrao or 0)
        # cuidado: na API, lt0 e o DESTINO e lt1 e a ORIGEM (contraintuitivo)
        destino = linha.get("lt0")
        for veiculo in linha.get("vs") or []:
            lat, lon = veiculo.get("py"), veiculo.get("px")
            if lat is None or lon is None:
                continue
            saida.append(
                {
                    "ts_coleta": ts_coleta,
                    "hr_api": hr,
                    "cl": int(cl),
                    "letreiro": letreiro_completo.split("-")[0] or None,
                    "sentido": sentido,
                    "destino": destino,
                    "prefixo": str(veiculo.get("p")) if veiculo.get("p") else None,
                    "acessivel": bool(veiculo.get("a")) if veiculo.get("a") is not None else None,
                    "ta": parse_ta(veiculo.get("ta")),
                    "lat": float(lat),
                    "lon": float(lon),
                }
            )
    return saida


def achatar_previsao_linha(
    payload: dict,
    cl: int,
    ts_coleta: datetime,
    letreiro: str | None,
    sentido: int | None,
    fuso: str = "local",
) -> list[dict]:
    """Payload de /Previsao/Linha -> uma linha por (parada, veiculo)."""
    hr = payload.get("hr")
    pontos = payload.get("ps")
    if pontos is None:
        ponto = payload.get("p")
        pontos = [ponto] if ponto else []

    saida: list[dict] = []
    for ponto in pontos or []:
        if not ponto:
            continue
        cp = ponto.get("cp")
        if cp is None:
            continue
        p_lat, p_lon = ponto.get("py"), ponto.get("px")

        # /Previsao/Linha traz `vs` direto no ponto; /Previsao e /Previsao/Parada
        # embrulham em `l` (uma entrada por linha que atende o ponto)
        blocos = []
        if ponto.get("vs") is not None:
            blocos.append((cl, sentido, ponto.get("vs")))
        for sub in ponto.get("l") or []:
            blocos.append(
                (
                    int(sub.get("cl") or cl),
                    int(sub.get("sl") or (sentido or 0)),
                    sub.get("vs"),
                )
            )

        for cl_bloco, sentido_bloco, veiculos in blocos:
            for veiculo in veiculos or []:
                t_str = veiculo.get("t")
                t_prev = resolver_horario(t_str, ts_coleta, fuso)
                horizonte = (
                    int((t_prev - ts_coleta).total_seconds()) if t_prev else None
                )
                saida.append(
                    {
                        "ts_coleta": ts_coleta,
                        "hr_api": hr,
                        "cl": int(cl_bloco),
                        "letreiro": letreiro,
                        "sentido": int(sentido_bloco or 0),
                        "cp": int(cp),
                        "parada_nome": ponto.get("np"),
                        "parada_lat": float(p_lat) if p_lat is not None else None,
                        "parada_lon": float(p_lon) if p_lon is not None else None,
                        "prefixo": str(veiculo.get("p")) if veiculo.get("p") else None,
                        "acessivel": bool(veiculo.get("a"))
                        if veiculo.get("a") is not None
                        else None,
                        "t_previsto_str": t_str,
                        "t_previsto": t_prev,
                        "horizonte_s": horizonte,
                        "ta": parse_ta(veiculo.get("ta")),
                        "lat": float(veiculo["py"]) if veiculo.get("py") is not None else None,
                        "lon": float(veiculo["px"]) if veiculo.get("px") is not None else None,
                    }
                )
    return saida


# --------------------------------------------------------------------- trava
class TravaColeta:
    """
    Impede dois coletores rodando ao mesmo tempo.

    Nao e frescura: duas sessoes autenticadas com o MESMO token podem derrubar
    a sessao uma da outra na API da SPTrans, e o resultado seria uma semana de
    coleta cheia de buracos por 401. Alem disso, dois processos escrevendo no
    mesmo diretorio duplicariam tudo.
    """

    def __init__(self, caminho: Path, validade_s: float = 600.0):
        self.caminho = Path(caminho)
        self.validade_s = validade_s
        self.adquirida = False

    def adquirir(self) -> None:
        import os

        if self.caminho.exists():
            idade = time.time() - self.caminho.stat().st_mtime
            conteudo = self.caminho.read_text(encoding="utf-8", errors="replace").strip()
            if idade < self.validade_s:
                raise RuntimeError(
                    f"ja existe uma coleta em andamento ({conteudo}). "
                    f"Se tiver certeza que nao, apague {self.caminho}"
                )
            log.warning("trava velha (%.0f min) encontrada; assumindo", idade / 60)
            self.caminho.unlink(missing_ok=True)

        self.caminho.write_text(
            f"pid={os.getpid()} inicio={agora_utc().isoformat()}", encoding="utf-8"
        )
        self.adquirida = True

    def renovar(self) -> None:
        if self.adquirida and self.caminho.exists():
            self.caminho.touch()

    def liberar(self) -> None:
        if self.adquirida:
            self.caminho.unlink(missing_ok=True)
            self.adquirida = False


# -------------------------------------------------------------------- estado
class EstadoColeta:
    """Guarda o inicio da semana para a coleta ser retomavel."""

    def __init__(self, caminho: Path):
        self.caminho = caminho
        self.dados: dict[str, Any] = {}
        if caminho.exists():
            try:
                self.dados = json.loads(caminho.read_text(encoding="utf-8"))
            except Exception:
                self.dados = {}

    @property
    def inicio(self) -> datetime | None:
        bruto = self.dados.get("inicio")
        return parse_ta(bruto) if bruto else None

    def iniciar(self, quando: datetime, reiniciar: bool = False) -> datetime:
        if reiniciar or not self.inicio:
            self.dados["inicio"] = quando.isoformat()
            self.dados["execucoes"] = self.dados.get("execucoes", 0) + 1
            self.salvar()
            return quando
        self.dados["execucoes"] = self.dados.get("execucoes", 0) + 1
        self.dados["ultima_retomada"] = quando.isoformat()
        self.salvar()
        return self.inicio

    def marcar_fim(self, quando: datetime) -> None:
        self.dados["ultimo_encerramento"] = quando.isoformat()
        self.salvar()

    def salvar(self) -> None:
        self.caminho.write_text(
            json.dumps(self.dados, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# ------------------------------------------------------------------- coletor
class Coletor:
    def __init__(
        self,
        cfg,
        cli: ClienteOlhoVivo,
        arm: Armazenamento,
        codigos: Sequence[int],
        catalogo_linhas: dict[int, dict] | None = None,
        fuso_previsao: str = "local",
    ):
        self.cfg = cfg
        self.cli = cli
        self.arm = arm
        self.codigos = list(codigos)
        self.catalogo = catalogo_linhas or {}
        self.fuso_previsao = fuso_previsao
        self.trava: "TravaColeta | None" = None

        self.modo_posicoes = cfg.get("coleta.posicoes.modo", "todas")
        self.intervalo_pos = float(cfg.get("coleta.posicoes.intervalo_s", 30))
        self.intervalo_prev = float(cfg.get("coleta.previsoes.intervalo_s", 60))
        self.workers = int(cfg.get("coleta.previsoes.workers", 4))
        self.janela_inicio = _hora_config(cfg.get("coleta.janela_inicio", "00:00"), dtime(0, 0))
        self.janela_fim = _hora_config(cfg.get("coleta.janela_fim", "23:59"), dtime(23, 59))

        self.parar = threading.Event()
        self._vistos: dict[str, datetime] = {}
        self._trava_vistos = threading.Lock()
        self._ultima_limpeza = time.monotonic()
        self._proximo_indice = 0

        self.contadores = {
            "ciclos_posicao": 0,
            "ciclos_previsao": 0,
            "posicoes_gravadas": 0,
            "posicoes_duplicadas": 0,
            "previsoes_gravadas": 0,
            "falhas": 0,
        }

    # -- dedup ---------------------------------------------------------------
    def _filtrar_novas(self, linhas: list[dict]) -> list[dict]:
        novas: list[dict] = []
        with self._trava_vistos:
            for r in linhas:
                if not r.get("prefixo"):
                    novas.append(r)
                    continue
                chave = f"{r['prefixo']}|{r['cl']}"
                ta = r.get("ta")
                if ta is None:
                    novas.append(r)
                    continue
                if self._vistos.get(chave) == ta:
                    self.contadores["posicoes_duplicadas"] += 1
                    continue
                self._vistos[chave] = ta
                novas.append(r)

            # limpeza periodica para o dicionario nao crescer sem fim
            if time.monotonic() - self._ultima_limpeza > 1800:
                corte = agora_utc() - timedelta(hours=2)
                self._vistos = {k: v for k, v in self._vistos.items() if v > corte}
                self._ultima_limpeza = time.monotonic()
        return novas

    # -- laco de posicoes ----------------------------------------------------
    def _laco_posicoes(self) -> None:
        while not self.parar.is_set():
            inicio = time.monotonic()
            try:
                if dentro_da_janela(agora_utc(), self.janela_inicio, self.janela_fim):
                    self._ciclo_posicoes()
            except Exception as exc:
                self.contadores["falhas"] += 1
                log.exception("ciclo de posicoes falhou: %s", exc)
                self.arm.eventos.registrar("falha_posicoes", erro=str(exc))

            gasto = time.monotonic() - inicio
            if gasto > self.intervalo_pos * 1.5:
                self.arm.eventos.registrar(
                    "ciclo_lento", laco="posicoes", segundos=round(gasto, 1)
                )
            self.parar.wait(max(0.0, self.intervalo_pos - gasto))

    def _ciclo_posicoes(self) -> None:
        ts = agora_utc()
        if self.modo_posicoes == "todas":
            linhas = achatar_posicoes(self.cli.posicoes(), ts)
        else:
            linhas = []
            for cl in self.codigos:
                meta = self.catalogo.get(cl, {})
                try:
                    linhas.extend(
                        achatar_posicoes(
                            self.cli.posicoes_linha(cl),
                            ts,
                            cl_padrao=cl,
                            letreiro_padrao=meta.get("letreiro"),
                            sentido_padrao=meta.get("sentido"),
                        )
                    )
                except Exception as exc:
                    log.warning("posicoes da linha %s falharam: %s", cl, exc)

        novas = self._filtrar_novas(linhas)
        self.arm.escrever_posicoes(novas)
        self.contadores["ciclos_posicao"] += 1
        self.contadores["posicoes_gravadas"] += len(novas)

    # -- laco de previsoes ---------------------------------------------------
    def _laco_previsoes(self) -> None:
        if not self.codigos:
            log.warning("nenhuma linha monitorada: laco de previsoes desligado")
            return

        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="prev") as pool:
            while not self.parar.is_set():
                inicio = time.monotonic()
                try:
                    if dentro_da_janela(agora_utc(), self.janela_inicio, self.janela_fim):
                        self._ciclo_previsoes(pool, inicio)
                except Exception as exc:
                    self.contadores["falhas"] += 1
                    log.exception("ciclo de previsoes falhou: %s", exc)
                    self.arm.eventos.registrar("falha_previsoes", erro=str(exc))

                gasto = time.monotonic() - inicio
                self.parar.wait(max(0.0, self.intervalo_prev - gasto))

    def _ciclo_previsoes(self, pool: ThreadPoolExecutor, inicio_mono: float) -> None:
        prazo = inicio_mono + self.intervalo_prev * 0.95
        total = len(self.codigos)
        ordem = [
            self.codigos[(self._proximo_indice + i) % total] for i in range(total)
        ]

        processados = 0
        futuros = {pool.submit(self._buscar_previsao, cl): cl for cl in ordem}
        for futuro, cl in futuros.items():
            # estourou o ciclo: cancela o que ainda nao comecou (nao gasta
            # requisicao) e retoma daqui no proximo ciclo
            if time.monotonic() > prazo and futuro.cancel():
                continue
            try:
                linhas = futuro.result()
            except Exception as exc:
                self.contadores["falhas"] += 1
                log.warning("previsao da linha %s falhou: %s", cl, exc)
                processados += 1
                continue
            processados += 1
            if linhas:
                self.arm.escrever_previsoes(linhas)
                self.contadores["previsoes_gravadas"] += len(linhas)

        self._proximo_indice = (self._proximo_indice + max(processados, 1)) % total
        self.contadores["ciclos_previsao"] += 1

        if processados < total:
            self.arm.eventos.registrar(
                "ciclo_previsao_incompleto",
                processadas=processados,
                total=total,
                dica="aumente coleta.previsoes.intervalo_s ou reduza o numero de linhas",
            )

    def _buscar_previsao(self, cl: int) -> list[dict]:
        ts = agora_utc()
        meta = self.catalogo.get(cl, {})
        payload = self.cli.previsao_linha(cl)
        if not payload:
            return []
        return achatar_previsao_linha(
            payload,
            cl,
            ts,
            meta.get("letreiro"),
            meta.get("sentido"),
            self.fuso_previsao,
        )

    # -- batimento cardiaco --------------------------------------------------
    def _laco_heartbeat(self) -> None:
        while not self.parar.wait(60.0):
            self.arm.flush()
            if self.trava is not None:
                self.trava.renovar()
            estado = {**self.contadores, **self.cli.stats}
            log.info(
                "posicoes=%(posicoes_gravadas)d (dup %(posicoes_duplicadas)d) "
                "previsoes=%(previsoes_gravadas)d ciclos=%(ciclos_posicao)d/"
                "%(ciclos_previsao)d falhas=%(falhas)d req=%(requisicoes)d",
                estado,
            )
            self.arm.eventos.registrar("heartbeat", **estado)

    # -- execucao ------------------------------------------------------------
    def executar(self, duracao_s: float, ate: datetime | None = None) -> dict:
        threads = [
            threading.Thread(target=self._laco_posicoes, name="posicoes", daemon=True),
            threading.Thread(target=self._laco_previsoes, name="previsoes", daemon=True),
            threading.Thread(target=self._laco_heartbeat, name="heartbeat", daemon=True),
        ]
        for t in threads:
            t.start()

        limite = ate or (agora_utc() + timedelta(seconds=duracao_s))
        log.info(
            "coleta em andamento ate %s (%.1f h), %d linhas monitoradas",
            limite.astimezone(TZ_SP).strftime("%d/%m %H:%M"),
            max(0.0, (limite - agora_utc()).total_seconds()) / 3600,
            len(self.codigos),
        )

        try:
            while not self.parar.is_set() and agora_utc() < limite:
                self.parar.wait(5.0)
        finally:
            self.parar.set()
            for t in threads:
                t.join(timeout=20)
            self.arm.flush()

        return dict(self.contadores)

    def solicitar_parada(self, *_args) -> None:
        log.warning("parada solicitada; descarregando buffers...")
        self.parar.set()


# ----------------------------------------------------------------- orquestra
def executar_coleta(
    cfg,
    cli: ClienteOlhoVivo,
    arm: Armazenamento,
    codigos: Sequence[int],
    catalogo_linhas: dict[int, dict] | None = None,
    duracao_horas: float | None = None,
    reiniciar: bool = False,
) -> dict:
    trava = TravaColeta(cfg.dir_dados / "coleta.lock")
    trava.adquirir()

    estado = EstadoColeta(cfg.dir_dados / "estado_coleta.json")
    agora = agora_utc()
    inicio = estado.iniciar(agora, reiniciar=reiniciar)

    duracao_s = (
        duracao_horas * 3600
        if duracao_horas
        else float(cfg.get("coleta.duracao_dias", 7)) * 86400
    )
    limite = inicio + timedelta(seconds=duracao_s)

    if inicio != agora:
        log.info(
            "retomando a semana iniciada em %s; restam %.1f h",
            inicio.astimezone(TZ_SP).strftime("%d/%m %H:%M"),
            max(0.0, (limite - agora).total_seconds()) / 3600,
        )
        arm.eventos.registrar("coleta_retomada", inicio=inicio.isoformat())

    if limite <= agora:
        trava.liberar()
        log.warning(
            "a janela de %.1f dias iniciada em %s ja terminou. "
            "Use --reiniciar para comecar uma nova coleta.",
            duracao_s / 86400,
            inicio.isoformat(),
        )
        return {}

    # o fuso do campo `t` nao e documentado: mede antes de comecar a gravar
    fuso, diagnostico = calibrar_fuso_previsao(cli, codigos)
    estado.dados["fuso_previsao"] = fuso
    estado.dados["calibracao_fuso"] = diagnostico
    estado.salvar()
    arm.eventos.registrar("fuso_calibrado", fuso=fuso, **diagnostico)

    coletor = Coletor(cfg, cli, arm, codigos, catalogo_linhas, fuso_previsao=fuso)
    coletor.trava = trava
    for sinal in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sinal, coletor.solicitar_parada)
        except (ValueError, AttributeError):
            pass

    arm.eventos.registrar(
        "coleta_iniciada",
        linhas_monitoradas=len(codigos),
        modo_posicoes=coletor.modo_posicoes,
        limite=limite.isoformat(),
    )
    try:
        resultado = coletor.executar(duracao_s, ate=limite)
    finally:
        arm.flush()
        estado.marcar_fim(agora_utc())
        arm.eventos.registrar("coleta_encerrada", **coletor.contadores)
        trava.liberar()

    return resultado
