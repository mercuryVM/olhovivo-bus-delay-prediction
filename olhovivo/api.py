"""
Cliente HTTP da API Olho Vivo (SPTrans), versao 2.1.

Autenticacao
------------
A API troca um token pessoal por um cookie de sessao:

    POST {base}/Login/Autenticar?token=SEU_TOKEN   -> corpo "true"/"false"
                                                   -> Set-Cookie: apiCredentials

Todas as chamadas seguintes precisam desse cookie. A sessao expira sozinha (o
prazo nao e documentado); este cliente detecta o 401 e reautentica sozinho.

Dicionario de campos (nomes abreviados devolvidos pela API)
-----------------------------------------------------------
Linha        cl   int   codigo interno da linha **por sentido**
             lc   bool  circular (sem terminal secundario)
             lt   str   letreiro numerico ("8000")
             tl   int   letreiro complementar / digito identificador
             sl   int   sentido: 1 = terminal principal -> secundario
                                 2 = terminal secundario -> principal
             tp   str   terminal principal (destino no sentido 1)
             ts   str   terminal secundario (destino no sentido 2)

Parada       cp   int   codigo da parada
             np   str   nome da parada
             ed   str   endereco/logradouro
             py   float latitude
             px   float longitude

Posicao      hr   str   hora de referencia da consulta ("HH:MM")
             l    list  linhas; dentro de cada uma:
               c    str   letreiro completo ("8000-10")
               cl   int   codigo da linha
               sl   int   sentido
               lt0  str   letreiro de destino (ida)
               lt1  str   letreiro de destino (volta)
               qv   int   quantidade de veiculos localizados
               vs   list  veiculos:
                 p   str      prefixo do veiculo (identificador da frota)
                 a   bool     acessivel a pessoa com deficiencia
                 ta  str      timestamp UTC da localizacao (ISO 8601, "...Z")
                 py  float    latitude
                 px  float    longitude

Previsao     hr   str   hora de referencia
             p    obj   ponto (em /Previsao e /Previsao/Parada)
             ps   list  pontos (em /Previsao/Linha)
               cp, np, py, px  -> como em Parada
               l / vs          -> linhas e veiculos, cada veiculo com:
                 t   str   HORARIO PREVISTO DE CHEGADA, "HH:MM". O fuso NAO e
                           documentado; medido nos dados coletados como
                           America/Sao_Paulo, e confirmado a cada coleta por
                           `coleta.calibrar_fuso_previsao`. Pode passar da
                           meia-noite.
                 sv  ?     nao documentado (observado nulo)
                 is  ?     nao documentado (observado nulo)

O campo `t` e o alvo do estudo: e a previsao que o passageiro ve no ponto.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

import requests

log = logging.getLogger("olhovivo.api")

BASE_PADRAO = "https://api.olhovivo.sptrans.com.br/v2.1"
UA = "olhovivo-each/1.0 (pesquisa academica; coleta de atraso de onibus)"


class OlhoVivoError(RuntimeError):
    """Falha generica de comunicacao com a API."""


class ErroAutenticacao(OlhoVivoError):
    """Token recusado ou sessao impossivel de renovar."""


def mascarar(token: str) -> str:
    """Nunca imprima o token inteiro — nem em log, nem em mensagem de erro."""
    if not token:
        return "(vazio)"
    return f"{token[:4]}...{token[-4:]} ({len(token)} chars)"


def _mensagem_login_recusado(resposta, corpo: str, token: str) -> str:
    """
    Mensagem util para o caso mais confuso da API.

    A SPTrans NAO devolve 401 para token invalido: devolve **HTTP 200 com o
    corpo `false`** e sem o cookie de sessao. Como o status e 200, e facil
    achar que o problema esta no codigo. Nao esta: se voce chegou ate aqui, a
    requisicao saiu correta e o servidor recusou a credencial.
    """
    if resposta.status_code == 411:
        return (
            "HTTP 411 (Length Required): o POST de login foi enviado sem "
            "Content-Length. Isso e bug do cliente, nao do token."
        )

    linhas = [
        f"login recusado pela SPTrans (HTTP {resposta.status_code}, corpo={corpo!r}).",
        f"Token enviado: {mascarar(token)}.",
        "",
        "A requisicao saiu correta - a API respondeu 200 e disse `false`, que e",
        "como ela recusa credencial. Um token invalido e um token inexistente",
        "produzem exatamente esta mesma resposta. Verifique, nesta ordem:",
        "",
        "  1. O cadastro foi CONFIRMADO por e-mail?",
        "     https://www.sptrans.com.br/desenvolvedores/cadastro-desenvolvedores/",
        "  2. O token saiu de 'Meus Aplicativos' no portal do desenvolvedor da",
        "     SPTrans? Cada aplicativo tem a sua propria chave.",
        "  3. O token foi regerado depois que voce copiou? A chave antiga para",
        "     de funcionar na hora.",
        "  4. E o token do Olho Vivo mesmo, e nao uma subscription key do",
        "     gateway APILIB da Prefeitura? Sao credenciais diferentes.",
        "",
        "Para testar um candidato sem mexer no .env:",
        "  python -m olhovivo testar --token SEU_TOKEN",
    ]
    return "\n".join(linhas)


class ClienteOlhoVivo:
    """Cliente com sessao persistente, rate limit de cortesia e retry."""

    def __init__(
        self,
        token: str,
        base_url: str = BASE_PADRAO,
        timeout_s: float = 25.0,
        max_tentativas: int = 5,
        backoff_base_s: float = 1.5,
        intervalo_minimo_s: float = 0.12,
        permitir_fallback_http: bool = True,
    ):
        self.token = token
        self.base = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tentativas = max_tentativas
        self.backoff_base_s = backoff_base_s
        self.intervalo_minimo_s = intervalo_minimo_s
        self.permitir_fallback_http = permitir_fallback_http

        self.sessao = requests.Session()
        self.sessao.headers.update({"User-Agent": UA, "Accept": "application/json"})

        self._trava = threading.Lock()
        self._trava_auth = threading.Lock()
        self._ultimo_request = 0.0
        self._autenticado = False

        self.stats = {"requisicoes": 0, "erros": 0, "reautenticacoes": 0}

    # ------------------------------------------------------------------ auth
    def autenticar(self) -> bool:
        with self._trava_auth:
            url = f"{self.base}/Login/Autenticar"
            try:
                # data=b"" garante o header Content-Length: sem ele o IIS da
                # SPTrans responde 411 Length Required (em HTML, nao JSON)
                r = self.sessao.post(
                    url,
                    params={"token": self.token},
                    data=b"",
                    timeout=self.timeout_s,
                )
            except requests.exceptions.SSLError:
                if not self._tentar_fallback_http():
                    raise
                self._autenticado = False
                return self.autenticar()
            except requests.RequestException as exc:
                raise OlhoVivoError(f"falha de rede no login: {exc}") from exc

            corpo = (r.text or "").strip().strip('"').lower()
            ok = r.status_code == 200 and corpo == "true"
            self._autenticado = ok
            if not ok:
                raise ErroAutenticacao(_mensagem_login_recusado(r, corpo, self.token))
            log.info("autenticado na API Olho Vivo (%s)", self.base)
            return True

    def _tentar_fallback_http(self) -> bool:
        if not self.permitir_fallback_http or self.base.startswith("http://"):
            return False
        self.base = self.base.replace("https://", "http://", 1)
        log.warning("erro de TLS na SPTrans; caindo para %s", self.base)
        return True

    def garantir_sessao(self) -> None:
        if not self._autenticado:
            self.autenticar()

    # --------------------------------------------------------------- request
    def _respeitar_ritmo(self) -> None:
        with self._trava:
            agora = time.monotonic()
            espera = self.intervalo_minimo_s - (agora - self._ultimo_request)
            if espera > 0:
                time.sleep(espera)
            self._ultimo_request = time.monotonic()

    def get(self, caminho: str, params: dict[str, Any] | None = None) -> Any:
        """GET autenticado com retry/backoff. Devolve o JSON ja decodificado."""
        self.garantir_sessao()
        sufixo = caminho.lstrip("/")
        url = f"{self.base}/{sufixo}"
        ultimo_erro: Exception | None = None

        for tentativa in range(1, self.max_tentativas + 1):
            self._respeitar_ritmo()
            try:
                self.stats["requisicoes"] += 1
                r = self.sessao.get(url, params=params, timeout=self.timeout_s)
            except requests.exceptions.SSLError as exc:
                ultimo_erro = exc
                if self._tentar_fallback_http():
                    url = f"{self.base}/{sufixo}"
                    self._autenticado = False
                    self.garantir_sessao()
                    continue
                raise OlhoVivoError(f"TLS: {exc}") from exc
            except requests.RequestException as exc:
                ultimo_erro = exc
                self.stats["erros"] += 1
                self._dormir(tentativa)
                continue

            # sessao caiu -> reautentica e repete
            if r.status_code in (401, 403):
                self.stats["reautenticacoes"] += 1
                self._autenticado = False
                log.warning(
                    "sessao expirada (HTTP %s) em %s; reautenticando",
                    r.status_code,
                    caminho,
                )
                self.autenticar()
                continue

            if r.status_code == 429 or 500 <= r.status_code < 600:
                ultimo_erro = OlhoVivoError(f"HTTP {r.status_code} em {caminho}")
                self.stats["erros"] += 1
                self._dormir(tentativa, servidor=True)
                continue

            if r.status_code != 200:
                raise OlhoVivoError(
                    f"HTTP {r.status_code} em {caminho}: {r.text[:200]}"
                )

            if not r.content or not r.text.strip():
                return None
            try:
                return r.json()
            except ValueError as exc:
                ultimo_erro = exc
                self.stats["erros"] += 1
                log.warning("resposta nao-JSON em %s: %r", caminho, r.text[:160])
                self._dormir(tentativa)
                continue

        raise OlhoVivoError(
            f"esgotadas {self.max_tentativas} tentativas em {caminho}: {ultimo_erro}"
        )

    def _dormir(self, tentativa: int, servidor: bool = False) -> None:
        base = self.backoff_base_s * (2 ** (tentativa - 1))
        if servidor:
            base *= 1.5
        time.sleep(min(60.0, base) + random.uniform(0, 0.4))

    # ------------------------------------------------------------- endpoints
    # Linhas
    def buscar_linhas(self, termo: str) -> list[dict]:
        return self.get("Linha/Buscar", {"termosBusca": termo}) or []

    def carregar_detalhes(self, codigo_linha: int) -> list[dict]:
        """Existe na v2.1 mas nao esta na documentacao (so na v0, ja desativada)."""
        return self.get("Linha/CarregarDetalhes", {"codigoLinha": codigo_linha}) or []

    def buscar_linha_sentido(self, termo: str, sentido: int) -> list[dict]:
        return (
            self.get(
                "Linha/BuscarLinhaSentido",
                {"termosBusca": termo, "sentido": sentido},
            )
            or []
        )

    # Paradas
    def buscar_paradas(self, termo: str) -> list[dict]:
        return self.get("Parada/Buscar", {"termosBusca": termo}) or []

    def paradas_por_linha(self, codigo_linha: int) -> list[dict]:
        return (
            self.get("Parada/BuscarParadasPorLinha", {"codigoLinha": codigo_linha})
            or []
        )

    def paradas_por_corredor(self, codigo_corredor: int) -> list[dict]:
        return (
            self.get(
                "Parada/BuscarParadasPorCorredor", {"codigoCorredor": codigo_corredor}
            )
            or []
        )

    # Referencias
    def corredores(self) -> list[dict]:
        return self.get("Corredor") or []

    def empresas(self) -> dict:
        return self.get("Empresa") or {}

    # Posicoes
    def posicoes(self) -> dict:
        return self.get("Posicao") or {}

    def posicoes_linha(self, codigo_linha: int) -> dict:
        return self.get("Posicao/Linha", {"codigoLinha": codigo_linha}) or {}

    def posicoes_garagem(
        self, codigo_empresa: int, codigo_linha: int | None = None
    ) -> dict:
        params: dict[str, Any] = {"codigoEmpresa": codigo_empresa}
        if codigo_linha is not None:
            params["codigoLinha"] = codigo_linha
        return self.get("Posicao/Garagem", params) or {}

    # Previsoes
    def previsao(self, codigo_parada: int, codigo_linha: int) -> dict:
        return (
            self.get(
                "Previsao",
                {"codigoParada": codigo_parada, "codigoLinha": codigo_linha},
            )
            or {}
        )

    def previsao_linha(self, codigo_linha: int) -> dict:
        return self.get("Previsao/Linha", {"codigoLinha": codigo_linha}) or {}

    def previsao_parada(self, codigo_parada: int) -> dict:
        return self.get("Previsao/Parada", {"codigoParada": codigo_parada}) or {}

    # Velocidade nas vias (KMZ)
    def baixar_kmz(self, destino, tipo: str = "") -> Any:
        """
        Salva o KMZ de fluidez do transito.

        Atencao: o /KMZ NAO traz o tracado das linhas de onibus — traz o mapa
        de velocidade media e tempo de percurso por trecho de via. E util como
        variavel explicativa do atraso, nao como itinerario.

        `tipo`: "" | "BC" | "CB" | "Corredor" | "Corredor/BC" | "OutrasVias" ...
        (BC = bairro->centro, CB = centro->bairro)
        """
        from pathlib import Path

        self.garantir_sessao()
        caminho = f"{self.base}/KMZ" + (f"/{tipo}" if tipo else "")
        r = self.sessao.get(caminho, timeout=self.timeout_s * 3)
        if r.status_code in (401, 403):
            self._autenticado = False
            self.autenticar()
            r = self.sessao.get(caminho, timeout=self.timeout_s * 3)
        r.raise_for_status()
        alvo = Path(destino)
        alvo.parent.mkdir(parents=True, exist_ok=True)
        alvo.write_bytes(r.content)
        return alvo

    # ------------------------------------------------------------------ util
    def fechar(self) -> None:
        self.sessao.close()

    def __enter__(self) -> "ClienteOlhoVivo":
        self.autenticar()
        return self

    def __exit__(self, *_exc) -> None:
        self.fechar()


def cliente_de_config(cfg) -> ClienteOlhoVivo:
    """Constroi o cliente a partir de um objeto Config."""
    return ClienteOlhoVivo(
        token=cfg.token,
        base_url=cfg.get("api.base_url", BASE_PADRAO),
        timeout_s=cfg.get("api.timeout_s", 25.0),
        max_tentativas=cfg.get("api.max_tentativas", 5),
        backoff_base_s=cfg.get("api.backoff_base_s", 1.5),
        intervalo_minimo_s=cfg.get("api.intervalo_minimo_s", 0.12),
        permitir_fallback_http=cfg.get("api.permitir_fallback_http", True),
    )
