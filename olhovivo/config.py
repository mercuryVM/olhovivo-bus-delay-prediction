"""Carregamento de configuração: YAML + .env + variáveis de ambiente."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

def _raiz_projeto() -> Path:
    """
    Diretorio raiz do projeto, a partir da localizacao deste arquivo.

    NAO use `.resolve()` aqui: quando o pacote e montado por symlink, resolver
    o link desloca a raiz do projeto e o arquivo de configuracao passa a ser
    procurado no lugar errado. `.absolute()` normaliza o caminho sem seguir
    symlink.
    """
    return Path(__file__).absolute().parent.parent


log = logging.getLogger("olhovivo.config")

RAIZ_PROJETO = _raiz_projeto()
CONFIG_PADRAO = RAIZ_PROJETO / "config" / "coleta.yaml"


def _carregar_dotenv(caminho: Path) -> None:
    """Carrega o .env sem depender do python-dotenv (mas usa se estiver lá)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(caminho, override=False)
        return
    except ImportError:
        pass

    if not caminho.exists():
        return
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, _, valor = linha.partition("=")
        os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


def _converter(texto: str) -> Any:
    baixo = texto.strip().lower()
    if baixo in ("true", "false"):
        return baixo == "true"
    for cast in (int, float):
        try:
            return cast(texto)
        except ValueError:
            pass
    return texto


class Config:
    """Acesso por caminho pontilhado: cfg["coleta.posicoes.intervalo_s"]."""

    def __init__(self, dados: dict[str, Any], raiz: Path = RAIZ_PROJETO):
        self._dados = dados
        self.raiz = raiz

    # -- leitura ------------------------------------------------------------
    def get(self, caminho: str, padrao: Any = None) -> Any:
        # override por ambiente: OLHOVIVO_COLETA__POSICOES__INTERVALO_S
        env_key = "OLHOVIVO_" + caminho.upper().replace(".", "__")
        if env_key in os.environ:
            return _converter(os.environ[env_key])

        no: Any = self._dados
        for parte in caminho.split("."):
            if not isinstance(no, dict) or parte not in no:
                return padrao
            no = no[parte]
        return no

    def __getitem__(self, caminho: str) -> Any:
        valor = self.get(caminho, _AUSENTE)
        if valor is _AUSENTE:
            raise KeyError(f"chave de configuração ausente: {caminho}")
        return valor

    def secao(self, caminho: str) -> dict[str, Any]:
        return dict(self.get(caminho, {}) or {})

    # -- derivados ----------------------------------------------------------
    @property
    def dir_dados(self) -> Path:
        bruto = Path(self.get("armazenamento.raiz", "dados"))
        caminho = bruto if bruto.is_absolute() else self.raiz / bruto
        caminho.mkdir(parents=True, exist_ok=True)
        return caminho

    @property
    def token(self) -> str:
        tok = os.environ.get("SPTRANS_TOKEN", "").strip()
        if not tok or tok.startswith("cole_seu"):
            raise RuntimeError(
                "SPTRANS_TOKEN não definido. Copie .env.example para .env e "
                "preencha com o token obtido em "
                "https://www.sptrans.com.br/desenvolvedores/"
            )
        return tok

    @property
    def mongo_uri(self) -> str | None:
        """
        URI do MongoDB, com as aspas e espacos que o shell deixa passar.

        Um .env escrito com aspas em volta do valor faz o pymongo recusar a
        conexao com "Invalid URI scheme". Limpar aqui e mais barato do que
        depurar isso de novo.
        """
        bruto = (os.environ.get("MONGO_URI") or "").strip().strip("'\"").strip()
        if not bruto:
            return None
        if not bruto.startswith(("mongodb://", "mongodb+srv://")):
            log.error(
                "MONGO_URI ignorado: precisa comecar com mongodb:// ou "
                "mongodb+srv:// (recebido algo de %d chars comecando em %r)",
                len(bruto),
                bruto[:12],
            )
            return None
        return bruto

    @property
    def mongo_db(self) -> str:
        return (os.environ.get("MONGO_DB") or "olhovivo").strip().strip("'\"") or "olhovivo"

    @property
    def backends(self) -> list[str]:
        ativos = list(self.get("armazenamento.backends", ["parquet"]))
        if "mongo" in ativos and not self.mongo_uri:
            ativos = [b for b in ativos if b != "mongo"]
        return ativos


_AUSENTE = object()


def carregar(caminho: str | Path | None = None) -> Config:
    """Lê o YAML de configuração e o .env do projeto."""
    _carregar_dotenv(RAIZ_PROJETO / ".env")

    # OLHOVIVO_CONFIG permite apontar o YAML explicitamente — util em container,
    # onde o layout do disco nao e o do repositorio
    if caminho:
        alvo = Path(caminho)
    elif os.environ.get("OLHOVIVO_CONFIG"):
        alvo = Path(os.environ["OLHOVIVO_CONFIG"])
    else:
        alvo = CONFIG_PADRAO

    if not alvo.exists():
        raise FileNotFoundError(
            f"configuração não encontrada: {alvo}\n"
            "Aponte o caminho com --config ou com a variável OLHOVIVO_CONFIG."
        )

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML é obrigatório: pip install PyYAML") from exc

    with alvo.open("r", encoding="utf-8") as fh:
        dados = yaml.safe_load(fh) or {}
    return Config(dados)
