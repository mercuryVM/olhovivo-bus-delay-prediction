"""Chuva horaria retroativa da Open-Meteo numa grade sobre a cidade de SP.

    python scripts/chuva_openmeteo.py saida.parquet 2026-09-02:2026-09-10 2026-10-16:2026-10-20

Os pluviometros do CEMADEN so podem ser lidos ao vivo (coletor_chuva no
cluster); para a semana de setembro e os dias antes dele ligar, esta e a
fonte. E chuva de modelo meteorologico (Historical Forecast API), nao
medicao: serve para "choveu forte nesta regiao nesta hora", nao para o
milimetro exato de um ponto. Grade de 0,05 grau (~5 km), horario local.
Sem chave.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LATS = np.round(np.arange(-23.85, -23.40 + 1e-9, 0.05), 2)
LONS = np.round(np.arange(-46.80, -46.35 + 1e-9, 0.05), 2)
LOTE = 50  # pontos por requisicao


def baixar(pontos: list[tuple[float, float]], inicio: str, fim: str) -> pd.DataFrame:
    q = urllib.parse.urlencode({
        "latitude": ",".join(str(p[0]) for p in pontos),
        "longitude": ",".join(str(p[1]) for p in pontos),
        "start_date": inicio, "end_date": fim,
        "hourly": "precipitation", "timezone": "America/Sao_Paulo",
    })
    with urllib.request.urlopen(f"{URL}?{q}", timeout=120) as r:
        dados = json.load(r)
    dados = dados if isinstance(dados, list) else [dados]
    partes = []
    for (lat, lon), d in zip(pontos, dados):
        h = d["hourly"]
        partes.append(pd.DataFrame({
            "lat": lat, "lon": lon,
            "hora_local": pd.to_datetime(h["time"]),
            "chuva_mm": h["precipitation"],
        }))
    return pd.concat(partes, ignore_index=True)


def main() -> None:
    saida = Path(sys.argv[1])
    periodos = [p.split(":") for p in sys.argv[2:]]
    pontos = [(float(a), float(b)) for a in LATS for b in LONS]
    partes = []
    for inicio, fim in periodos:
        for i in range(0, len(pontos), LOTE):
            partes.append(baixar(pontos[i:i + LOTE], inicio, fim))
            time.sleep(1)
        print(f"{inicio} a {fim}: ok")
    df = pd.concat(partes, ignore_index=True)
    df["hora_local"] = df["hora_local"].dt.tz_localize("America/Sao_Paulo")
    saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(saida, index=False)
    por_dia = df.groupby(df["hora_local"].dt.date)["chuva_mm"].agg(["mean", "max"])
    print(f"{len(pontos)} pontos, {len(df)} linhas -> {saida}")
    print("chuva por dia (media da grade e maximo de um ponto em 1 h, mm):")
    print(por_dia.round(2).to_string())


if __name__ == "__main__":
    main()
