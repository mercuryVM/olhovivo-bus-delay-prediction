"""Resume o previsao_realizado (milhões de linhas) numa tabela pequena para o
painel: erro da previsão por parada, linha, hora, dia da semana e antecedência.

    python scripts/agrega_previsao.py [pasta_dos_dados]

MAE e proporções ficam por grupo junto com o número de pares, para o painel
poder recombinar os grupos por média ponderada.
"""
import sys
from pathlib import Path

import duckdb

raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "dados")
origem = (raiz / "derivado" / "previsao_realizado.parquet").as_posix()
destino = (raiz / "derivado" / "painel_previsao.parquet").as_posix()

con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=2;")
con.execute(f"""
COPY (
  SELECT
    cp, any_value(parada_nome) AS parada_nome, letreiro, sentido,
    hora, dia_semana, any_value(faixa) AS faixa,
    CASE WHEN horizonte_s < 300 THEN '0-5 min'
         WHEN horizonte_s < 900 THEN '5-15 min'
         WHEN horizonte_s < 1800 THEN '15-30 min'
         ELSE '30-60 min' END AS horizonte,
    count(*) AS n,
    avg(erro_abs_s) AS mae_s,
    median(erro_s) AS erro_mediano_s,
    quantile_cont(erro_abs_s, 0.9) AS erro_abs_p90_s,
    avg(CASE WHEN erro_s >= 300 THEN 1 ELSE 0 END) AS p_atraso5,
    avg(CASE WHEN erro_s <= -60 THEN 1 ELSE 0 END) AS p_adiantado1,
    any_value(lat) AS lat, any_value(lon) AS lon
  FROM read_parquet('{origem}')
  GROUP BY ALL
) TO '{destino}' (FORMAT parquet, COMPRESSION zstd)
""")
grupos, pares = con.execute(f"SELECT count(*), sum(n) FROM '{destino}'").fetchone()
print(f"{destino}: {grupos} grupos, {pares} pares")
