-- =====================================================================
-- Consultas DuckDB sobre os Parquet gerados pelo pipeline.
--
--   duckdb
--   .read sql/analytics.sql
--
-- Ajuste o caminho da base se rodar de outro diretorio.
-- =====================================================================

SET VARIABLE base = 'dados';

-- Views sobre os datasets particionados -------------------------------
CREATE OR REPLACE VIEW posicoes AS
  SELECT * FROM read_parquet('dados/bruto/posicoes/**/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW previsoes AS
  SELECT * FROM read_parquet('dados/bruto/previsoes/**/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW chegadas AS
  SELECT * FROM read_parquet('dados/derivado/chegadas.parquet');

CREATE OR REPLACE VIEW pr AS
  SELECT * FROM read_parquet('dados/derivado/previsao_realizado.parquet');

CREATE OR REPLACE VIEW paradas AS
  SELECT * FROM read_parquet('dados/catalogo/paradas.parquet');

CREATE OR REPLACE VIEW linha_parada AS
  SELECT * FROM read_parquet('dados/catalogo/linha_parada.parquet');


-- 1. Cobertura da coleta: existe buraco na semana? --------------------
-- Um dia com muito menos amostra que os outros indica lacuna, nao
-- necessariamente menos onibus. Confira contra `python -m olhovivo status`.
SELECT
    date_trunc('hour', ts_coleta)               AS hora,
    count(*)                                    AS amostras,
    count(DISTINCT prefixo)                     AS veiculos,
    count(DISTINCT cl)                          AS linhas
FROM posicoes
GROUP BY 1
ORDER BY 1;


-- 2. Ranking das linhas por probabilidade de atraso -------------------
SELECT
    letreiro,
    sentido,
    count(*)                                            AS n,
    round(avg(erro_s), 1)                               AS erro_medio_s,
    round(median(erro_s), 1)                            AS erro_mediano_s,
    round(quantile_cont(erro_s, 0.9), 1)                AS p90_erro_s,
    round(avg(abs(erro_s)), 1)                          AS mae_s,
    round(avg(CASE WHEN erro_s >= 300 THEN 1 ELSE 0 END), 4) AS prob_atraso_5min,
    round(avg(CASE WHEN erro_s <= -60 THEN 1 ELSE 0 END), 4) AS prob_adiantado_1min,
    round(median(headway_obs_s) / 60.0, 1)              AS headway_mediano_min
FROM pr
WHERE confianca >= 0.5
GROUP BY 1, 2
HAVING count(*) >= 100
ORDER BY prob_atraso_5min DESC
LIMIT 40;


-- 3. O erro cresce com o horizonte de previsao? -----------------------
-- Esta e a leitura mais importante do estudo: uma previsao para daqui a
-- 3 minutos e outra para daqui a 40 minutos nao sao comparaveis.
SELECT
    CASE
        WHEN horizonte_s <  300 THEN '0-5 min'
        WHEN horizonte_s <  600 THEN '5-10 min'
        WHEN horizonte_s < 1200 THEN '10-20 min'
        WHEN horizonte_s < 1800 THEN '20-30 min'
        ELSE '30-60 min'
    END                                         AS faixa_horizonte,
    count(*)                                    AS n,
    round(avg(abs(erro_s)), 1)                  AS mae_s,
    round(median(erro_s), 1)                    AS vies_mediano_s,
    round(quantile_cont(erro_s, 0.1), 1)        AS p10_s,
    round(quantile_cont(erro_s, 0.9), 1)        AS p90_s
FROM pr
WHERE confianca >= 0.5
GROUP BY 1
ORDER BY min(horizonte_s);


-- 4. Perfil por faixa horaria e dia da semana -------------------------
SELECT
    faixa,
    dia_semana,   -- 0 = segunda
    count(*)                                    AS n,
    round(avg(erro_s), 1)                       AS erro_medio_s,
    round(avg(CASE WHEN erro_s >= 300 THEN 1 ELSE 0 END), 4) AS prob_atraso_5min
FROM pr
WHERE confianca >= 0.5
GROUP BY 1, 2
ORDER BY prob_atraso_5min DESC;


-- 5. Regioes (celulas) mais impactadas --------------------------------
SELECT
    celula,
    count(*)                                    AS n,
    count(DISTINCT cl)                          AS linhas_afetadas,
    round(avg(lat), 5)                          AS lat,
    round(avg(lon), 5)                          AS lon,
    round(avg(erro_s), 1)                       AS erro_medio_s,
    round(quantile_cont(erro_s, 0.9), 1)        AS p90_erro_s,
    round(avg(CASE WHEN erro_s >= 300 THEN 1 ELSE 0 END), 4) AS prob_atraso_5min
FROM pr
WHERE confianca >= 0.5
GROUP BY 1
HAVING count(*) >= 50
ORDER BY prob_atraso_5min DESC, n DESC
LIMIT 50;


-- 6. Paradas criticas, com nome e posicao no itinerario ---------------
SELECT
    pr.cp,
    any_value(pr.parada_nome)                   AS parada,
    any_value(pr.letreiro)                      AS linha,
    any_value(lp.ordem)                         AS ordem_no_itinerario,
    count(*)                                    AS n,
    round(avg(pr.erro_s), 1)                    AS erro_medio_s,
    round(avg(CASE WHEN pr.erro_s >= 300 THEN 1 ELSE 0 END), 4) AS prob_atraso_5min
FROM pr
LEFT JOIN linha_parada lp ON lp.cl = pr.cl AND lp.cp = pr.cp
WHERE pr.confianca >= 0.5
GROUP BY pr.cp
HAVING count(*) >= 50
ORDER BY prob_atraso_5min DESC
LIMIT 40;


-- 7. Regularidade: intervalo entre onibus (headway) --------------------
-- Coeficiente de variacao alto = bunching (onibus andando em comboio).
SELECT
    letreiro,
    sentido,
    count(*)                                        AS n,
    round(median(headway_obs_s) / 60.0, 1)          AS headway_mediano_min,
    round(stddev_samp(headway_obs_s) / 60.0, 1)     AS desvio_min,
    round(stddev_samp(headway_obs_s) / nullif(avg(headway_obs_s), 0), 3) AS coef_variacao
FROM pr
WHERE headway_obs_s BETWEEN 30 AND 5400
GROUP BY 1, 2
HAVING count(*) >= 100
ORDER BY coef_variacao DESC
LIMIT 30;


-- 8. Velocidade comercial por trecho ----------------------------------
-- Trechos lentos sao onde o atraso nasce.
WITH trechos AS (
    SELECT
        c.cl,
        c.viagem_id,
        c.ordem,
        c.cp,
        c.t_chegada,
        lag(c.t_chegada) OVER (PARTITION BY c.viagem_id ORDER BY c.ordem) AS t_anterior,
        lag(c.ordem)     OVER (PARTITION BY c.viagem_id ORDER BY c.ordem) AS ordem_anterior,
        lp.s_m,
        lag(lp.s_m)      OVER (PARTITION BY c.viagem_id ORDER BY c.ordem) AS s_anterior
    FROM chegadas c
    LEFT JOIN linha_parada lp ON lp.cl = c.cl AND lp.cp = c.cp
    WHERE c.confianca >= 0.5
)
SELECT
    cl,
    ordem_anterior                                  AS trecho,
    count(*)                                        AS viagens,
    round(median(s_m - s_anterior), 0)              AS extensao_m,
    round(median(epoch(t_chegada - t_anterior)), 0) AS tempo_mediano_s,
    round(
        median((s_m - s_anterior) / nullif(epoch(t_chegada - t_anterior), 0)) * 3.6,
        1
    )                                               AS velocidade_kmh
FROM trechos
WHERE ordem - ordem_anterior = 1
  AND s_m > s_anterior
  AND epoch(t_chegada - t_anterior) BETWEEN 10 AND 1800
GROUP BY 1, 2
HAVING count(*) >= 20
ORDER BY velocidade_kmh ASC
LIMIT 40;


-- 9. Qualidade da deteccao de chegada ---------------------------------
SELECT
    metodo,
    count(*)                            AS n,
    round(avg(confianca), 3)            AS confianca_media,
    round(median(dist_min_m), 1)        AS dist_mediana_m,
    round(median(salto_m), 1)           AS salto_mediano_m,
    round(median(salto_s), 1)           AS salto_mediano_s
FROM chegadas
GROUP BY 1;
