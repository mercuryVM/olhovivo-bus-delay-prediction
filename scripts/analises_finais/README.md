# Análises finais

Scripts de análise que sustentam os números do relatório final e da apresentação.
Todos só leem os dados (`/dados/derivado`, montado só para leitura) e escrevem o
resultado na saída padrão. Rodaram como Jobs no cluster, na base de 30 dias.
Nenhum treina ou ajusta modelo: a semana de teste (02 a 08/11) foi avaliada uma única vez.

| Script | O que calcula |
|---|---|
| `descritivo.py` | Proporção de passagens atrasadas por tipo de dia e hora, velocidade, erro da API |
| `pico_velocidade_atraso.py` | Velocidade por hora contra % de atrasadas; pico x fora do pico no dia útil |
| `frota_outras_linhas_atraso.py` | Velocidade dos ônibus de outras linhas (15 min antes) contra % de atrasadas |
| `sensibilidade_corte.py` | Cortes de 10% a 50%: tipos de dia e ranking de zonas |
| `atraso_por_data.py` | % de atrasadas por data, intervalos por bootstrap, 2º turno contra os outros domingos |
| `zonas_sem_terminal.py` | Taxa por zona OD sem os trechos de terminal |
| `zonas_robustez.py` | Zona da parada de origem, bootstrap da linha 5300, linha x zona |
| `zonas_excesso.py` | Excesso de atraso por zona contra o esperado pela linha, hora e tipo de dia (bootstrap de dias, Bonferroni) e queda de velocidade |
| `viagem_sintetica.py` | Viagens montadas com trechos de outras viagens (compensação entre trechos) |
| `episodios_recorrencia.py` | Episódios ST-DBSCAN por zona e dia útil (fim de tarde) |
| `episodios_acaso.py` | Ônibus por episódio e teste ao acaso da recorrência |
| `tatuape_velocidade.py` | Velocidade em Tatuapé, dia útil x domingo, nas linhas e na frota |
| `teste_final_extra.py` | Métricas extras do teste final a partir das previsões guardadas |
| `modelo_por_linha.py` | Erro do XGBoost e do modelo simples por linha no teste |
| `piso_erro.py` | Diferença entre pares de ônibus no mesmo trecho e estimativa do piso do erro |
