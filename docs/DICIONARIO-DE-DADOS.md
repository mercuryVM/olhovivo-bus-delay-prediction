# Dicionário de Dados

Conjunto de dados de saída do projeto, produzido a partir de 30 dias de coleta
da API Olho Vivo da SPTrans. A coleta bruta somou 408 milhões de posições de
veículos (a frota inteira da cidade) e 70,5 milhões de previsões de chegada.
Delas foram reconstruídas 3.031.494 chegadas reais às paradas e 2.848.075
passagens por trecho.

| | |
|---|---|
| Período | 02/09/2026 19h a 09/09/2026 19h (7 dias, com pausa diária das 01h às 04h) e 16/10/2026 20h a 08/11/2026 20h (23 dias, 24 h por dia); 32 datas no calendário |
| Linhas de ônibus | 10, nos dois sentidos (29 códigos de linha × sentido) |
| Paradas e trechos | Paradas e traçado do GTFS da SPTrans; 676 trechos entre paradas consecutivas |
| Semana de teste | 02 a 08/11/2026, reservada para a avaliação final dos modelos |
| Formato | Parquet (`derivado/`) |

---

## Conjunto principal — `percursos`

**Unidade de observação:** uma passagem, isto é, um veículo percorrendo um
trecho entre duas paradas consecutivas, numa viagem.

| | |
|---|---|
| Linhas | 2.848.075 |
| Viagens | 80.923 |
| Veículos | 1.236 |
| Colunas | 32, das quais 16 principais |
| Taxa de atraso | 24,9 % das passagens 20 % ou mais acima do tempo de referência (27,2 % no dia útil, 21,2 % no sábado, 15,1 % no domingo e feriado) |

### Principais colunas

| Coluna | Tipo | Descrição |
|---|---|---|
| `atraso_s` | numérica | **Variável resposta.** Desvio, em segundos, entre o tempo de percurso observado e o de referência do mesmo trecho e faixa horária |
| `atrasado` | binária | 1 quando o desvio é de 20 % ou mais do tempo de referência |
| `tempo_percurso_s` | numérica | Tempo observado no trecho |
| `percurso_tipico_s` | numérica | Tempo de referência: mediana do mesmo trecho, sentido e faixa horária, com todos os dias juntos |
| `letreiro` | categórica | Linha (ex.: `6000`) |
| `sentido` | categórica | 1 = terminal principal → secundário; 2 = inverso |
| `trecho` | categórica | Identificador do trecho dentro do itinerário (`cl:ordem`) |
| `cp_origem`, `cp_destino` | categórica | Paradas (código do GTFS) que delimitam o trecho |
| `regiao_origem` | categórica | Terminal de origem da linha |
| `regiao_destino` | categórica | Terminal de destino da linha |
| `hora` | categórica | Hora do dia (0–23), no horário de São Paulo |
| `dia_semana` | categórica | 0 = segunda … 6 = domingo |
| `faixa` | categórica | madrugada, pico_manha, entrepico, pico_tarde, noite |
| `headway_s` | numérica | Intervalo até o ônibus anterior da linha na parada |
| `velocidade_kmh` | numérica | Velocidade média no trecho |
| `lat`, `lon` | geográfica | Ponto médio do trecho |

### Demais colunas

`cl`, `prefixo`, `viagem_id`, `ordem_origem`, `ordem_destino`, `t_origem`,
`t_chegada`, `atraso_rel`, `n_referencia`, `extensao_m`, `pico`,
`fim_de_semana`, `celula`, `confianca` — identificação, geometria e controle de
qualidade.

> A mediana de `atraso_s` é 0 s por construção — a referência é a própria
> mediana do trecho. A informação está na cauda (p90 de 39,8 s) e na taxa.

---

## Conjuntos complementares

### `chegadas`

**Unidade de observação:** a chegada estimada de um veículo a uma parada. Base
dos `percursos`. 3.031.494 linhas.

| Coluna | Tipo | Descrição |
|---|---|---|
| `cl`, `letreiro`, `sentido`, `prefixo`, `viagem_id` | categórica | Linha, veículo e viagem |
| `cp`, `ordem` | categórica | Parada e sua posição no itinerário |
| `t_chegada` | temporal | Instante estimado de chegada (UTC) |
| `metodo` | categórica | `cruzamento` (o veículo passou pela parada entre duas leituras) ou `aproximacao` (leitura mais próxima; confiança menor) |
| `dist_min_m` | numérica | Menor distância entre o veículo e a parada |
| `salto_m`, `salto_s` | numérica | Distância e tempo entre as duas leituras usadas na interpolação |
| `velocidade_kmh` | numérica | Velocidade entre as duas leituras |
| `confianca` | numérica | 0 a 1; cai quando o salto entre leituras é grande ou o veículo passa longe da parada |
| `lat`, `lon` | geográfica | Posição da parada |

### `previsao_realizado`

Compara o horário previsto pela API com o horário em que o veículo de fato
chegou. Mede a **qualidade da previsão da SPTrans**, e não a irregularidade da
operação.

**Unidade de observação:** uma previsão de chegada e a chegada correspondente.
58.712.357 linhas (das 70,5 milhões de previsões coletadas) e 21 colunas.

| Coluna | Tipo | Descrição |
|---|---|---|
| `erro_s`, `erro_abs_s` | numérica | Chegada real menos horário previsto, e seu valor absoluto. Positivo = chegou depois do prometido |
| `horizonte_s` | numérica | Antecedência da previsão. **Toda comparação precisa ser estratificada por ela** |
| `letreiro`, `cl`, `sentido` | categórica | Linha |
| `cp`, `parada_nome` | categórica | Parada |
| `prefixo`, `viagem_id` | categórica | Veículo e viagem |
| `ts_coleta`, `t_previsto`, `t_chegada` | temporal | Momento da consulta, horário prometido e horário realizado |
| `headway_obs_s` | numérica | Intervalo até o ônibus anterior |
| `hora`, `dia_semana`, `faixa` | categórica | Recortes temporais |
| `lat`, `lon`, `celula` | geográfica | Localização da parada |
| `confianca` | numérica | Confiança da chegada casada |

Só são casadas chegadas até 60 min depois da consulta; por isso o erro das
previsões de 30 a 60 min está subestimado.

### `frota` (particionado por dia, `frota/dt=AAAA-MM-DD`)

Velocidade da **frota inteira** da cidade sobre os 676 trechos estudados, usada
como sensor de trânsito. 36.339.879 linhas.

| Coluna | Tipo | Descrição |
|---|---|---|
| `t` | temporal | Instante da leitura (UTC) |
| `vel_ms` | numérica | Velocidade do veículo, em m/s |
| `cl_medidor` | categórica | Código de linha × sentido do veículo que mediu (qualquer linha da cidade) |
| `prefixo` | categórica | Veículo |
| `cp_origem`, `cp_destino` | categórica | Trecho sobre o qual a leitura caiu |

### `frota_atributos`

Atributos da frota por passagem, usados pelos modelos. Uma linha por passagem
(2.848.075), ligada pelo índice e por `viagem_id`.

| Coluna | Tipo | Descrição |
|---|---|---|
| `frota_vel10_kmh`, `frota_n10` | numérica | Velocidade mediana da frota no trecho nos 10 min anteriores e número de leituras |
| `frota_vel30_kmh`, `frota_n30` | numérica | O mesmo, nos 30 min anteriores |
| `frota_idade_s` | numérica | Tempo entre a leitura mais recente da frota e o início da passagem |
| `frota_tempo_s` | numérica | Extensão do trecho dividida pela velocidade da frota (10 min; se faltar, 30 min) |

### GeoSampa (`geosampa_trechos`, `geosampa_faixas`, `geosampa_feiras`)

Camadas do GeoSampa por trecho, testadas como atributos e deixadas de fora do
modelo final (ganho de 0,02 a 0,03 s).

| Arquivo | Colunas | Descrição |
|---|---|---|
| `geosampa_trechos` | `cp_origem`, `cp_destino`, `extensao_m`, `geo_semaforos`, `geo_semaforos_km`, `geo_corredor`, `geo_faixa` | Semáforos, corredor e faixa exclusiva em cada trecho (687 linhas) |
| `geosampa_faixas` | `dia`, `h_ini`, `h_fim`, `fracao`, `cp_origem`, `cp_destino` | Horários de funcionamento das faixas exclusivas por trecho |
| `geosampa_feiras` | `dia`, `h_ini`, `h_fim`, `cp_origem`, `cp_destino` | Feiras livres que afetam o trecho, por dia da semana e horário |

---

## Como o atraso é definido

A API não publica horário por trecho, e o GTFS traz só uma viagem-modelo de
duração fixa, folgada em relação à operação real. Por isso o tempo de referência
é a **mediana observada** do mesmo trecho, sentido e faixa horária, calculada
com os 30 dias (a mesma para todos os dias da semana) — robusta a valores
extremos. Trechos com menos de 5 observações ficam sem referência e são
excluídos. Uma passagem está atrasada quando leva 20 % ou mais acima dessa
referência; com cortes de 10 % a 50 %, a ordem dia útil > sábado > domingo e o
ranking das zonas praticamente não mudam.

Nos modelos preditivos, a referência é recalculada só com os dias anteriores ao
dia previsto, para não usar informação do futuro.

O horário de chegada também não é publicado por nenhuma fonte: é reconstruído a
partir das posições GPS, projetando veículo e paradas sobre o traçado da linha e
tomando o instante em que a posição do veículo cruza a da parada.

---

## Limitações

- **A chegada é estimada, não observada.** A interpolação supõe velocidade
  constante entre amostras de GPS, mas o ônibus desacelera justamente na parada.
  O erro daí é sempre positivo (+12 a +35 s) e maior em pontos de grande
  embarque — logo, maior nas áreas centrais. Comparações entre regiões de
  demanda muito diferente devem ser lidas com essa ressalva.
- **A coleta tem duas janelas.** Entre 09/09 e 16/10 não há dados, e na
  primeira semana a coleta pausou todos os dias das 01h às 04h. A faixa
  `madrugada` fica sub-representada nessa semana.
- **A referência por faixa horária cria saltos.** Às 5h e às 20h a referência
  muda de faixa, o que produz picos artificiais na proporção de atrasadas
  nessas horas.
- **Trechos de terminal.** A saída e a chegada no terminal (5,8 % das
  passagens) incluem tempo parado e distorcem viagem e zonas; as análises por
  zona e por viagem os excluem.
- **Só existe dado onde a API publica previsão.** Das 150 linhas de maior frota,
  a maioria não tem previsão publicada em parcela relevante do itinerário; as
  linhas monitoradas foram escolhidas por esse critério, então não constituem
  amostra aleatória da rede.
- **Sazonalidade parcial.** Os 30 dias incluem feriados (07/09 e 02/11) e o
  segundo turno das eleições (25/10), mas não férias escolares nem outros
  meses, e a chuva não foi medida.
- **Só entra veículo em circulação.** Viagem não realizada não aparece, o que
  subestima o problema sentido pelo passageiro.

---

Detalhamento do método, formalização das medidas e referências:
[`apendice-metodologico.md`](apendice-metodologico.md).
