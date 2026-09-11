# Dicionário de Dados

Conjunto de dados de saída do projeto, produzido a partir de uma semana de
coleta da API Olho Vivo da SPTrans. A coleta bruta somou 90,0 milhões de
posições de veículos e 15,8 milhões de previsões de chegada; delas foram
reconstruídas 451.409 chegadas reais às paradas.

---

## Conjunto principal — `percursos`

**Unidade de observação:** um veículo percorrendo um trecho entre duas paradas
consecutivas, numa viagem.

| | |
|---|---|
| Linhas | 421.325 |
| Viagens | 17.450 |
| Linhas de ônibus | 10, nos dois sentidos (29 códigos de linha × sentido) |
| Colunas | 32, das quais 16 principais |
| Período | 02/09/2026 19h22 a 09/09/2026 19h22, com pausa diária das 01h às 04h |
| Taxa de atraso | 24,4 % dos trechos acima de 20 % do tempo de referência |
| Formato | Parquet e MongoDB |

### Principais colunas

| Coluna | Tipo | Descrição |
|---|---|---|
| `atraso_s` | numérica | **Variável resposta.** Desvio, em segundos, entre o tempo de percurso observado e o de referência do mesmo trecho e faixa horária |
| `atrasado` | binária | 1 quando o desvio supera 20 % do tempo de referência |
| `tempo_percurso_s` | numérica | Tempo observado no trecho |
| `percurso_tipico_s` | numérica | Tempo de referência: mediana do mesmo trecho, sentido e faixa horária |
| `letreiro` | categórica | Linha (ex.: `6000`) |
| `sentido` | categórica | 1 = terminal principal → secundário; 2 = inverso |
| `trecho` | categórica | Identificador do trecho dentro do itinerário |
| `cp_origem`, `cp_destino` | categórica | Paradas que delimitam o trecho |
| `regiao_origem` | categórica | Terminal de origem da linha |
| `regiao_destino` | categórica | Terminal de destino da linha |
| `hora` | categórica | Hora do dia (0–23) |
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
> mediana do trecho. A informação está na cauda (p90 de 40,5 s) e na taxa.

---

## Conjunto complementar — `previsao_realizado`

Compara o horário previsto pela API com o horário em que o veículo de fato
chegou. Mede a **qualidade da previsão da SPTrans**, e não a irregularidade da
operação — por isso é secundário em relação ao `percursos`.

**Unidade de observação:** uma previsão de chegada e a chegada correspondente.
12.982.114 linhas — 88,7 % das previsões avaliadas casaram com uma chegada —
e 21 colunas.

| Coluna | Tipo | Descrição |
|---|---|---|
| `erro_s` | numérica | Chegada real menos horário previsto. Positivo = chegou depois do prometido |
| `horizonte_s` | numérica | Antecedência da previsão. **Toda comparação precisa ser estratificada por ela** |
| `letreiro`, `sentido` | categórica | Linha |
| `cp`, `parada_nome` | categórica | Parada |
| `t_previsto`, `t_chegada` | temporal | Horário prometido e horário realizado |
| `headway_obs_s` | numérica | Intervalo até o ônibus anterior |
| `hora`, `dia_semana`, `faixa` | categórica | Recortes temporais |
| `lat`, `lon`, `celula` | geográfica | Localização da parada |

---

## Como o atraso é definido

A API não publica tabela de horários, então não existe "atraso contra o
programado". O tempo de referência é a **mediana observada** do mesmo trecho,
sentido e faixa horária, calculada na própria semana de coleta — robusta a
valores extremos. Trechos com menos de 5 observações ficam sem referência e são
excluídos.

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
- **A madrugada foi coberta só em parte.** A coleta pausou todos os dias das 01h
  às 04h (21 h das 168 h), e houve uma interrupção de 15 min em 04/09 às 21h49.
  Nenhuma das linhas monitoradas é noturna, mas a faixa `madrugada` fica
  sub-representada.
- **Só existe dado onde a API publica previsão.** Das 150 linhas de maior frota,
  a maioria não tem previsão publicada em parcela relevante do itinerário; as
  linhas monitoradas foram escolhidas por esse critério, então não constituem
  amostra aleatória da rede.
- **Uma semana não cobre sazonalidade.** Feriado, chuva forte ou obra podem
  dominar a janela — e a semana coletada inclui o feriado de 07/09.
- **Só entra veículo em circulação.** Viagem não realizada não aparece, o que
  subestima o problema sentido pelo passageiro.

---

Detalhamento do método, formalização das medidas e referências:
[`apendice-metodologico.md`](apendice-metodologico.md).
