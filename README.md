# Olho Vivo — atraso de ônibus em São Paulo

Coleta e análise de dados da API Olho Vivo da SPTrans, para identificar padrões
probabilísticos de atraso das linhas de ônibus de São Paulo e as regiões mais
impactadas.

Trabalho de Sistemas de Informação — EACH/USP. Autores: Alan M. Silva e
Gabriel L. da Silva.

A base final tem **30 dias de coleta** (02 a 09/09/2026 e 16/10 a 08/11/2026):
408 milhões de posições da frota inteira, 70,5 milhões de previsões da API e
2,85 milhões de passagens reconstruídas em 676 trechos de 10 linhas.

O conjunto de dados dos 30 dias (`percursos`, `chegadas`, `previsao_realizado`,
a frota inteira por trecho, as camadas do GeoSampa e as métricas dos modelos)
está na release
[v2.0.0](https://github.com/mercuryVM/olhovivo-bus-delay-prediction/releases/tag/v2.0.0),
com somas de verificação. A primeira semana, em formato anterior, continua na
[v1.0.0](https://github.com/mercuryVM/olhovivo-bus-delay-prediction/releases/tag/v1.0.0).

- [`docs/DICIONARIO-DE-DADOS.md`](docs/DICIONARIO-DE-DADOS.md) — descrição do
  conjunto de dados de saída: linhas, principais colunas e limitações.
- [`docs/apendice-metodologico.md`](docs/apendice-metodologico.md) — detalhamento
  do método e referências.
- [`scripts/analises_finais/`](scripts/analises_finais/) — scripts que
  sustentam os números do relatório final e da apresentação.

---

## O problema

A API entrega dois lados da mesma coisa: a **previsão** de chegada nas paradas e
a **posição** de cada veículo, atualizada a cada ~30 s. O que ela **não** entrega
é o horário em que o ônibus de fato chegou.

Esse horário é reconstruído aqui, a partir das trajetórias. Dele saem as duas
medidas do trabalho: o desvio de tempo de percurso por trecho (a variável de
atraso) e o erro da previsão publicada.

---

## Principais resultados

Nas 10 linhas, nos 30 dias:

- **Quanto:** 1 em cada 4 passagens de dia útil leva 20 % ou mais acima do
  normal do trecho (27,2 %); no domingo e feriado, 1 em 7 (15,1 %). A ordem se
  mantém com cortes de 10 % a 50 %.
- **Quando:** às 18h de dia útil o ônibus anda a 17,7 km/h. Nas horas em que os
  ônibus de outras linhas andam mais devagar no mesmo trecho, as 10 linhas
  atrasam mais (r = −0,87 por hora; −0,71 no mesmo horário, de um dia para outro).
- **Onde:** o lugar pesa mais que a linha. As zonas da Pesquisa Origem-Destino
  vão de 1 em 7 a 1 em 3 passagens atrasadas; as linhas ficam todas perto de
  1 em 4. Na mesma linha 5300, 28 % na Praça João Mendes contra 18 % no Centro
  Cultural.
- **Episódios (ST-DBSCAN):** 7.792 episódios; 17 zonas têm atraso agrupado em
  15 ou mais dos 20 fins de tarde de dia útil, padrão que acompanha o volume de
  ônibus e a hora (teste ao acaso).
- **2º turno (25/10):** a cidade rodou com 2 em cada 3 ônibus de um dia útil; nas
  10 linhas a frota foi a de um domingo comum, e mesmo assim das 12h às 15h foi o
  pior dos 7 domingos e feriados.
- **Previsão da API:** otimista; a 15 a 30 min, 72 em cada 100 ônibus chegam
  1 min ou mais depois do previsto.
- **Modelos:** na semana de teste (02 a 08/11, avaliada uma única vez), o
  XGBoost erra 24,1 s por trecho contra 29,7 s do modelo simples (o tempo normal),
  e ganha nos 7 dias e nas 10 linhas. LSTM e GNN ficam a menos de 1 s. O limite
  estimado com estes dados é perto de 18 s.

---

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para a parte de análise:

```bash
pip install -r requirements-analise.txt
```

O token da API sai do cadastro gratuito em
<https://www.sptrans.com.br/desenvolvedores/>. Copie `.env.example` para `.env` e
preencha `SPTRANS_TOKEN`. O `MONGO_URI` é opcional — sem ele grava só em Parquet.

---

## Uso

```bash
python -m olhovivo testar      # valida token e conexão
python -m olhovivo gtfs        # baixa o GTFS estático (traçados e paradas)
python -m olhovivo catalogo    # mapeia linhas e paradas
python -m olhovivo coletar     # coleta contínua (padrão 7 dias; --horas muda)
```

Durante a coleta, `python -m olhovivo status` mostra o volume coletado e as
lacunas. No Windows, `scripts\coleta_semana.ps1` mantém a coleta rodando e
reinicia se cair. A coleta de 30 dias rodou num cluster Kubernetes, 24 h por dia.

Depois da coleta:

```bash
python -m olhovivo chegadas    # reconstrói as chegadas reais
python -m olhovivo atraso      # variável de atraso por trecho
python -m olhovivo casar       # previsão × realizado
python -m olhovivo regioes     # K-means, DBSCAN, HDBSCAN, ST-DBSCAN
```

A velocidade da frota inteira sobre os trechos estudados (`olhovivo/frota.py`)
funciona como sensor de trânsito e é o grupo de atributos que mais ajudou os
modelos.

### Modelos e avaliação

```bash
python scripts/avaliacao_diaria.py                 # dias de escolha (21 a 23/10), só XGBoost
python scripts/avaliacao_diaria.py --redes         # + LSTM, GNN e o conjunto
python scripts/avaliacao_diaria.py --final --redes # semana de teste, 02 a 08/11
```

A avaliação é dia a dia (walk-forward): para prever um dia, os modelos treinam
só com os dias anteriores, e o tempo normal do trecho também é recalculado só com
eles. As escolhas de configuração usaram só dias de outubro; a semana de 02 a
08/11 foi rodada uma única vez, no fim.

### Painel

Os resultados da primeira semana podem ser explorados num mapa interativo de
São Paulo:

```bash
streamlit run app/painel.py
```

O painel lê as saídas de `dados/derivado/` (ou de `dados_semana/`, se existir;
a variável `OLHOVIVO_DADOS` aponta outra pasta) e mostra o atraso por parada, as
regiões encontradas pelos agrupamentos, os fluxos origem → destino, os padrões
por hora e dia, e o erro da previsão da API. A aba de previsão usa
`painel_previsao.parquet`, gerado por `scripts/agrega_previsao.py`.

Dois comandos auxiliares ajudam a preparar a coleta:
`linhas-candidatas` ranqueia as linhas por quanto rendem de dado, e
`mapa --letreiro 8000` exporta o itinerário em GeoJSON para conferir no mapa.

---

## Como o atraso é definido

A SPTrans não publica tabela de horários pela API, então não existe "atraso
contra o programado". O trabalho usa três medidas:

| medida | o que é |
|---|---|
| `atraso_s` | desvio entre o tempo de percurso observado num trecho e a mediana do mesmo trecho, sentido e faixa horária |
| `headway_obs_s` | intervalo real entre ônibus consecutivos na parada |
| `erro_s` | chegada real menos o horário previsto pela API |

A primeira é a variável do estudo: a mediana usa os 30 dias, e uma passagem
conta como atrasada quando leva 20 % ou mais acima dela. A terceira mede a qualidade da previsão da
SPTrans, que é outra coisa.

---

## Como a chegada real é detectada

Detecção por raio não funciona: com posição a cada 30–60 s, o ônibus anda
250–500 m entre amostras e atravessa um buffer de 50 m sem ser visto.

O método usa **referenciamento linear**. O traçado vem do `shapes.txt` do GTFS;
paradas e posições são projetadas sobre ele, e a chegada é o instante em que a
abscissa curvilínea do veículo cruza a da parada, interpolado entre as duas
amostras que cercam o cruzamento. Cada chegada recebe um escore de confiança que
cai conforme o vão da interpolação cresce.

Detalhes e limitações em [`docs/DICIONARIO-DE-DADOS.md`](docs/DICIONARIO-DE-DADOS.md).

---

## Verificação

```bash
python scripts/autoteste.py
```

Monta uma linha com horários de chegada conhecidos, roda o pipeline em cima e
compara com a verdade — inclusive sobre a geometria real de uma linha da SPTrans,
se o GTFS já tiver sido baixado.

---

## Estrutura

```
olhovivo/
├── config/coleta.yaml     # parametrização
├── olhovivo/
│   ├── api.py             # cliente da API
│   ├── geo.py             # projeção e referenciamento linear
│   ├── storage.py         # Parquet particionado
│   ├── mongo.py           # MongoDB geoespacial
│   ├── gtfs.py            # GTFS estático
│   ├── catalogo.py        # linhas e paradas
│   ├── coleta.py          # coleta contínua
│   ├── chegadas.py        # detecção da chegada real
│   ├── atraso.py          # variável de atraso
│   ├── casamento.py       # previsão × realizado
│   ├── regioes.py         # agrupamento
│   ├── features.py        # atributos
│   ├── vizinhanca.py      # ônibus que acabaram de passar e previsão da API
│   ├── contexto_viagem.py # atraso acumulado na viagem, tendência, feriado
│   ├── frota.py           # frota inteira como sensor de trânsito
│   ├── geosampa.py        # semáforos, corredores, faixas, feiras (testado)
│   ├── programacao.py     # intervalo programado do GTFS (testado)
│   ├── osm.py             # atributos do OpenStreetMap (testado)
│   ├── modelos.py         # atributos de atraso e linhas de base
│   └── modelos_atraso.py  # XGBoost, LSTM (quantis) e GNN por trecho
├── app/painel.py          # painel Streamlit
├── scripts/
│   ├── avaliacao_diaria.py
│   └── analises_finais/   # números do relatório final
├── sql/analytics.sql
└── docs/
```

---

## Licenciamento

Os dados vêm da API Olho Vivo e do GTFS da SPTrans, públicos mediante cadastro.
Os termos da SPTrans vedam sublicenciamento e comercialização — o uso aqui é de
pesquisa. O conjunto não contém dados pessoais: registra veículos por número de
frota, não passageiros.
