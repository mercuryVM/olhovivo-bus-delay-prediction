# Olho Vivo — atraso de ônibus em São Paulo

Coleta e análise de dados da API Olho Vivo da SPTrans, para identificar padrões
probabilísticos de atraso das linhas de ônibus de São Paulo e as regiões mais
impactadas.

Trabalho de Sistemas de Informação — EACH/USP.

**Painel interativo com os resultados da semana coletada:
<https://onibus.tagg.chat>**

O conjunto de dados de saída (`percursos`, `previsao_realizado` e as regiões)
está nas [Releases](https://github.com/mercuryVM/olhovivo-bus-delay-prediction/releases)
do repositório, com somas de verificação.

- [`docs/DICIONARIO-DE-DADOS.md`](docs/DICIONARIO-DE-DADOS.md) — descrição do
  conjunto de dados de saída: linhas, principais colunas e limitações.
- [`docs/apendice-metodologico.md`](docs/apendice-metodologico.md) — detalhamento
  do método e referências.

---

## O problema

A API entrega dois lados da mesma coisa: a **previsão** de chegada nas paradas e
a **posição** de cada veículo, atualizada a cada ~30 s. O que ela **não** entrega
é o horário em que o ônibus de fato chegou.

Esse horário é reconstruído aqui, a partir das trajetórias. Dele saem as duas
medidas do trabalho: o desvio de tempo de percurso por trecho (a variável de
atraso) e o erro da previsão publicada.

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
python -m olhovivo coletar     # coleta contínua de 7 dias
```

Durante a semana, `python -m olhovivo status` mostra o volume coletado e as
lacunas. No Windows, `scripts\coleta_semana.ps1` mantém a coleta rodando e
reinicia se cair.

Depois da coleta:

```bash
python -m olhovivo chegadas    # reconstrói as chegadas reais
python -m olhovivo atraso      # variável de atraso por trecho
python -m olhovivo casar       # previsão × realizado
python -m olhovivo regioes     # K-means, DBSCAN, HDBSCAN, ST-DBSCAN
```

Os resultados podem ser explorados num mapa interativo de São Paulo:

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

A primeira é a variável do estudo. A terceira mede a qualidade da previsão da
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
│   └── modelos.py         # XGBoost, LSTM, GNN
├── scripts/
├── sql/analytics.sql
└── docs/
```

---

## Licenciamento

Os dados vêm da API Olho Vivo e do GTFS da SPTrans, públicos mediante cadastro.
Os termos da SPTrans vedam sublicenciamento e comercialização — o uso aqui é de
pesquisa. O conjunto não contém dados pessoais: registra veículos por número de
frota, não passageiros.
