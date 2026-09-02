# Dicionário de Dados

Documentação do conjunto de dados construído para identificar padrões
probabilísticos de atraso das linhas de ônibus de São Paulo e as regiões mais
impactadas.

---

## 1. Identificação do conjunto

| Campo | Valor |
|---|---|
| Título | Previsão × realizado do transporte público de São Paulo |
| Domínio | Mobilidade urbana; sistemas de localização automática de veículos (AVL) |
| Unidade de observação | Par *(previsão de chegada, chegada observada)* de um veículo em uma parada |
| Cobertura espacial | Município de São Paulo, Brasil |
| Cobertura temporal | 7 dias consecutivos (janela contínua, 24 h/dia) |
| Resolução temporal | 30 s (posições), 60 s (previsões) |
| Sistema de referência | EPSG:4326 na coleta; EPSG:31983 (SIRGAS 2000 / UTM 23S) nas operações métricas |
| Formatos | Apache Parquet (zstd) e MongoDB (coleções de série temporal + índices `2dsphere`) |

---

## 2. Proveniência

O conjunto combina três fontes independentes. Nenhuma delas publica o horário
realizado de chegada — essa é a variável que o trabalho reconstrói.

| Fonte | Natureza | Papel | Acesso |
|---|---|---|---|
| API Olho Vivo v2.1 (SPTrans) | Dinâmica, tempo real | Posições de veículos e previsões de chegada | Token individual, cadastro gratuito |
| GTFS estático (SPTrans) | Estática, regerada diariamente | Traçado das linhas (`shapes.txt`) e sequência de paradas (`stop_times.txt`) | Público, sem autenticação |
| Catálogo derivado | Construída | Vínculo linha–parada com ordem corrigida e abscissa curvilínea | Produzido pelo pipeline |

**Instrumento primário.** A API [SPTrans 2026] expõe a posição de cada veículo através do
equipamento AVL embarcado, com carimbo de tempo próprio (`ta`), e a previsão de
chegada calculada pelo sistema da SPTrans. A previsão é o objeto de avaliação;
a posição é o instrumento de medida do realizado.

**Registro de proveniência.** Cada download do GTFS armazena SHA-256, tamanho e
cabeçalho `Last-Modified`, permitindo identificar exatamente qual versão do
itinerário sustentou cada execução da análise.

---

## 3. Delineamento amostral

### 3.1 População e amostra

A população é o conjunto de pares *(previsão, chegada)* gerados pela operação
do sistema. A amostragem opera em duas dimensões:

- **Espacial/operacional.** As posições são coletadas para a **totalidade** da
  frota em circulação (~7.600 a 10.900 veículos por ciclo, ~2.000 códigos de
  linha). As previsões são coletadas para um subconjunto de linhas monitoradas,
  selecionado por critério mensurável (§3.2).
- **Temporal.** Amostragem sistemática de intervalo fixo, não aleatória. O
  intervalo de 30 s para posições foi escolhido por ser inferior ao período de
  atualização do AVL embarcado (mediana observada de 44 s entre carimbos `ta`
  distintos), garantindo que nenhuma atualização seja perdida.

### 3.2 Critério de seleção das linhas monitoradas

A seleção não é arbitrária. Mediu-se, para as 150 linhas de maior frota, a
razão entre paradas com previsão disponível na API e paradas do itinerário
segundo o GTFS:

$$\text{cobertura}(\ell) = \frac{|P_{\text{API}}(\ell)|}{|P_{\text{GTFS}}(\ell)|}$$

A mediana observada foi de **0,029** — a maioria das linhas não possui previsão
publicada em parcela relevante do itinerário. Foram selecionadas linhas com
cobertura ≥ 0,40, no mínimo 12 paradas com previsão e frota ≥ 5 veículos,
ordenadas por *rendimento* (frota × paradas com previsão). O procedimento é
reprodutível por `python -m olhovivo linhas-candidatas`.

**Implicação para inferência.** As linhas monitoradas não constituem amostra
aleatória da rede. Conclusões sobre erro de previsão são válidas
condicionalmente ao subconjunto de linhas com infraestrutura de previsão
consolidada, e não devem ser extrapoladas para a rede inteira sem ressalva.

### 3.3 Ausências e lacunas

Cada arquivo Parquet gravado registra sua janela temporal em um manifesto
(`_manifesto.jsonl`). Intervalos sem cobertura são detectáveis e quantificáveis
*a posteriori* — condição necessária para distinguir "não houve ônibus" de "não
houve coleta", distinção que, se ignorada, produz viés silencioso.

---

## 4. Arquitetura do conjunto

Três camadas, com dependência unidirecional. As camadas derivadas são
integralmente reconstrutíveis a partir da bruta.

```
CATÁLOGO (estático)          FATOS (bruto, append-only)      DERIVADAS
─────────────────────        ──────────────────────────      ─────────────────────
linhas        ──┐            posicoes  ─────┐                chegadas
paradas       ──┼── linha_parada             ├─────────────► previsao_realizado
tracados      ──┘  (ordem, s_m)  previsoes ──┘               regioes
```

### 4.1 Chaves

| Chave | Domínio | Estabilidade | Observação |
|---|---|---|---|
| `cl` | Inteiro | **Instável** | Código de linha **por sentido**; opaco, pode mudar entre execuções. Não usar como chave de longo prazo |
| `letreiro_completo` | `NNNN-DD` | Estável | Chave canônica da linha; equivale a `route_id` do GTFS |
| `cp` | Inteiro | Semi-estável | Código de parada; renumerado entre versões do feed. Casou 100 % por igualdade com `stop_id` na versão corrente |
| `prefixo` | String | Estável no período | Número de frota; permite rastrear o veículo ao longo da viagem |
| `viagem_id` | `cl-prefixo-ISO8601` | Derivada | Identifica uma viagem contínua |

---

## 5. Dicionário de variáveis

Notação: **N** numérica, **C** categórica, **T** temporal, **G** geoespacial, **B** booleana.

### 5.1 `posicoes` — fato bruto (11 variáveis)

| Variável | Tipo | Unidade / domínio | Descrição |
|---|---|---|---|
| `ts_coleta` | T | UTC, ms | Instante da requisição à API. Define a partição `dt`/`h` |
| `ta` | T | UTC, s | Carimbo do GPS embarcado. **Variável temporal de referência** para trajetórias; independente de `ts_coleta` |
| `hr_api` | C | `HH:MM`, America/São_Paulo | Hora de referência declarada pela API |
| `cl` | C | ℤ⁺ | Código da linha por sentido |
| `letreiro` | C | — | Designação pública da linha |
| `sentido` | C | {1, 2} | 1 = terminal principal → secundário; 2 = inverso |
| `destino` | C | — | Letreiro de destino |
| `prefixo` | C | — | Identificador do veículo na frota |
| `acessivel` | B | {V, F} | Acessibilidade para pessoa com deficiência |
| `lat`, `lon` | G | graus decimais (EPSG:4326) | Posição do veículo |

### 5.2 `previsoes` — fato bruto (17 variáveis)

| Variável | Tipo | Unidade / domínio | Descrição |
|---|---|---|---|
| `ts_coleta` | T | UTC, ms | Instante da consulta — define a antecedência da previsão |
| `t_previsto` | T | UTC, ms | **Horário prometido**, convertido para instante absoluto |
| `t_previsto_str` | C | `HH:MM` | Valor bruto, preservado para auditoria da conversão |
| `horizonte_s` | N | segundos, ℤ⁺ | `t_previsto − ts_coleta`. **Variável de controle obrigatória** (§7.1) |
| `cp` | C | ℤ⁺ | Código da parada |
| `parada_nome` | C | — | Denominação; devolvida **vazia** pela API, preenchida via GTFS |
| `parada_lat`, `parada_lon` | G | graus decimais | Posição da parada |
| `lat`, `lon` | G | graus decimais | Posição do veículo ao qual a previsão se refere |
| `ta` | T | UTC, s | Carimbo do GPS desse veículo |
| `cl`, `letreiro`, `sentido`, `prefixo`, `acessivel`, `hr_api` | — | — | Como em `posicoes` |

### 5.3 `linha_parada` — catálogo derivado (7 variáveis)

| Variável | Tipo | Unidade | Descrição |
|---|---|---|---|
| `cl`, `cp` | C | — | Par linha–parada (chave composta) |
| `ordem` | N | ℤ⁺ | **Ordem real do itinerário**, obtida por projeção no traçado |
| `ordem_api` | N | ℤ⁺ ∪ {∅} | Índice no array devolvido pela API. **Não corresponde à ordem do itinerário**; preservado apenas para comparação |
| `s_m` | N | metros | Abscissa curvilínea da parada sobre o traçado (§6.1) |
| `dist_tracado_m` | N | metros | Afastamento perpendicular da parada em relação ao traçado; critério de exclusão (> 120 m) |

### 5.4 `chegadas` — derivada (16 variáveis)

| Variável | Tipo | Unidade | Descrição |
|---|---|---|---|
| `t_chegada` | T | UTC, ms | **Horário realizado**, estimado por interpolação (§6.2) |
| `viagem_id` | C | — | Identificador da viagem contínua |
| `metodo` | C | {cruzamento, aproximacao} | Mecanismo de detecção; `cruzamento` é o caso limpo |
| `confianca` | N | [0, 1] | Escore de qualidade do evento (§6.5) |
| `dist_min_m` | N | metros | Distância do ponto interpolado à parada |
| `salto_m`, `salto_s` | N | metros, segundos | Vão entre as duas amostras usadas na interpolação |
| `velocidade_kmh` | N | km/h | Velocidade implícita no trecho; > 80 indica descontinuidade de GPS |
| `ordem` | N | ℤ⁺ | Posição da parada no itinerário |
| `cl`, `letreiro`, `sentido`, `prefixo`, `cp`, `lat`, `lon` | — | — | Identificação e geometria |

### 5.5 `previsao_realizado` — conjunto analítico final (21 variáveis)

| Variável | Tipo | Unidade | Papel |
|---|---|---|---|
| `erro_s` | N | segundos, ℤ | **Variável resposta** (§6.3) |
| `erro_abs_s` | N | segundos, ℤ⁺ | Módulo do erro, para MAE |
| `horizonte_s` | N | segundos | **Covariável de controle** |
| `headway_obs_s` | N | segundos | Intervalo até o veículo anterior da linha na parada (§6.4) |
| `celula` | C | índice H3 res. 9 (~174 m) [Uber Technologies 2018] | Unidade de agregação espacial |
| `dia_semana` | C | {0…6}, 0 = segunda | Recorte temporal (hora local) |
| `hora` | C | {0…23} | Recorte temporal (hora local) |
| `faixa` | C | {madrugada, pico_manha, entrepico, pico_tarde, noite} | Estrato operacional |
| `confianca` | N | [0, 1] | Critério de inclusão; padrão ≥ 0,35 |
| `ts_coleta`, `t_previsto`, `t_chegada` | T | UTC | Os três instantes que definem a observação |
| `cl`, `letreiro`, `sentido`, `cp`, `parada_nome`, `prefixo`, `viagem_id`, `lat`, `lon` | — | — | Identificação e geometria |

### 5.6 `regioes` — derivada

Saída do agrupamento de §6.7. Uma linha por agrupamento encontrado, por método.

| Variável | Tipo | Unidade / domínio | Descrição |
|---|---|---|---|
| `algoritmo` | C | {kmeans, dbscan, hdbscan, st-dbscan} | Método que produziu o agrupamento |
| `rotulo` | C | ℤ; −1 = ruído | O rótulo −1 só ocorre nos métodos por densidade |
| `n_amostras` | N | ℤ⁺ | Eventos atribuídos ao agrupamento |
| `erro_medio_s`, `erro_mediano_s`, `erro_p90_s` | N | segundos | Estatísticas do atraso no agrupamento |
| `lat`, `lon` | G | graus decimais | Centroide |
| `raio_m` | N | metros | Percentil 90 da distância ao centroide |
| `geometry` | G | GeoJSON Point | Centroide, indexado com `2dsphere` |
| `envoltoria` | G | GeoJSON Polygon | Casco convexo do agrupamento |
| `paradas`, `celulas`, `linhas` | C | listas | Composição do agrupamento |
| `inicio`, `fim`, `horas_predominantes` | T, C | UTC, hora local | Só nos métodos com dimensão temporal |

---

## 6. Formalização das variáveis derivadas

### 6.1 Abscissa curvilínea

Seja $\Gamma$ a polilinha do itinerário da linha, projetada em coordenadas
métricas, e $\pi(\cdot)$ a projeção ortogonal sobre $\Gamma$. Para um ponto
$p$, define-se a abscissa curvilínea $s(p)$ como o comprimento de arco de
$\Gamma$ entre sua origem e $\pi(p)$.

A projeção de uma trajetória é encadeada: a projeção da amostra $i$ é
restrita à janela $[s_{i-1} - 800,\; s_{i-1} + 8000]$ metros. Essa restrição
resolve a ambiguidade de itinerários que percorrem duas vezes o mesmo
logradouro — situação em que a projeção global escolheria arbitrariamente
entre ida e volta.

A operação é um **referenciamento linear**: a posição do veículo é expressa como
medida de comprimento ao longo de um traçado conhecido, e não como resultado de um
*map matching* sobre a malha viária — o itinerário já é dado pelo `shapes.txt`.
Essa é a representação adotada desde o "Tracker" de [Cathey and Dailey 2003], no
qual a posição AVL é expressa como distância linear desde o início da rota, e
reaplicada por [Wessel et al. 2017] e por [Braga et al. 2023] sobre dados
brasileiros. O encadeamento da janela de busca cumpre, de forma determinística, o
papel que o termo de transição do modelo oculto de Markov cumpre em
[Newson and Krumm 2009]: impedir que amostras sucessivas sejam projetadas de forma
independente e incoerente quando a geometria é ambígua e a amostragem, esparsa. O
HMM só se justificaria para decidir entre variantes de traçado da mesma linha, o
que não é o caso aqui.

### 6.2 Evento de chegada

Detecção por raio é inadequada sob a amostragem disponível: com intervalo
mediano de 44 s e velocidade típica de 30 km/h, o deslocamento entre amostras
consecutivas é da ordem de 250–500 m, de modo que um buffer de 50 m em torno da
parada é frequentemente **transposto sem amostragem**.

Adota-se, portanto, a travessia da abscissa. Sejam $t_i, t_{i+1}$ amostras
consecutivas de uma viagem com $s(t_i) \le s_k \le s(t_{i+1})$, onde $s_k$ é a
abscissa da parada $k$. O horário de chegada é estimado por interpolação
linear:

$$\hat{t}_k = t_i + (t_{i+1} - t_i)\cdot\frac{s_k - s(t_i)}{s(t_{i+1}) - s(t_i)}$$

**Boa definição.** Quando $s(t_{i+1}) \le s(t_i)$ — veículo parado, situação
frequente, já que se amostra a cada 30 s e o AVL atualiza a cada ~44 s — a razão
interpolante é indefinida. A implementação atribui $f = 0$ nesse caso e satura
$f$ em $[0,1]$, o que garante $\hat{t}_k \in [t_i, t_{i+1}]$ sem exceção. Recuos
de abscissa superiores ao limiar de segmentação não são corrigidos: são
interpretados como início de nova viagem.

**Hipótese e viés.** A interpolação supõe velocidade constante no trecho. A
hipótese é violada exatamente na parada, que é onde o veículo desacelera, abre
portas e reacelera. Seja $L$ o tempo perdido no evento de parada contido no
intervalo e $w = (s_k - s(t_i))/(s(t_{i+1}) - s(t_i))$. Como
$t_{i+1} - t_i = (s(t_{i+1}) - s(t_i))/v + L$, segue

$$\hat{t}_k - t_k^{\text{real}} = L\,w \;\ge\; 0,$$

ou seja, **o estimador nunca antecipa a chegada** e, em média ($E[w] = 1/2$),
atrasa-a em $L/2$. Pelo TCQSM [Kittelson &amp; Associates et al. 2013], $L$ soma
cerca de 10 s de aceleração e desaceleração a um tempo de parada de referência de
15 s (ponto típico de periferia), 30 s (ponto principal) ou 60 s (centro,
terminal ou polo de transferência), de onde se estima viés esperado de +12 s a
+35 s e limite superior da ordem de +70 s. Formalmente, $\hat{t}_k$ não estima a
chegada, mas um instante situado entre a chegada e a partida do veículo — tanto
mais próximo da partida quanto maior $w$. A consequência para a comparação entre
regiões está registrada em §7.7.

**Segmentação em viagens.** Nova viagem quando o intervalo entre amostras
excede 900 s ou quando a abscissa retrocede mais de 1.500 m.

**Critérios de exclusão.** Descartam-se eventos com salto superior a 1.200 m ou
300 s entre as amostras interpolantes, velocidade implícita acima de 80 km/h, ou
distância do ponto interpolado à parada superior a três vezes o raio de
aceitação. Amostras a mais de 150 m do traçado são removidas antes da
segmentação — filtro que elimina o modo de falha dominante do AVL paulistano
(veículo em garagem com equipamento ativo). Esses limiares são operacionais,
calibrados sobre a própria base, e não derivados de norma.

### 6.3 Erro de previsão

$$\varepsilon = \hat{t}_k - t_{\text{previsto}}$$

com $\varepsilon > 0$ indicando chegada posterior ao prometido. O pareamento
entre previsão e chegada é feito por junção temporal assimétrica (`merge_asof`,
direção *forward*) sobre a chave $(cl, cp, prefixo)$: associa-se a cada previsão
a **primeira** chegada daquele veículo naquela parada em instante igual ou
posterior à consulta, com tolerância retroativa de 60 s e horizonte máximo de
3.600 s.

**Quantização do horário previsto.** A API publica o horário previsto como cadeia
`HH:MM`, isto é, com passo $w = 60$ s; a documentação da SPTrans não declara se o
valor é truncado ou arredondado. As duas convenções produzem o mesmo $\sigma_q$ e
diferem apenas na média, o que as torna distinguíveis nos próprios dados: no
horizonte de 0–2 min mediu-se erro **médio de +30,2 s** ($n = 2.526$), compatível
com truncamento (previsão: +30 s) e incompatível com arredondamento (previsão:
0 s). Adota-se, portanto, o modelo de truncamento.

Como o valor publicado é $p = p^* - u$, com $u \sim \mathcal{U}[0,w)$ e $u \ge 0$,
o erro medido é $\varepsilon = \hat t_k - p = (\hat t_k - p^*) + u$: o truncamento
antecipa a promessa e o componente de quantização entra em $\varepsilon$ com
**sinal positivo**, inflando o atraso aparente. Admite-se $u$ independente do erro
latente. Daí a componente sistemática de $+w/2 = 30$ s e o desvio-padrão de
quantização $\sigma_q = w/\sqrt{12} \approx 17{,}3$ s — o resultado padrão para a
resolução de uma indicação digital [JCGM 2008, §F.2.2.1]. O modelo de erro
uniforme requer que a densidade da grandeza latente seja suave na escala do passo
[Widrow et al. 1996], condição satisfeita aqui.

As variâncias somam em quadratura, $\sigma_\varepsilon = \sqrt{\sigma_e^2 + \sigma_q^2}$.
O deslocamento da **mediana** coincide com $+30$ s apenas sob erro latente
simétrico; com a assimetria à direita que a tabela de §7.1 exibe, o deslocamento
é maior. Conclui-se que medianas dessa ordem não constituem evidência de atraso,
e não que $+30$ s seja um limite irredutível: o piso informacional de um passo de
60 s é $\mathrm{MAE} = 15$ s.

### 6.4 Regularidade (headway)

$$h_{j} = t_{j} - t_{j-1}$$

entre chegadas consecutivas de veículos da mesma linha na mesma parada. Em
serviço de alta frequência — que o TCQSM caracteriza por *headway* médio
programado de 10 min ou menos, faixa em que o passageiro chega à parada sem
consultar horário — a regularidade é indicador de qualidade mais pertinente que a
pontualidade, e a **aderência de headway**, medida pelo coeficiente de variação
$c_{vh} = \sigma_h/\bar{h}$, é a estatística de referência para o efeito de
comboio (*bunching*) [Kittelson &amp; Associates et al. 2013].

| $c_{vh}$ | $P(\lvert h_i-\bar h\rvert > 0{,}5\,\bar h)$ | Interpretação |
|---|---|---|
| 0,00–0,21 | ~0 % | serviço regular como um relógio |
| 0,22–0,30 | ≤ 10 % | veículos levemente fora do headway |
| 0,31–0,39 | ≤ 20 % | veículos frequentemente fora do headway |
| 0,40–0,52 | ≤ 33 % | headways irregulares, com algum comboio |
| 0,53–0,74 | ≤ 50 % | comboio frequente |
| ≥ 0,75 | > 50 % | maioria dos veículos em comboio |

As faixas são as do TCQSM 3ª ed. (Exhibit 5-22). A 2ª edição (TCRP Report 100)
rotulava estas mesmas seis faixas como níveis de serviço A–F, **rótulos
eliminados na 3ª edição** — não devem ser apresentados como LOS ao citar o
TCRP 165.

**Espera do passageiro.** Sob chegada aleatória — hipótese válida na mesma faixa
de headway ≤ 10 min — o tempo médio de espera não é $\bar h/2$, mas

$$\mathrm{AWT} = \tfrac{1}{2}\bar h\,(1 + c_{vh}^2) = \frac{\sum_i h_i^2}{2\sum_i h_i}$$

[Furth et al. 2006, Eq. 1]. As duas formas são algebricamente idênticas, de modo
que a irregularidade medida por $c_{vh}$ é exatamente o que separa a espera
efetiva da espera ideal. O *excess wait time* é a diferença entre a espera
efetiva e a que decorreria do headway programado.

### 6.5 Escore de confiança

$$c = c_1 \cdot c_2 \cdot c_3 \cdot m, \qquad
c_1 = \min\!\left(1, \frac{\Delta s_{\max}}{\Delta s}\right),\;
c_2 = \min\!\left(1, \frac{\Delta t_{\max}}{\Delta t}\right),\;
c_3 = \min\!\left(1, \frac{r}{d}\right)$$

com $\Delta s_{\max} = 500$ m, $\Delta t_{\max} = 120$ s, raio de aceitação
$r = 90$ m, $d$ a distância do ponto interpolado à parada, e $m = 1$ para
detecção por cruzamento, $m = 0{,}7$ por aproximação. O escore degrada
continuamente com a incerteza da interpolação, permitindo análise de
sensibilidade por limiar em vez de exclusão binária. **É construção própria deste
trabalho, sem precedente na literatura**, e os pesos não decorrem de estimação:
servem para ordenar eventos por qualidade, não para quantificar probabilidade de
acerto.

### 6.6 Vizinhança espaço-temporal (DBSCAN com restrição temporal)

Para o agrupamento sensível ao tempo, a vizinhança de um evento $i$ é a
**interseção** de dois testes de densidade independentes:

$$N(i) = \{\, j : \|x_i - x_j\|_2 \le \varepsilon_1 \;\wedge\; |t_i - t_j| \le \varepsilon_2 \,\}$$

com $\varepsilon_1 = 400$ m e $\varepsilon_2 = 1.800$ s. Um evento é **ponto
núcleo** se $|N(i)| \ge \texttt{MinPts}$; pontos não núcleo alcançáveis por
densidade a partir de um núcleo são **pontos de borda**; os demais são ruído. Um
ponto marcado como ruído é reclassificado como borda se for alcançado por
densidade mais tarde. Um ponto de borda disputado por dois agrupamentos é
atribuído ao **descoberto primeiro**, o que torna o resultado dependente da ordem
de varredura [Ester et al. 1996].

A vizinhança é um cilindro no espaço-tempo — disco de raio $\varepsilon_1$ em
$(x,y)$ extrudado por $\pm\varepsilon_2$ em $t$. A alternativa seria uma métrica
combinada $\sqrt{\|x_i-x_j\|_2^2 + (s\,|t_i-t_j|)^2} \le R$, que exige fixar um
fator de escala $s$ convertendo segundos em metros — espaço e tempo não são
comensuráveis sem essa conversão — e que, por construção, permite **substituir**
distância espacial por proximidade temporal. Rejeita-se essa substituição por
razão de domínio: dois registros separados por 400 m e 30 min são passagens
distintas do mesmo trecho, não o mesmo evento. Todas as operações usam
coordenadas projetadas em metros; em graus, $\varepsilon_1$ seria anisotrópico
(em São Paulo, 1° de longitude ≈ 102 km contra 1° de latitude ≈ 111 km).

**Relação com o ST-DBSCAN.** [Birant and Kut 2007] propõem três extensões ao
DBSCAN: (i) vizinhança como interseção de $\varepsilon_1$ (espacial) e
$\varepsilon_2$ (**não espacial** — no exemplo do próprio artigo, temperaturas),
com o tempo entrando por um pré-filtro de vizinhos temporais; (ii) um *density
factor* por agrupamento, para detectar ruído quando coexistem densidades
diferentes; e (iii) o parâmetro $\Delta\varepsilon$, que impede anexar ao
agrupamento um objeto cujo valor não espacial se afaste da média em mais de
$\Delta\varepsilon$, evitando a fusão de agrupamentos adjacentes. **Implementa-se
aqui apenas a estrutura de dupla restrição, com $\varepsilon_2$ reinterpretado
como janela temporal [Cakmak et al. 2021]; não se implementam (ii) nem (iii).**
O método é, portanto, DBSCAN com restrição temporal, e não o ST-DBSCAN de
[Birant and Kut 2007] na íntegra.

### 6.7 Agrupamento espacial das regiões

Quatro métodos são executados sobre o mesmo conjunto de eventos: K-means
[MacQueen 1967], DBSCAN [Ester et al. 1996], HDBSCAN [Campello et al. 2013] e a
variante com restrição temporal de §6.6. O K-means pondera cada amostra pela
magnitude do atraso, de modo que os centroides são atraídos por onde o problema é
maior, e não por onde há mais observações.

---

## 7. Qualidade dos dados, vieses e limitações

### 7.1 Confundimento pelo horizonte de previsão

O erro cresce monotonicamente com a antecedência. Medição em coleta piloto
(*n* = 17.264):

| Horizonte | *n* | MAE (s) | p90 (s) | P(ε ≥ 300 s) |
|---|---|---|---|---|
| 0–2 min | 2.526 | 37,8 | 77 | 0,0000 |
| 2–5 min | 4.240 | 49,7 | 101 | 0,0005 |
| 5–10 min | 5.248 | 70,4 | 142 | 0,0034 |
| 10–20 min | 4.709 | 89,7 | 180 | 0,0130 |

Como a distribuição de horizontes varia entre linhas e faixas horárias, **toda
comparação agregada deve estratificar ou controlar por `horizonte_s`**. Métricas
marginais são não interpretáveis.

### 7.2 Ausência de linha de base programada

A API não publica tabela de horários, e o GTFS da SPTrans é baseado em
`frequencies.txt` [MobilityData 2024] — os horários de `stop_times.txt` constituem modelo de tempo
de percurso, não grade horária efetiva. Consequentemente, **não existe neste
conjunto a medida "atraso contra o programado"**. A ausência de grade horária efetiva impede aplicar as medidas baseadas em
programação do TCQSM — *on-time performance* e *headway adherence*
[Kittelson &amp; Associates et al. 2013]. As três medidas disponíveis
são: erro de previsão (§6.3), regularidade (§6.4) e desvio de tempo de percurso
contra a mediana do mesmo trecho na mesma faixa horária.

### 7.3 Variável resposta inferida

O horário realizado não é observado diretamente; é estimado (§6.2). Validação
por simulação com horários conhecidos, sobre geometria real de linha da SPTrans
e ruído de GPS de 8 m: erro mediano de 0,7 s, p90 de 1,6 s, com 38 de 38 eventos
recuperados. Esses valores caracterizam o **algoritmo sob condições
controladas** e constituem piso, não estimativa do erro em campo, onde
obstrução de sinal e falha de transmissão degradam a medida.

### 7.4 Seleção não aleatória de unidades espaciais

Conforme §3.2, apenas paradas com previsão publicada podem contribuir com
observações. Essas paradas concentram-se em corredores de maior demanda,
introduzindo seleção espacial que deve ser declarada em qualquer inferência
sobre "a cidade".

### 7.5 Cobertura temporal restrita

Sete dias não caracterizam sazonalidade. Feriados, eventos climáticos extremos,
greves e obras podem dominar a janela. O conjunto registra indicador de feriado;
variáveis meteorológicas não são coletadas e constituem extensão natural.

### 7.6 Truncamento por condição operacional

A previsão só existe para veículo já em circulação. O conjunto é, portanto,
condicionado ao evento "veículo em operação", excluindo por construção as
falhas mais severas de oferta — viagem não realizada, veículo que não deixou a
garagem. Isso **subestima** o atraso experimentado pelo usuário.

### 7.7 Viés de interpolação correlacionado com a demanda

O viés do estimador de chegada (§6.2) vale $L\,w$, em que $L$ é o tempo perdido no
evento de parada. Como $L$ cresce com o volume de embarque — o TCQSM
[Kittelson &amp; Associates et al. 2013] sugere 15 s em ponto típico de periferia
contra 60 s em ponto central, terminal ou polo de transferência —, o viés é cerca
de quatro vezes maior nas áreas centrais e de maior demanda.

Trata-se de confundimento direto do objetivo de identificar as regiões mais
impactadas: **parte do gradiente espacial de atraso observado é artefato do
estimador, e não da operação**. Recomenda-se (i) tratar como exploratória toda
comparação entre regiões de densidade de demanda distinta e (ii) estratificar por
tipologia de parada antes de interpretar os agrupamentos de §6.6 e §6.7.

---

## 8. Reprodutibilidade

| Elemento | Mecanismo |
|---|---|
| Versão do itinerário | SHA-256 e `Last-Modified` do GTFS registrados a cada download |
| Janela de coleta | `estado_coleta.json` fixa o instante inicial; reinícios retomam a mesma janela |
| Lacunas | Manifesto por arquivo, com janela temporal; log estruturado de eventos |
| Fuso do horário previsto | Não documentado pela fonte; determinado empiricamente a cada execução e registrado |
| Verificação do pipeline | `scripts/autoteste.py` — cenários com horário de chegada conhecido, incluindo geometria real |
| Ambiente | `requirements.txt` com versões mínimas dos pacotes |

---

## 9. Licenciamento e considerações éticas

Os dados originam-se de API pública mediante cadastro. Os termos da SPTrans
vedam sublicenciamento e comercialização; o uso aqui é de pesquisa. O GTFS é
distribuído publicamente pela mesma operadora.

O conjunto **não contém dados pessoais**: registra veículos identificados por
número de frota, não passageiros. O prefixo permite rastrear um veículo, e
indiretamente turnos de trabalho de motoristas não identificados — recomenda-se
que publicações agreguem por linha, parada ou região, sem exposição de
trajetórias individuais de prefixo.

---

## 10. Referências

- Birant, D. and Kut, A. (2007). ST-DBSCAN: an algorithm for clustering spatial-temporal data. *Data & Knowledge Engineering*, 60(1):208-221.
- Braga, C. K. V., Loureiro, C. F. G., and Pereira, R. H. M. (2023). Evaluating the impact of public transport travel time inaccuracy and variability on socio-spatial inequalities in accessibility. *Journal of Transport Geography*, 109:103590.
- Cakmak, E., Plank, M., Calovi, D. S., Jordan, A., and Keim, D. (2021). Spatio-temporal clustering benchmark for collective animal behavior. In *Proceedings of the 1st ACM SIGSPATIAL International Workshop on Animal Movement Ecology and Human Mobility (HANIMOB '21)*, pages 5-8.
- Campello, R. J. G. B., Moulavi, D., and Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2013)*, volume 7819 of LNCS, pages 160-172. Springer.
- Cathey, F. W. and Dailey, D. J. (2003). A prescription for transit arrival/departure prediction using automatic vehicle location data. *Transportation Research Part C*, 11(3-4):241-264.
- Ester, M., Kriegel, H.-P., Sander, J., and Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD-96)*, pages 226-231.
- Furth, P. G., Hemily, B., Muller, T. H. J., and Strathman, J. G. (2006). Using archived AVL-APC data to improve transit performance and management. TCRP Report 113, Transportation Research Board, Washington, DC.
- JCGM (2008). *Evaluation of measurement data — Guide to the expression of uncertainty in measurement* (JCGM 100:2008). BIPM.
- Kittelson & Associates, Parsons Brinckerhoff, KFH Group, Texas A&M Transportation Institute, and Arup (2013). *Transit Capacity and Quality of Service Manual*. TCRP Report 165, Transportation Research Board, Washington, DC, 3rd edition.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, pages 281-297.
- MobilityData (2024). *General Transit Feed Specification Reference*. https://gtfs.org/documentation/schedule/reference/.
- Newson, P. and Krumm, J. (2009). Hidden Markov map matching through noise and sparseness. In *Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, pages 336-343.
- SPTrans (2026). *API Olho Vivo v2.1 — Guia de referência*. São Paulo Transporte S.A. https://www.sptrans.com.br/desenvolvedores/.
- Uber Technologies (2018). *H3: A Hexagonal Hierarchical Geospatial Indexing System*. https://h3geo.org.
- Wessel, N., Allen, J., and Farber, S. (2017). Constructing a routable retrospective transit timetable from a real-time vehicle location feed and GTFS. *Journal of Transport Geography*, 62:92-103.
- Widrow, B., Kollár, I., and Liu, M.-C. (1996). Statistical theory of quantization. *IEEE Transactions on Instrumentation and Measurement*, 45(2):353-361.
