# Análise Epidemiológica de Dengue - Rio de Janeiro (2010-2016)

## Descrição do Projeto

Este projeto realiza análise epidemiológica de dados de dengue no estado do Rio de Janeiro, utilizando técnicas de séries temporais, normalização, cálculo de distâncias e topologia algébrica (complexos simpliciais).

## Estrutura do Projeto

```
dengue/
├── Dengue_Brasil_2010-2016_RJ.xlsx  # Dados brutos
├── requirements.txt                  # Dependências Python
├── README.md                         # Este arquivo
├── src/
│   ├── __init__.py
│   ├── tarefa0_carregar_dados.py     # Carregamento e unificação dos dados
│   ├── tarefa1_semanas_epidemiologicas.py  # Análise de semanas epidemiológicas
│   ├── tarefa2_normalizacao.py       # Normalização por população e área unitária
│   ├── tarefa3_distancias.py         # Cálculo de distâncias L1 e L2
│   ├── tarefa4_complexo_simplicial.py # Criação de complexos simpliciais
│   └── utils.py                      # Funções utilitárias
├── data/                             # Dados processados
│   └── dengue_unificado.csv
└── output/                           # Resultados e gráficos
    ├── matriz_distancia_L1.csv
    └── matriz_distancia_L2.csv
```

## Instalação

1. Clone ou baixe este repositório
2. Crie um ambiente virtual Python:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Tarefas

### Tarefa 0: Carregamento dos Dados
O arquivo Excel contém 7 abas (2010-2016), cada uma com dados de 92 municípios do RJ.

```python
from src.tarefa0_carregar_dados import load_and_merge_dengue_data
df = load_and_merge_dengue_data()
```

### Tarefa 1: Semanas Epidemiológicas
Análise do calendário epidemiológico:
- **2014 tem 53 semanas** nos dados
- 2016 tem apenas 32 semanas (dados incompletos)
- Padronização para 52 semanas combinando semana 53 com semana 52

```python
python src/tarefa1_semanas_epidemiologicas.py
```

### Tarefa 2: Normalização
- **Normalização 1**: Por população do Censo 2010 (taxa de incidência por 100.000 hab)
- **Normalização 2**: Por total de infectados (área unitária)

```python
python src/tarefa2_normalizacao.py
```

### Tarefa 3: Distâncias L1 e L2
Cálculo de matrizes de distância entre séries temporais normalizadas:
- **L1 (Manhattan)**: Σ|x_i - y_i| - mais robusta a outliers
- **L2 (Euclidiana)**: √(Σ(x_i - y_i)²) - penaliza diferenças grandes

```python
python src/tarefa3_distancias.py
```

### Tarefa 4: Complexos Simpliciais
Criação de estruturas topológicas a partir das matrizes de distância:
- Vértices = Municípios
- Arestas = Pares com distância < limiar
- Triângulos = Trios mutuamente conectados

```python
python src/tarefa4_complexo_simplicial.py
```

## Conceitos Chave

### Semanas Epidemiológicas
- Padrão OMS/CDC para contagem de tempo em epidemiologia
- A maioria dos anos tem 52 semanas, alguns têm 53
- No Brasil, o pico de dengue ocorre entre janeiro e abril (semanas 1-16)

### Período Epidêmico (PE)
- Intervalo entre início e fim de uma epidemia
- Geralmente cruza a virada do ano (semana 40 a semana 20)
- Definido quando casos ultrapassam um limiar

### Normalização por Área Unitária
- Séries normalizadas têm soma = 1
- Permite comparar o FORMATO das curvas, não a magnitude
- Municípios com curvas similares são "sincronizados"

### Complexo Simplicial
- Generalização de grafos para dimensões maiores
- Captura relações de grupo (não apenas pares)
- Triângulos indicam trios de municípios com dinâmica similar

## Resultados Esperados

1. **Municípios mais afetados**: Identificados pela taxa de incidência
2. **Municípios sincronizados**: Identificados pela baixa distância L1/L2
3. **Clusters epidemiológicos**: Grupos de municípios no complexo simplicial
4. **Anos com 53 semanas**: 2014

## Autores

Projeto desenvolvido para análise epidemiológica de dengue.

## Licença

MIT License
