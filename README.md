# Checkpoint 1 – Exercício de Análise de Dados

**Aluno:** Diogo Julio Oliveira
**RM:** 553837

## Objetivo

Aplicar os conceitos estudados de Python, NumPy, Pandas, Matplotlib, Seaborn e
pré-processamento de dados sobre um conjunto de dados escolhido.

## Sobre o dataset

Foi utilizado um **conjunto de dados fictício** (`vendas_bicicletas.csv`),
criado especialmente para este exercício, simulando vendas de uma loja de
bicicletas ao longo do ano de 2024. Ele contém 358 registros com as
seguintes colunas:

| Coluna             | Descrição                                              |
|---------------------|---------------------------------------------------------|
| `id_venda`          | Identificador único da venda                            |
| `data`              | Data da venda (2024)                                     |
| `produto`           | Nome do produto vendido (bicicletas, peças, acessórios, vestuário) |
| `categoria`         | Categoria do produto (Bicicletas, Peças, Acessórios, Vestuário) |
| `preco_unitario`    | Preço unitário do produto (R$)                           |
| `quantidade`        | Quantidade vendida                                       |
| `cliente_idade`     | Idade do cliente                                          |
| `cliente_genero`    | Gênero do cliente                                         |
| `cidade`            | Cidade do cliente                                          |
| `forma_pagamento`   | Forma de pagamento utilizada                              |
| `avaliacao`         | Avaliação da compra dada pelo cliente (1 a 5)             |

Os produtos incluem itens como Bike Speed Aro 700, Bike Mountain Aro 29, Bike
Urbana Aro 26, Bike Infantil Aro 20, capacetes, luzes de LED, garrafas
térmicas, bermudas e camisas de ciclismo, kits de reparo, correntes e pedais.

O dataset foi gerado propositalmente com "sujeiras" (valores ausentes,
linhas duplicadas, inconsistências de digitação em nomes de cidade, preços
negativos, quantidades zeradas e um outlier de idade), para permitir a
aplicação de técnicas reais de limpeza e pré-processamento.


## Arquivos entregues

- `analise_vendas.py` — código-fonte principal com toda a análise (carga dos
  dados, informações gerais, identificação de problemas, limpeza,
  visualizações e respostas às questões analíticas).
- `vendas_bicicletas.csv` — conjunto de dados utilizado na análise.
- `README.md` — este arquivo, com as instruções de execução.
- `grafico1_distribuicao_idade.png` a `grafico5_pagamento_boxplot.png` —
  imagens geradas pelo script (também são recriadas automaticamente ao
  executar `analise_vendas.py`).

## Como executar

### 1. Pré-requisitos

- Python 3.9 ou superior instalado.
- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`.

Instale as dependências com:

```bash
pip install pandas numpy matplotlib seaborn
```

### 2. Estrutura de pastas

Mantenha os arquivos `analise_vendas.py` e `vendas_bicicletas.csv` na
mesma pasta (o script lê o CSV pelo caminho relativo).

### 3. Executar a análise

No terminal, dentro da pasta do projeto, execute:

```bash
python analise_vendas.py
```

O script irá, na ordem:

1. Carregar e apresentar o dataset (`.head()`, `.tail()`).
2. Exibir informações gerais (`.info()`), tipos das colunas e estatísticas
   descritivas (`.describe()`).
3. Identificar e imprimir no console os valores ausentes, duplicados e
   inconsistências encontradas (cidades escritas de formas diferentes,
   preços negativos, quantidades zeradas, outlier de idade).
4. Realizar a limpeza e o pré-processamento dos dados (remoção de
   duplicados, padronização de texto, correção de inconsistências,
   tratamento de outliers, imputação de valores ausentes e criação de
   colunas auxiliares).
5. Gerar 5 visualizações, salvas como arquivos `.png` na mesma pasta:
   - `grafico1_distribuicao_idade.png` — histograma da idade dos clientes;
   - `grafico2_faturamento_categoria.png` — faturamento por categoria;
   - `grafico3_faturamento_mensal.png` — evolução do faturamento no ano;
   - `grafico4_correlacao.png` — mapa de calor de correlação;
   - `grafico5_pagamento_boxplot.png` — boxplot do valor da venda por forma
     de pagamento.
