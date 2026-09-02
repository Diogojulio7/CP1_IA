# =============================================================================
# Checkpoint 1 - Exercício de Análise de Dados
# Aluno: Diogo Julio Oliveira | RM: 553837
#
# Dataset: vendas_bicicletas.csv (dataset fictício de vendas de uma loja
# de bicicletas, com 12 produtos, 6 cidades, ao longo do ano de 2024).
#
# Bibliotecas: Python, NumPy, Pandas, Matplotlib, Seaborn
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


def titulo(texto):
    print("\n" + "=" * 80)
    print(texto)
    print("=" * 80)


# =============================================================================
# 1. CARREGAMENTO E APRESENTAÇÃO DOS DADOS
# =============================================================================
titulo("1. CARREGAMENTO E APRESENTAÇÃO DOS DADOS")

df = pd.read_csv("vendas_bicicletas.csv")

print(f"\nDimensões do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
print("\nPrimeiras 10 linhas:")
print(df.head(10))

print("\nÚltimas 5 linhas:")
print(df.tail(5))


# =============================================================================
# 2. INFORMAÇÕES GERAIS, TIPOS DAS COLUNAS E ESTATÍSTICAS DESCRITIVAS
# =============================================================================
titulo("2. INFORMAÇÕES GERAIS DO DATASET")

print("\nTipos de dados de cada coluna (df.info()):")
df.info()

print("\nEstatísticas descritivas - colunas numéricas:")
print(df.describe())

print("\nEstatísticas descritivas - colunas categóricas:")
print(df.describe(include="object"))


# =============================================================================
# 3. IDENTIFICAÇÃO DE VALORES AUSENTES, DUPLICADOS E INCONSISTÊNCIAS
# =============================================================================
titulo("3. VALORES AUSENTES, DUPLICADOS E INCONSISTÊNCIAS")

print("\nValores ausentes por coluna:")
print(df.isna().sum())
print(f"\nTotal de valores ausentes no dataset: {df.isna().sum().sum()}")

print(f"\nLinhas totalmente duplicadas: {df.duplicated().sum()}")

print("\nValores únicos na coluna 'cidade' (aqui aparecem as inconsistências "
      "de digitação/formatação, ex: 'São Paulo', 'sao paulo', 'SP', 'São paulo '):")
print(sorted(df["cidade"].unique()))

print("\nValores mínimo e máximo de 'preco_unitario' (verificando preços negativos):")
print(df["preco_unitario"].agg(["min", "max"]))

print("\nValores mínimo e máximo de 'quantidade' (verificando quantidades <= 0):")
print(df["quantidade"].agg(["min", "max"]))

print("\nValores mínimo e máximo de 'cliente_idade' (verificando outliers, ex: idade 199):")
print(df["cliente_idade"].agg(["min", "max"]))

print("\n--- CONCLUSÃO DO DIAGNÓSTICO (inconsistências encontradas) ---")
print("""
- Valores ausentes em: cliente_idade, cliente_genero, forma_pagamento e avaliacao.
- 8 linhas duplicadas.
- Cidade 'São Paulo' e 'Rio de Janeiro' escritas de formas diferentes
  (maiúsculas/minúsculas, siglas e espaços extras).
- Alguns 'preco_unitario' negativos (erro de digitação).
- Algumas 'quantidade' iguais a 0 (venda inválida).
- Outliers de idade (ex: 199 anos), claramente erro de digitação.
""")


# =============================================================================
# 4. LIMPEZA E PRÉ-PROCESSAMENTO
# =============================================================================
titulo("4. LIMPEZA E PRÉ-PROCESSAMENTO")

df_limpo = df.copy()

# 4.1 Remover linhas duplicadas
qtd_antes = len(df_limpo)
df_limpo = df_limpo.drop_duplicates()
print(f"\n[Duplicados] Linhas removidas: {qtd_antes - len(df_limpo)}")

# 4.2 Padronizar nomes de cidade (remover espaços, ajustar caixa e siglas)
mapa_cidades = {
    "sao paulo": "São Paulo",
    "são paulo": "São Paulo",
    "sp": "São Paulo",
    "rio de janeiro": "Rio de Janeiro",
    "rj": "Rio de Janeiro",
}
df_limpo["cidade"] = df_limpo["cidade"].str.strip().str.lower().map(
    lambda x: mapa_cidades.get(x, x.title())
)
print("\n[Cidades] Valores únicos após padronização:")
print(sorted(df_limpo["cidade"].unique()))

# 4.3 Corrigir preços negativos (erro de digitação -> usar valor absoluto)
qtd_precos_neg = (df_limpo["preco_unitario"] < 0).sum()
df_limpo["preco_unitario"] = df_limpo["preco_unitario"].abs()
print(f"\n[Preços] Preços negativos corrigidos (valor absoluto): {qtd_precos_neg}")

# 4.4 Remover vendas com quantidade igual a zero (não representam venda real)
qtd_antes = len(df_limpo)
df_limpo = df_limpo[df_limpo["quantidade"] > 0]
print(f"[Quantidade] Linhas com quantidade <= 0 removidas: {qtd_antes - len(df_limpo)}")

# 4.5 Tratar outliers de idade (ex: 199 anos) -> tratar como ausente e depois imputar
df_limpo.loc[df_limpo["cliente_idade"] > 100, "cliente_idade"] = np.nan

# 4.6 Imputar valores ausentes
# Idade: mediana (robusta a outliers)
mediana_idade = df_limpo["cliente_idade"].median()
df_limpo["cliente_idade"] = df_limpo["cliente_idade"].fillna(mediana_idade)
print(f"\n[Idade] Valores ausentes/outliers preenchidos com a mediana ({mediana_idade:.0f} anos)")

# Gênero e forma de pagamento: categoria "Não informado"
df_limpo["cliente_genero"] = df_limpo["cliente_genero"].fillna("Não informado")
df_limpo["forma_pagamento"] = df_limpo["forma_pagamento"].fillna("Não informado")
print("[Gênero / Pagamento] Valores ausentes preenchidos com 'Não informado'")

# Avaliação: moda (valor mais frequente), pois é uma variável discreta (1 a 5)
moda_avaliacao = df_limpo["avaliacao"].mode()[0]
df_limpo["avaliacao"] = df_limpo["avaliacao"].fillna(moda_avaliacao)
df_limpo["avaliacao"] = df_limpo["avaliacao"].astype(int)
print(f"[Avaliação] Valores ausentes preenchidos com a moda ({moda_avaliacao})")

# 4.7 Ajustar tipos de dados
df_limpo["data"] = pd.to_datetime(df_limpo["data"])
df_limpo["cliente_idade"] = df_limpo["cliente_idade"].astype(int)

# 4.8 Criar coluna derivada: valor total da venda (feature engineering)
df_limpo["valor_total"] = (df_limpo["preco_unitario"] * df_limpo["quantidade"]).round(2)

# 4.9 Criar coluna de mês (útil para análises temporais)
df_limpo["mes"] = df_limpo["data"].dt.month

print(f"\nDataset após limpeza: {df_limpo.shape[0]} linhas x {df_limpo.shape[1]} colunas")
print("\nConferência final - valores ausentes:")
print(df_limpo.isna().sum())
print(f"\nConferência final - duplicados: {df_limpo.duplicated().sum()}")


# =============================================================================
# 5. VISUALIZAÇÕES
# =============================================================================
titulo("5. VISUALIZAÇÕES (salvas como arquivos .png)")

# --- Visualização 1: Histograma da distribuição de idade dos clientes ---
plt.figure(figsize=(8, 5))
sns.histplot(df_limpo["cliente_idade"], bins=15, kde=True, color="#4C72B0")
plt.title("Distribuição da Idade dos Clientes")
plt.xlabel("Idade")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("grafico1_distribuicao_idade.png", dpi=150)
plt.close()
print("\n[OK] grafico1_distribuicao_idade.png salvo")

# --- Visualização 2: Faturamento total por categoria de produto (barras) ---
faturamento_categoria = (
    df_limpo.groupby("categoria")["valor_total"].sum().sort_values(ascending=False)
)
plt.figure(figsize=(8, 5))
sns.barplot(x=faturamento_categoria.values, y=faturamento_categoria.index, palette="viridis")
plt.title("Faturamento Total por Categoria de Produto")
plt.xlabel("Faturamento (R$)")
plt.ylabel("Categoria")
plt.tight_layout()
plt.savefig("grafico2_faturamento_categoria.png", dpi=150)
plt.close()
print("[OK] grafico2_faturamento_categoria.png salvo")

# --- Visualização 3: Evolução do faturamento mensal (linha) ---
faturamento_mensal = df_limpo.groupby("mes")["valor_total"].sum()
plt.figure(figsize=(8, 5))
sns.lineplot(x=faturamento_mensal.index, y=faturamento_mensal.values, marker="o", color="#C44E52")
plt.title("Evolução do Faturamento Mensal (2024)")
plt.xlabel("Mês")
plt.ylabel("Faturamento (R$)")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig("grafico3_faturamento_mensal.png", dpi=150)
plt.close()
print("[OK] grafico3_faturamento_mensal.png salvo")

# --- Visualização 4: Mapa de calor de correlação entre variáveis numéricas ---
colunas_numericas = ["preco_unitario", "quantidade", "cliente_idade", "avaliacao", "valor_total"]
matriz_corr = df_limpo[colunas_numericas].corr()
plt.figure(figsize=(7, 6))
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Mapa de Correlação entre Variáveis Numéricas")
plt.tight_layout()
plt.savefig("grafico4_correlacao.png", dpi=150)
plt.close()
print("[OK] grafico4_correlacao.png salvo")

# --- Visualização 5 (bônus): Boxplot de valor total por forma de pagamento ---
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_limpo, x="forma_pagamento", y="valor_total", palette="Set2")
plt.title("Valor Total da Venda por Forma de Pagamento")
plt.xlabel("Forma de Pagamento")
plt.ylabel("Valor Total (R$)")
plt.tight_layout()
plt.savefig("grafico5_pagamento_boxplot.png", dpi=150)
plt.close()
print("[OK] grafico5_pagamento_boxplot.png salvo")


# =============================================================================
# QUESTÕES ANALÍTICAS
# =============================================================================
titulo("QUESTÕES ANALÍTICAS")

# ----------------------------------------------------------------------------
# 1) Qual é a distribuição das principais variáveis?
# ----------------------------------------------------------------------------
print("""
1) Qual é a distribuição das principais variáveis?

- Idade dos clientes: distribuição aproximadamente uniforme entre 18 e 70 anos
  (ver grafico1_distribuicao_idade.png), sem concentração forte em nenhuma faixa,
  já que os dados foram sorteados de forma equilibrada.
""")
print("Estatísticas de idade:")
print(df_limpo["cliente_idade"].describe())

print("\nDistribuição de vendas por categoria de produto:")
print(df_limpo["categoria"].value_counts())

print("\nDistribuição de vendas por forma de pagamento:")
print(df_limpo["forma_pagamento"].value_counts())

print("\nDistribuição das avaliações (notas de 1 a 5):")
print(df_limpo["avaliacao"].value_counts().sort_index())

# ----------------------------------------------------------------------------
# 2) Existem correlações importantes entre as variáveis?
# ----------------------------------------------------------------------------
print("""
2) Existem correlações importantes entre as variáveis?

Veja grafico4_correlacao.png. Principais pontos:
""")
print(matriz_corr["valor_total"].sort_values(ascending=False))
print("""
- 'valor_total' tem correlação forte e óbvia com 'preco_unitario' e 'quantidade'
  (pois é calculado a partir delas), o que é esperado.
- 'cliente_idade' e 'avaliacao' apresentam correlação muito fraca (próxima de 0)
  com as demais variáveis, indicando que, neste dataset fictício, idade do
  cliente e nota de avaliação não têm relação linear relevante com o valor
  gasto na compra.
""")

# ----------------------------------------------------------------------------
# 3) Quais insights podem ser extraídos dos dados?
# ----------------------------------------------------------------------------
categoria_top = faturamento_categoria.idxmax()
mes_top = faturamento_mensal.idxmax()
pagamento_top = df_limpo["forma_pagamento"].value_counts().idxmax()
cidade_top = df_limpo["cidade"].value_counts().idxmax()

print(f"""
3) Quais insights podem ser extraídos dos dados?

- A categoria com maior faturamento é '{categoria_top}', respondendo por uma
  parcela relevante do total vendido.
- O mês com maior faturamento foi o mês {mes_top}.
- A forma de pagamento mais utilizada pelos clientes é '{pagamento_top}'.
- A cidade com maior número de vendas é '{cidade_top}'.
- A avaliação média dada pelos clientes é de {df_limpo['avaliacao'].mean():.2f}
  (em uma escala de 1 a 5), sugerindo satisfação predominantemente neutra/positiva.
- O ticket médio (valor_total médio por venda) é de R$ {df_limpo['valor_total'].mean():.2f}.
""")

# ----------------------------------------------------------------------------
# 4) Qual pré-processamento foi necessário e por quê?
# ----------------------------------------------------------------------------
print("""
4) Qual pré-processamento foi necessário e por quê?

- Remoção de linhas duplicadas: evitar contagem dupla de vendas e distorção
  de métricas de faturamento.
- Padronização de texto na coluna 'cidade' (maiúsculas/minúsculas, siglas
  como 'SP'/'RJ' e espaços extras): sem isso, a mesma cidade seria contada
  como várias categorias diferentes, prejudicando agrupamentos (groupby).
- Correção de preços negativos: eram erros de digitação; usar valor absoluto
  preserva a informação sem descartar a linha.
- Remoção de vendas com quantidade igual a 0: não representam uma venda real.
- Tratamento de outlier de idade (ex: 199 anos): valor fisicamente impossível,
  tratado como ausente e depois imputado.
- Imputação de valores ausentes:
    * 'cliente_idade' -> mediana (robusta a outliers residuais);
    * 'avaliacao' -> moda (variável discreta, categórica-ordinal);
    * 'cliente_genero' e 'forma_pagamento' -> categoria 'Não informado'
      (evita perder linhas e mantém a informação de que o dado faltou).
- Conversão de tipos: 'data' convertida para datetime e 'cliente_idade' para
  inteiro, permitindo análises temporais e cálculos corretos.
- Criação de colunas derivadas ('valor_total' e 'mes'): necessárias para
  responder às perguntas de faturamento e sazonalidade.
""")


# =============================================================================
# CONCLUSÕES GERAIS
# =============================================================================
titulo("CONCLUSÕES GERAIS")
print(f"""
- O dataset fictício de vendas de bicicletas, após limpeza, ficou com
  {df_limpo.shape[0]} vendas válidas (de {df.shape[0]} linhas originais).
- A categoria '{categoria_top}' se destaca em faturamento, o que pode orientar
  decisões de estoque e campanhas de marketing.
- A forma de pagamento '{pagamento_top}' é predominante, reforçando a
  importância de manter essa opção sempre disponível e sem fricção no
  checkout.
- Não foram encontradas correlações lineares fortes entre idade/avaliação e
  valor gasto, sugerindo que o comportamento de compra não depende
  fortemente do perfil demográfico simulado neste conjunto de dados.
- O pré-processamento (remoção de duplicados, padronização de texto,
  tratamento de outliers e imputação de ausentes) foi essencial para que as
  métricas de faturamento e as visualizações refletissem a realidade dos
  dados, evitando distorções causadas por erros de digitação e formatação
  inconsistente.
""")