# 📦 O Peso das Palavras — Análise de Sentimentos em Marketplace Digital

> Investigando a coerência entre notas e comentários de clientes usando técnicas de NLP e modelos de análise de sentimento.

---

## 👥 Autores

| Nome | LinkedIn | Email |
|------|----------|-------|
| Giovanna Coelho Gilio de Sá | [vanna-gilio](https://www.linkedin.com/in/vanna-gilio/) | 07gilio@gmail.com |
| Gustavo Gomes Jacinto | — | gustavo10.ggj@gmail.com |
| Beatriz Boletini Julião | [beatriz-boletini](https://www.linkedin.com/in/beatriz-boletini-95b6b5295/) | boletinibeatriz@gmail.com |
| Enzo Preite Fontes | [enzo-preite](https://www.linkedin.com/in/enzo-preite/) | preiteenzo08@gmail.com |

---

## 📋 Descrição

Em marketplaces digitais, a satisfação do cliente é frequentemente expressa por meio de notas numéricas e comentários textuais realizados após a compra. Embora as notas ofereçam uma medida quantitativa da experiência do consumidor, os comentários possibilitam uma compreensão mais detalhada dos motivos por trás de cada avaliação.

Este projeto realiza uma **análise exploratória de dados** sobre um marketplace digital (dataset da AWS), com foco em investigar a relação entre as notas atribuídas pelos clientes e o conteúdo textual dos seus comentários identificando padrões de satisfação, tendências de comportamento e **possíveis incoerências** entre avaliação numérica e sentimento expresso.

---

## 🎯 Objetivo

Investigar a relação entre as notas atribuídas pelos clientes e o conteúdo textual dos comentários, identificando padrões, tendências e possíveis incoerências por meio de técnicas de análise exploratória de dados e modelos de análise de sentimento.

- Compreender a estrutura das tabelas disponibilizadas
- Realizar limpeza e tratamento dos dados
- Identificar inconsistências e valores ausentes
- Aplicar modelos de análise de sentimento em comentários textuais
- Comparar os resultados obtidos pelos modelos
- Avaliar o grau de coerência entre nota e sentimento identificado
- Identificar padrões de satisfação e insatisfação dos consumidores

---

## 🛠️ Tecnologias e Ferramentas

| Categoria | Ferramentas |
|-----------|-------------|
| **Linguagem** | Python 3 |
| **Manipulação de dados** | Pandas, NumPy |
| **Visualização** | Matplotlib, Seaborn, WordCloud |
| **NLP / Sentimento** | LeIA (leia-br), VADER (vaderSentiment), Transformers (HuggingFace) |
| **Modelos Transformer** | `lxyuan/distilbert-base-multilingual-cased-sentiments-student`, `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
| **Deep Learning** | PyTorch |
| **Otimização** | TQDM (progress bar), Parquet (backup de execuções) |
| **Formatação tabular** | Tabulate |
| **Ambiente** | Google Colab / Jupyter Notebook |
| **Dados** | AWS Marketplace Dataset (9 tabelas CSV) |

---

## 🔬 Metodologia

O projeto foi desenvolvido em quatro etapas principais:

```
1. Carregamento e exploração das 9 tabelas do marketplace
        ↓
2. Limpeza, tipagem e tratamento dos dados
        ↓
3. Criação de variável textual unificada (título + comentário)
        ↓
4. Aplicação e comparação de 4 modelos de análise de sentimento
        ↓
5. Análise exploratória, gráficos e comparação dos 4 modelos.
```

### Modelos Avaliados

| Modelo | Tipo | Idioma | Destaque |
|--------|------|--------|---------|
| **LeIA** | Léxico (VADER adaptado) | Português BR | Boa coerência com PT-BR |
| **VADER** | Léxico | Inglês | Usado como baseline comparativo |
| **Student Sentiments** | Transformer multilíngue | Multilíngue | Melhor resultado em notas 5 (81% positivo) |
| **Twitter RoBERTa** | Transformer (redes sociais) | Multilíngue | Melhor distinção entre extremos; 84% das notas 5 como positivo |

Os sentimentos foram classificados em cinco categorias: **Muito Negativo**, **Negativo**, **Neutro**, **Positivo** e **Muito Positivo**.

---

## 📊 Resultados e Insights

### Base de Dados

- **38.032** avaliações textuais analisadas (após limpeza e remoção de duplicatas)
- **Distribuição das notas**: concentração nos extremos nota 5 (48,4%) e nota 1 (22,3%), totalizando 70,7% dos dados
- Padrão típico de marketplace: usuários comentam principalmente em experiências muito positivas ou muito negativas

### Desempenho dos Modelos

| Modelo | % Positivo em nota 5 | % Negativo em nota 1 | % Neutro geral |
|--------|----------------------|----------------------|----------------|
| LeIA | 77% | 68% | 19,3% |
| Student Sentiments | **81%** | 63% | **6,2%** |
| Twitter RoBERTa | **84%** | 67% | 22,0% |
| VADER | 14% | 16% | **78,6%** |

### Principais Insights

- **VADER** se mostrou inadequado para o português, com 78,6% dos comentários classificados como neutros, sem diferenciar satisfação por nota
- **Twitter RoBERTa** apresentou a transição mais nítida entre notas extremas e a maior capacidade de identificar sentimentos positivos
- **Student Sentiments** alcançou o melhor resultado em notas 5 e apresentou a menor taxa de neutralidade
- **LeIA** manteve desempenho equilibrado, sendo a melhor opção léxica para o português
- **Incoerências identificadas**: mesmo os melhores modelos encontraram casos onde nota e sentimento divergem, fenômeno que não indica apenas erro do modelo, mas também **complexidade real do comportamento do consumidor** (avaliações de aspectos distintos, ironia, ambiguidade)
- A análise de sentimento **complementa** a nota numérica, mas não a substitui

---

## 🗂️ Estrutura do Projeto

```
O-Peso-das-Palavras-Analise-AWS/
│
├── Projeto_O_Peso_das_Palavras_AWS.ipynb   # Notebook principal
│
├── dados/                                  # CSVs do marketplace (não versionados)
│   ├── avaliacoes.csv
│   ├── clientes.csv
│   ├── geolocalizacao.csv
│   ├── itens_pedidos.csv
│   ├── pagamentos.csv
│   ├── pedidos.csv
│   ├── produtos.csv
│   ├── vendedores.csv
│   └── tabela_auxiliar.csv
│
├── backups/                                # Parquets gerados durante execução
│   ├── avaliacoes_completas.parquet
│
└── README.md
```

---

## ⚙️ Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/VannaGilio/O-Peso-das-Palavras-Analise-AWS.git
```

### 2. Imports e Instalar dependências

```bash
import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
import seaborn as sns
import os
```

Execute as células de instalação no início do notebook, ou instale manualmente:

```bash
pip install pandas numpy matplotlib seaborn wordcloud tabulate
pip install leia-br
pip install vaderSentiment
pip install transformers torch tqdm
```

> 💡 **No Google Colab**, utilize `!pip install` diretamente nas células do notebook. O uso de GPU (via `Runtime > Change runtime type > T4 GPU`) é altamente recomendado para os modelos Transformer.

### 3. Adicionar os dados

Coloque os arquivos CSV na mesma pasta do notebook (ou ajuste os caminhos de leitura no código → "dados/exemplo.csv")


### 4. Executar o notebook

Abra `Projeto_O_Peso_das_Palavras_AWS.ipynb` no Google Colab e execute as células em ordem.

> ⚠️ **Atenção:** As seções 9.1 e 9.2 (modelos Transformer) possuem alto custo computacional. Após a primeira execução, os resultados são salvos em `.parquet` nas próximas execuções, carregue apenas a celula de backup(9.1) para economizar tempo.

---

## 📌 Observações Técnicas

- Os modelos Transformer utilizam `truncation=True` e `max_length=512` para textos longos
- A variável `review_text` foi criada pela concatenação de `review_comment_title` e `review_comment_message`, com limpeza de valores nulos e duplicatas
- A geolocalização foi agregada por prefixo de CEP (média de lat/lng + moda de cidade/estado) para funcionar como chave primária
- Strings de cidade/estado foram normalizadas com `unicodedata` para remoção de acentos e padronização de caixa

