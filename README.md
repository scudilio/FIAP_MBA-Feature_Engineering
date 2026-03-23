# 📚 Feature Engineering — Planejamento de Aulas Práticas

## FIAP — MBA em Data Science

---

## Visão Geral da Disciplina

**Abordagem:** Problem-Based Learning — cada aula parte de um cenário de negócio real antes de formalizar conceitos e implementar código.

**Projeto Integrador:** Todas as aulas contribuem para um pipeline completo de Feature Engineering aplicado a um dataset de negócios (crédito bancário), permitindo ao aluno construir um portfólio contínuo ao longo da disciplina.

---

## AULA 1 — Missing Data Imputation

### 🏢 Cenário de Negócio
**Setor:** Varejo / E-commerce  
**Problema:** Uma rede de e-commerce tem um dataset de clientes com informações de compras, dados demográficos e comportamento de navegação. Porém, ~15% dos registros possuem campos ausentes (renda, idade, estado civil, tempo de navegação). A empresa precisa construir um modelo de propensão à compra, mas não pode descartar esses clientes.

**Pergunta-chave:** *"Como tratar dados faltantes sem introduzir viés que prejudique nosso modelo de propensão?"*

### 🎯 Objetivo
O aluno será capaz de identificar padrões de dados ausentes (MCAR, MAR, MNAR), escolher a técnica de imputação adequada e avaliar o impacto de cada abordagem na qualidade do modelo.

### 📖 O que o aluno irá aprender

| Tópico | Descrição |
|--------|-----------|
| Tipos de missing data | MCAR, MAR, MNAR — quando cada um ocorre e suas implicações |
| Diagnóstico visual | Heatmaps de nulidade (msno), matriz de correlação de missingness |
| Imputação simples | Média, mediana, moda — quando usar cada uma |
| Imputação por valor arbitrário | Quando e por que usar valores como 999 ou -1 |
| Imputação avançada | KNN Imputer, Iterative Imputer (MICE) |
| Indicador de missing | Criação de feature binária "dado_era_ausente" |
| Avaliação de impacto | Comparação de distribuições antes/depois da imputação |

### 🛠️ Dataset
Dataset sintético de e-commerce (~10.000 registros) com colunas: `idade`, `renda_mensal`, `estado_civil`, `qtd_compras_mes`, `tempo_navegacao_min`, `ticket_medio`, `canal_preferido`, `comprou` (target).

### 📦 Bibliotecas
`pandas`, `numpy`, `matplotlib`, `seaborn`, `missingno`, `scikit-learn` (SimpleImputer, KNNImputer, IterativeImputer)

### 📋 Entregáveis do Notebook
1. EDA dos dados ausentes com visualizações (heatmap, bar chart de missing)
2. Implementação de 4 técnicas de imputação
3. Comparação visual das distribuições antes/depois
4. Treinamento de um classificador simples com cada abordagem e comparação de métricas

---

## AULA 2 — Feature Scaling (Normalização e Padronização)

### 🏢 Cenário de Negócio
**Setor:** Imobiliário  
**Problema:** Uma imobiliária quer construir um modelo de precificação de imóveis. O dataset contém features com escalas muito diferentes: `area_m2` (50–500), `num_quartos` (1–6), `renda_bairro` (2.000–50.000), `distancia_metro_km` (0.1–30). Modelos baseados em distância (KNN, SVM) e gradient descent (redes neurais) são sensíveis a essas diferenças de escala.

**Pergunta-chave:** *"Qual técnica de scaling devo usar para que features com magnitudes diferentes não dominem o modelo?"*

### 🎯 Objetivo
O aluno será capaz de aplicar diferentes técnicas de scaling, entender quando cada uma é apropriada e demonstrar o impacto no desempenho de algoritmos sensíveis à escala.

### 📖 O que o aluno irá aprender

| Tópico | Descrição |
|--------|-----------|
| Por que escalar? | Impacto em gradient descent, distâncias, regularização |
| StandardScaler | Z-score: (x - μ) / σ — quando a distribuição é ~normal |
| MinMaxScaler | Normalização [0,1] — quando se precisa de range fixo |
| RobustScaler | Usa mediana e IQR — robusto a outliers |
| MaxAbsScaler | Escala por valor absoluto máximo — para dados esparsos |
| Log Transform | Transformação logarítmica para distribuições muito enviesadas |
| Quando NÃO escalar | Modelos baseados em árvore (Random Forest, XGBoost) |
| Pipeline do sklearn | Encapsular scaling + modelo para evitar data leakage |

### 🛠️ Dataset
Dataset sintético de imóveis (~5.000 registros) com colunas: `area_m2`, `num_quartos`, `num_banheiros`, `vagas_garagem`, `renda_media_bairro`, `distancia_metro_km`, `idade_imovel_anos`, `condominio`, `preco` (target).

### 📦 Bibliotecas
`pandas`, `numpy`, `matplotlib`, `scikit-learn` (StandardScaler, MinMaxScaler, RobustScaler, Pipeline, KNeighborsRegressor, LinearRegression)

### 📋 Entregáveis do Notebook
1. Visualização das distribuições originais (histogramas, boxplots)
2. Aplicação de cada scaler com visualização lado a lado
3. Comparação de KNN Regressor com/sem scaling
4. Implementação de Pipeline (scaling + modelo) para evitar data leakage

---

## AULA 3 — Dados Desbalanceados: Downsampling e Upsampling (SMOTE)

### 🏢 Cenário de Negócio
**Setor:** Financeiro  
**Problema:** Um banco precisa detectar transações fraudulentas. De 100.000 transações, apenas 500 (~0.5%) são fraudes. Um modelo treinado ingenuamente alcança 99.5% de acurácia simplesmente prevendo "não fraude" para tudo — mas não detecta nenhuma fraude real.

**Pergunta-chave:** *"Como treinar um modelo que efetivamente detecte fraudes quando 99.5% dos dados são de transações legítimas?"*

### 🎯 Objetivo
O aluno será capaz de diagnosticar desbalanceamento, aplicar técnicas de reamostragem (under/oversampling), utilizar SMOTE para geração sintética e avaliar modelos com métricas adequadas (Precision, Recall, F1, AUC-ROC).

### 📖 O que o aluno irá aprender

| Tópico | Descrição |
|--------|-----------|
| Diagnóstico | Proporção de classes, acurácia paradoxal |
| Downsampling (undersampling) | Random UnderSampler, NearMiss — reduz a classe majoritária |
| Upsampling (oversampling) | Random OverSampler — replica a classe minoritária |
| SMOTE | Geração de amostras sintéticas com interpolação KNN |
| SMOTE variantes | BorderlineSMOTE, ADASYN — quando usar cada um |
| Class Weight | Ajuste de pesos no classificador (class_weight='balanced') |
| Métricas corretas | Precision, Recall, F1-Score, AUC-ROC, Precision-Recall Curve |
| Combinação de técnicas | SMOTETomek, SMOTEENN — oversample + clean |

### 🛠️ Dataset
Dataset sintético de transações bancárias (~50.000 registros, ~1% fraude) com colunas: `valor_transacao`, `hora_dia`, `dia_semana`, `tipo_cartao`, `distancia_residencia_km`, `num_transacoes_24h`, `media_historica_valor`, `pais_diferente`, `fraude` (target).

### 📦 Bibliotecas
`pandas`, `numpy`, `matplotlib`, `seaborn`, `imbalanced-learn` (SMOTE, RandomUnderSampler, RandomOverSampler, SMOTETomek), `scikit-learn` (RandomForestClassifier, classification_report, roc_auc_score, roc_curve)

### 📋 Entregáveis do Notebook
1. EDA do desbalanceamento com gráficos de proporção
2. Modelo baseline sem tratamento (acurácia vs Recall)
3. Implementação de Downsampling, Upsampling e SMOTE
4. Comparação de métricas (F1, AUC-ROC) para cada abordagem
5. Visualização 2D do efeito do SMOTE no espaço de features

---

## AULA 4 — Redução de Dimensionalidade: PCA

### 🏢 Cenário de Negócio
**Setor:** Telecomunicações  
**Problema:** Uma operadora de telecom tem 50+ features sobre cada cliente (uso de dados, ligações, reclamações, plano, etc.) e quer segmentá-los para campanhas de marketing. Porém, muitas features são correlacionadas (ex.: minutos de ligação e valor da fatura). O excesso de dimensões dificulta a visualização e aumenta ruído.

**Pergunta-chave:** *"Como reduzir 50 features para um número menor de componentes que expliquem a maior parte da variância, sem perder informação relevante?"*

### 🎯 Objetivo
O aluno será capaz de aplicar PCA, interpretar variância explicada, escolher o número ideal de componentes e usar os componentes principais como input para modelos de ML.

### 📖 O que o aluno irá aprender

| Tópico | Descrição |
|--------|-----------|
| Maldição da dimensionalidade | Por que muitas features nem sempre ajudam |
| Correlação entre features | Matriz de correlação, multicolinearidade |
| Conceito de PCA | Autovalores, autovetores, projeção linear |
| Variância explicada | Scree plot, variância acumulada, regra do cotovelo |
| Implementação PCA | sklearn PCA, escolha de n_components |
| PCA para visualização | Projeção em 2D/3D para clusters |
| PCA como pré-processamento | Usar componentes como features de um classificador |
| Limitações do PCA | Linearidade, sensibilidade à escala (precisa StandardScaler antes!) |

### 🛠️ Dataset
Dataset sintético de clientes telecom (~8.000 registros, 30 features numéricas) incluindo: `minutos_ligacao`, `qtd_sms`, `dados_mb`, `valor_fatura`, `dias_atraso_pagamento`, `qtd_reclamacoes`, `meses_como_cliente`, `churn` (target), entre outros.

### 📦 Bibliotecas
`pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn` (PCA, StandardScaler, RandomForestClassifier)

### 📋 Entregáveis do Notebook
1. Análise de correlação (heatmap) mostrando redundância
2. Aplicação de PCA com scree plot e variância explicada acumulada
3. Visualização 2D dos clusters (com/sem PCA)
4. Comparação de modelo com features originais vs componentes PCA
5. Interpretação dos loadings (quais features mais contribuem)

---

## AULA 5 — Redução de Dimensionalidade: LDA e T-SNE

### 🏢 Cenário de Negócio
**Setor:** Saúde / Diagnóstico  
**Problema:** Um laboratório possui dados de exames de sangue (30+ biomarcadores) e quer criar um sistema de triagem automática para 4 tipos de diagnóstico. O PCA (não supervisionado) não considera os rótulos. Precisamos de uma técnica que maximize a separação entre as classes.

**Pergunta-chave:** *"Como reduzir dimensões de forma que as classes fiquem o mais separadas possível no novo espaço?"*

### 🎯 Objetivo
O aluno será capaz de diferenciar PCA (não supervisionado) de LDA (supervisionado), aplicar T-SNE para visualização e entender quando usar cada técnica.

### 📖 O que o aluno irá aprender

| Tópico | Descrição |
|--------|-----------|
| PCA vs LDA | Não supervisionado vs supervisionado; variância vs separabilidade |
| LDA — conceito | Maximiza razão between-class / within-class scatter |
| LDA — restrição | Máximo de (n_classes - 1) componentes |
| T-SNE — conceito | Preserva vizinhanças locais em baixa dimensão (não-linear) |
| T-SNE — perplexity | Hiperparâmetro que controla o trade-off local vs global |
| T-SNE — limitações | Não determinístico, lento para datasets grandes, não preserva distâncias globais |
| Comparação visual | PCA vs LDA vs T-SNE no mesmo dataset |
| Quando usar cada um | Checklist de decisão |

### 🛠️ Dataset
Dataset sintético de exames laboratoriais (~5.000 registros, 25 biomarcadores, 4 classes de diagnóstico).

### 📦 Bibliotecas
`pandas`, `numpy`, `matplotlib`, `plotly`, `scikit-learn` (PCA, LinearDiscriminantAnalysis, TSNE)

### 📋 Entregáveis do Notebook
1. Projeção PCA 2D com cores por classe (overlap)
2. Projeção LDA 2D com cores por classe (separação)
3. Projeção T-SNE 2D com diferentes valores de perplexity
4. Comparação lado a lado: PCA vs LDA vs T-SNE
5. Tabela comparativa: vantagens, limitações e casos de uso

---

## AULA 6 — Feature Extraction (Dados Não Estruturados) + Feature Store

### 🏢 Cenário de Negócio
**Setor:** Varejo / Atendimento ao cliente  
**Problema:** Uma empresa recebe milhares de avaliações de produtos (texto livre) e fotos enviadas por clientes. A equipe de dados precisa transformar esses dados não estruturados em features numéricas para alimentar um modelo de satisfação do cliente.

**Pergunta-chave:** *"Como extrair features úteis de texto e imagens para usar em modelos de ML tradicionais?"*

### 🎯 Objetivo
O aluno será capaz de extrair features de dados não estruturados (texto e imagem), entender o conceito de embeddings e conhecer os fundamentos de Feature Store.

### 📖 O que o aluno irá aprender

| Tópico | Descrição |
|--------|-----------|
| Feature Extraction vs Feature Selection | Criar novas features vs selecionar existentes |
| Texto → Números | Bag of Words, TF-IDF |
| Embeddings de texto | Word2Vec, conceito de embeddings pré-treinados |
| Imagem → Números | Histograma de cores, HOG (Histogram of Oriented Gradients) |
| Transfer Learning como Feature Extractor | Usar CNN pré-treinada para extrair features de imagens |
| Feature Store — Conceito | O que é, por que surgiu, problemas que resolve |
| Feature Store — Componentes | Offline store, online store, feature registry |
| Feature Store — Ferramentas | Feast, Tecton, Hopsworks (visão geral) |

### 🛠️ Dataset
Dataset sintético com avaliações de produtos (~3.000 registros): `texto_avaliacao`, `nota_estrelas` (target), `categoria_produto`. Parte prática de imagem opcional com Fashion MNIST.

### 📦 Bibliotecas
`pandas`, `numpy`, `scikit-learn` (TfidfVectorizer, CountVectorizer), `matplotlib`

### 📋 Entregáveis do Notebook
1. Pipeline de extração TF-IDF de textos de avaliação
2. Treinamento de classificador usando features textuais
3. Demonstração conceitual de Feature Store (diagrama + explicação)
4. Discussão: quando investir em um Feature Store

---

## AULA 7 — Feature Selection: Métodos de Filtro

### 🏢 Cenário de Negócio
**Setor:** Bancário / Crédito  
**Problema:** Um banco possui 80+ features sobre cada solicitante de crédito (dados cadastrais, renda, histórico de pagamentos, score bureau, etc.). Muitas features são irrelevantes ou redundantes. O time precisa de um modelo interpretável e rápido para aprovação de crédito.

**Pergunta-chave:** *"Quais features são realmente relevantes para a decisão de crédito, sem precisar treinar dezenas de modelos?"*

### 🎯 Objetivo
O aluno será capaz de aplicar métodos de filtro que avaliam cada feature independentemente, sem treinar modelos, para uma pré-seleção rápida e eficiente.

### 📖 O que o aluno irá aprender

| Tópico | Descrição |
|--------|-----------|
| Variance Threshold | Remove features com variância muito baixa (quase constantes) |
| Correlação de Pearson | Identifica features altamente correlacionadas entre si (redundância) |
| Correlação com target | Seleciona features com alta correlação com a variável alvo |
| Mutual Information | Captura relações não-lineares entre feature e target |
| Chi-quadrado (χ²) | Para features categóricas vs target categórico |
| ANOVA F-test | Para features numéricas vs target categórico |
| SelectKBest / SelectPercentile | Wrappers do sklearn para automatizar a seleção |

### 🛠️ Dataset
Dataset sintético de crédito bancário (~10.000 registros, 40+ features) incluindo features propositalmente irrelevantes e redundantes para o aluno identificar e remover.

### 📦 Bibliotecas
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (VarianceThreshold, SelectKBest, mutual_info_classif, chi2, f_classif)

### 📋 Entregáveis do Notebook
1. Análise de variância e remoção de features constantes
2. Heatmap de correlação e remoção de features redundantes (correlação > 0.9)
3. Ranking por Mutual Information e Chi-quadrado
4. Comparação de modelo treinado com todas features vs features selecionadas
5. Conclusão: quantas features foram removidas sem perda de performance

---

## AULA 8 — Feature Selection: Métodos Embarcados (Embedded)

### 🏢 Cenário de Negócio
**Setor:** Seguros  
**Problema:** Uma seguradora quer prever sinistros de automóvel. O dataset tem 60+ features. Diferente dos métodos de filtro (aula anterior), queremos que o próprio modelo nos diga quais features são importantes durante o treinamento.

**Pergunta-chave:** *"Como aproveitar o próprio processo de treinamento do modelo para selecionar as features mais importantes?"*

### 🎯 Objetivo
O aluno será capaz de utilizar modelos que realizam seleção de features como parte do seu treinamento (Lasso, Ridge, importância de árvores).

### 📖 O que o aluno irá aprender

| Tópico | Descrição |
|--------|-----------|
| Regularização L1 (Lasso) | Zera coeficientes de features irrelevantes |
| Regularização L2 (Ridge) | Reduz coeficientes, mas não zera (shrinkage) |
| Elastic Net | Combinação de L1 + L2 |
| Feature Importance (árvores) | Random Forest, Gradient Boosting — importância baseada em impurity ou permutation |
| Permutation Importance | Model-agnostic: embaralha cada feature e mede queda de performance |
| SelectFromModel | Automatiza seleção usando threshold de importância |
| Redes Neurais e Feature Selection | Como dropout e weight decay atuam como regularizadores |

### 🛠️ Dataset
Dataset sintético de sinistros de seguros (~12.000 registros, 50+ features) com features propositalmente com diferentes níveis de relevância.

### 📦 Bibliotecas
`pandas`, `numpy`, `matplotlib`, `scikit-learn` (Lasso, Ridge, ElasticNet, RandomForestClassifier, SelectFromModel, permutation_importance)

### 📋 Entregáveis do Notebook
1. Treinamento de Lasso com coeficientes zerados identificados
2. Feature Importance com Random Forest (gráfico de barras)
3. Permutation Importance (model-agnostic)
4. SelectFromModel automatizando a seleção
5. Comparação: modelo com todas features vs modelo com features selecionadas (embedded)

---

## AULA 9 — Feature Selection: Métodos Wrapper (Backward Elimination + Forward Selection)

### 🏢 Cenário de Negócio
**Setor:** Marketing / CRM  
**Problema:** Uma empresa de assinaturas quer prever churn de clientes. Já fizeram filtro e embedded (aulas anteriores), mas querem garantir que encontraram o subset ótimo de features. Estão dispostos a investir mais tempo computacional para testar combinações de features.

**Pergunta-chave:** *"Qual é o subconjunto de features que maximiza a performance do modelo, testando combinações de inclusão e exclusão?"*

### 🎯 Objetivo
O aluno será capaz de implementar Backward Elimination (começando com todas e removendo) e Forward Selection (começando vazio e adicionando), além de Recursive Feature Elimination (RFE).

### 📖 O que o aluno irá aprender

| Tópico | Descrição |
|--------|-----------|
| Wrapper vs Filtro vs Embedded | Comparação das 3 abordagens |
| Backward Elimination | Começa com todas features, remove a menos importante iterativamente |
| Forward Selection | Começa sem features, adiciona a mais relevante iterativamente |
| RFE (Recursive Feature Elimination) | Remove features com menores coeficientes/importância recursivamente |
| RFECV | RFE com cross-validation para encontrar número ótimo de features |
| Custo computacional | 2^n combinações possíveis — quando wrapper é viável |
| Step Forward/Backward com mlxtend | Biblioteca dedicada para sequential feature selection |

### 🛠️ Dataset
Reutilização do dataset de churn (telecom), com ~30 features, tamanho adequado para wrapper methods sem explodir o tempo de execução.

### 📦 Bibliotecas
`pandas`, `numpy`, `matplotlib`, `scikit-learn` (RFE, RFECV, LogisticRegression, RandomForestClassifier), `mlxtend` (SequentialFeatureSelector)

### 📋 Entregáveis do Notebook
1. Implementação de Backward Elimination step-by-step (com log de remoção)
2. Implementação de Forward Selection step-by-step (com log de adição)
3. RFE e RFECV com gráfico de performance vs número de features
4. Comparação final: Filtro vs Embedded vs Wrapper — qual deu melhor resultado?
5. Tabela-resumo: quando usar cada método

---

## AULA 10 — Projeto Integrado + Revisão

### 🏢 Cenário de Negócio
**Setor:** Fintech / Crédito digital  
**Problema:** Uma fintech precisa construir um pipeline completo de Feature Engineering para seu modelo de aprovação de crédito. O aluno aplicará TUDO que aprendeu nas 9 aulas anteriores em um único projeto end-to-end.

**Pergunta-chave:** *"Como aplicar todas as técnicas de Feature Engineering de forma integrada em um projeto real?"*

### 🎯 Objetivo
O aluno construirá um pipeline completo de pré-processamento e feature engineering, desde dados brutos até features prontas para modelagem.

### 📖 Pipeline completo que o aluno irá construir

| Etapa | Técnica | Aula de referência |
|-------|---------|-------------------|
| 1. EDA e diagnóstico | Missing data, distribuições, correlações | Aulas 1, 2 |
| 2. Tratamento de missing | KNN Imputer + indicador de missing | Aula 1 |
| 3. Feature Scaling | StandardScaler via Pipeline | Aula 2 |
| 4. Tratamento de desbalanceamento | SMOTE ou class_weight | Aula 3 |
| 5. Redução de dimensionalidade | PCA para features numéricas correlacionadas | Aula 4 |
| 6. Feature Extraction | TF-IDF para campo texto (motivo da solicitação) | Aula 6 |
| 7. Feature Selection — Filtro | Variance Threshold + Mutual Information | Aula 7 |
| 8. Feature Selection — Embedded | Feature Importance com Random Forest | Aula 8 |
| 9. Feature Selection — Wrapper | RFECV para subset final | Aula 9 |
| 10. Modelo final | Treinamento, avaliação e interpretação | Todas |

### 🛠️ Dataset
Dataset sintético robusto de crédito fintech (~15.000 registros, 50+ features mistas: numéricas, categóricas, texto) com missing data, desbalanceamento e features redundantes intencionais.

### 📋 Entregáveis
1. Notebook completo com pipeline documentado
2. Comparação de métricas: modelo baseline (sem FE) vs modelo final (com FE completo)
3. Relatório de decisões: por que cada técnica foi escolhida
4. Apresentação dos resultados para a turma

---

## Resumo — Mapa das Aulas

| # | Tema | Dataset de Negócio | Técnicas Principais |
|---|------|-------------------|-------------------|
| 1 | Missing Data Imputation | E-commerce (propensão) | SimpleImputer, KNNImputer, IterativeImputer |
| 2 | Feature Scaling | Imobiliário (preços) | StandardScaler, MinMaxScaler, RobustScaler |
| 3 | Dados Desbalanceados | Financeiro (fraude) | SMOTE, Downsampling, Upsampling, SMOTETomek |
| 4 | PCA | Telecom (segmentação) | PCA, Scree Plot, Variância Explicada |
| 5 | LDA + T-SNE | Saúde (diagnóstico) | LDA, T-SNE, Comparação com PCA |
| 6 | Feature Extraction + Feature Store | Varejo (avaliações NLP) | TF-IDF, BOW, Feature Store (Feast) |
| 7 | Feature Selection — Filtro | Bancário (crédito) | Variance, MI, Chi², ANOVA, SelectKBest |
| 8 | Feature Selection — Embedded | Seguros (sinistros) | Lasso, Ridge, Feature Importance, Permutation |
| 9 | Feature Selection — Wrapper | Marketing (churn) | Backward, Forward, RFE, RFECV |
| 10 | Projeto Integrado | Fintech (crédito digital) | Pipeline completo end-to-end |

---

## Estrutura Padrão de Cada Notebook

Todos os notebooks seguem a mesma estrutura didática:

```
1. CONTEXTO DE NEGÓCIO
   → Cenário real, problema, pergunta-chave

2. IMPORTS E CONFIGURAÇÃO
   → Bibliotecas, seed, configurações

3. CARREGAMENTO E EDA
   → Leitura do CSV, .shape, .info(), .describe()
   → Visualizações exploratórias

4. CONCEITO TEÓRICO
   → Explicação em markdown com fórmulas e diagramas

5. IMPLEMENTAÇÃO PRÁTICA
   → Código sequencial, sem funções wrappers
   → Comentários explicativos em cada bloco

6. AVALIAÇÃO E COMPARAÇÃO
   → Métricas, gráficos comparativos

7. CONCLUSÃO
   → O que aprendemos, quando usar, quando NÃO usar
```
