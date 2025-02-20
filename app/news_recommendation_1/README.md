# News Recommendation System - Hot/Warm Start
Este arquivo descreve o sistema de recomendação de notícias, focando no processo de Hot/Warm Start. Ele inclui instruções para execução local, instalação de dependências e credenciamento AWS. O documento detalha etapas desde a carga e pré-processamento de dados até a clusterização, engenharia de características e treino/validação de modelos. Por fim, apresenta resultados de acurácia e uma conclusão sobre o desempenho do sistema.

---

## Índice
- [Execução local](#execução-local)
  - [Instalação de Dependências](#instalação-de-dependências)
  - [Credenciamento AWS](#credenciamento-aws)
- [Publicação Github Actions/AWS](#publicação-github-actionsaws)
- [Passo a Passo](#passo-a-passo)
  - [1. Carga de Dados: *DataService*](#1-carga-de-dados-dataservice)
  - [2. Pré-Processamento de Notícias: *NewsProcessor*](#2-pré-processamento-de-notícias-newsprocessor)
  - [3. Clusterização de Notícias: *NewsClusterizer*](#3-clusterização-de-notícias-newsclusterizer)
  - [4. Feature Engineering: *UserFeatureEngineering*](#4-feature-engineering-userfeatureengineering)
  - [5. Treino e Validação de Modelo Neural: *NewsClusterNNPredictor*](#5-treino-e-validação-de-modelo-neural-newsclusternnpredictor)
  - [6. Treino e Validação de Modelo Clássico: *NewsClusterClassicPredictor*](#6-treino-e-validação-de-modelo-clássico-newsclusterclassicpredictor)
  - [7. Validação de Acurácia Final: *NewsPagePredictor*](#7-validação-de-acurácia-final-newspagepredictor)
- [Resultados](#resultados)
  - [Acurácia de Predição de Clusters (Redes Neurais)](#acurácia-de-predição-de-clusters-redes-neurais)
  - [Acurácia de Predição de Clusters (Algoritmos Clássicos)](#acurácia-de-predição-de-clusters)
  - [Acurácia de Predição de Páginas (Hot/Warm Start)](#acurácia-de-predição-de-páginas-hotwarm-start)
- [Conclusão](#conclusão)

---

## Execução Local
Execute os passos a seguir para executar o procedimento localmente.

### Instalação de Dependências
A partir da raiz do projeto, crie um ambiente virtual usando **Python 3.10** e instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

### Credenciamento AWS
Crie ou edite o arquivo `~/.aws/credentials` e adicione as credenciais fornecidas:
```txt
[default]
aws_access_key_id = SEU_ACCESS_KEY_ID
aws_secret_access_key = SEU_SECRET_ACCESS_KEY
```
> Esta etapa garante que os datasets utilizados serão baixados do Bucket S3 da solução.

---

## Publicação Github Actions/AWS
* **Configuração do Workflow.** O workflow do *Github Actions* é acionado quando um pull request é fechado nas branches main ou develop.
* **Checkout do Código.** O código do repositório é baixado usando a ação actions/checkout@v3.
* **Configuração do Terraform.** O Terraform é configurado usando a ação hashicorp/setup-terraform@v1.
* **Verificação das Versões do Python e AWS CLI.** As versões do Python e AWS CLI são verificadas e exibidas.
* **Exibição do Conteúdo do Diretório App.** O conteúdo do diretório app é exibido para verificação.
* **Extração do Nome da Branch.** O nome da branch é extraído e armazenado para uso posterior.
* **Inicialização do Terraform.** O Terraform é inicializado com a configuração do backend.
* **Planejamento do Terraform.** O plano do Terraform é gerado para verificar as mudanças que serão aplicadas.
* **Instalação dos Pacotes Python.** Os pacotes Python necessários são instalados a partir do arquivo requirements.txt.
* **Aplicação do Terraform (Treinamento e Exportação dos Modelos).** O Terraform aplica as mudanças, treinando e exportando os modelos.
  * **Módulo Terraform - Executors.** Utiliza o recurso null_resource com triggers baseados no hash do arquivo zipado do código-fonte. Isso garante que o script só será executado se houver alterações no código, permitindo a presença de múltiplos modelos no repositório.
* **Upload dos Modelos.** Os modelos treinados e outros entregáveis como diagramas e dados são sincronizados com um bucket S3.
* **Criação de Pull Request para a Branch Main.** Se a branch atual for develop, um pull request é criado para a branch main.

---

## Passo a Passo

### 1. Carga de Dados: *DataService*
A classe DataService é responsável por gerenciar operações de dados para o sistema de recomendação de notícias. Quando o método de entrada onboard é executado, ele realiza as seguintes etapas:
* **Verificação e Download de Arquivos Parquet.** Verifica se os arquivos Parquet existem localmente. Se não existirem, faz o download a partir de um bucket S3.
* **Carga de Dados.** Carrega os dados de treinamento, teste e notícias a partir dos arquivos Parquet para *Dataframes Polars*.
* **Ajuste dos Tipos de Dados.** Ajusta os tipos de dados das colunas nos DataFrames de treinamento, teste e notícias.
* **Filtragem dos Dados.** Filtra os dados de treinamento, teste e notícias para remover inconsistências e garantir a qualidade.
* **Sanitização dos Dados.** Remove colunas desnecessárias dos DataFrames para otimizar o conjunto de dados.
   
### 2. Pré-Processamento de Notícias: *NewsProcessor*
A classe NewsProcessor é responsável por processar os dados de notícias para o sistema de recomendação de notícias. Quando o método de entrada execute é chamado, ele realiza as seguintes etapas:
* **Verificação de Dados Pré-processados**. Verifica se os dados de notícias pré-processados já existem. Se existirem e o reprocessamento não for forçado, carrega esses dados e retorna.
* **Ordenação dos Dados.** Ordena os dados de notícias pela coluna modified em ordem decrescente.
* **Aplicação de Etapas de Processamento.** Aplica uma série de etapas de processamento nos dados de notícias. As etapas incluem:
  * **Preparação e Tokenização da Coluna soup.** Concatena as colunas title e caption em uma nova coluna soup e a tokeniza.
  * **Remoção de Pontuação e Conversão para Minúsculas.** 
  * **Remoção de Stopwords.** Remove palavras comuns (stopwords) em português.
  * **Remoção de Palavras com Números.**
  * **Lematização.** Converte as palavras para suas formas base (lematização).
  * **Remoção de Acentos.**.
  * **Junção das Listas de Palavras.** Junta a lista de palavras na coluna soup_clean em uma única string.
* **Salvamento dos Dados Pré-processados.** Salva os dados de notícias pré-processados em um arquivo Parquet.

### 3. Clusterização de Notícias: *NewsClusterizer*
A classe NewsClusterizer é responsável por agrupar artigos de notícias com base em seu conteúdo. Quando o método de entrada execute é chamado, ele realiza as seguintes etapas:  
* **Verificação de Dados Classificados.** Verifica se os dados de notícias já foram classificados anteriormente. Se existirem e o reprocessamento não for forçado, carrega esses dados e a matriz de similaridade e os retorna.
* **Ordenação dos Dados.** Ordena os dados de notícias pela coluna modified em ordem decrescente.
* **Construção dos Clusters.** Converte os dados textuais em uma matriz de características TF-IDF usando o TfidfVectorizer e aplica o algoritmo de clustering KMeans para formar os clusters.
* **Construção da Matriz de Similaridade.** Calcula a similaridade coseno entre cada par de centros dos clusters e cria uma matriz de similaridade.
* **Salvamento dos Dados Classificados.** Salva os dados de notícias classificados e a matriz de similaridade em arquivos Parquet.

### 4. Feature Engineering: *UserFeatureEngineering*
A classe UserFeatureEngineering é responsável por realizar a engenharia de características nos dados dos usuários para o sistema de recomendação de notícias. Quando o método de entrada execute é chamado, ele realiza as seguintes etapas:  
* **Verificação de Dados Pré-processados.** Verifica se os dados de usuários já foram processados anteriormente. Se existirem e o reprocessamento não for forçado, carrega esses dados e os retorna.
* **Identificação de Clusters no Histórico do Usuário.** Substitui as páginas no histórico do usuário pelos clusters correspondentes dos dados de notícias.
* **Tratamento do Histórico de Timestamp.** Calcula a diferença de tempo entre cada timestamp e o último timestamp, criando uma nova característica baseada nessa diferença.
* **Normalização de Colunas Numéricas.** Normaliza as colunas de listas numéricas nos dados dos usuários, dividindo cada valor pelo valor máximo da lista.
* **Filtragem de Linhas Irrelevantes.** Filtra as linhas dos dados de treinamento onde o valor máximo na coluna timestampHistory_norm está abaixo de um certo limite.
* **Criação de Colunas de Características e Alvos.** Inicializa colunas de características para cada cluster e colunas de alvo para o cluster e a página nos dados de treinamento e teste.
* **População das Colunas de Características e Alvos nos Dados de Treinamento.** Itera pelo histórico do usuário e atribui pesos às colunas de características com base na interação do usuário com cada cluster.
* **População das Colunas de Características e Alvos nos Dados de Teste.** Itera pelo histórico do usuário e atribui pesos às colunas de características com base na interação do usuário com cada cluster.
* **Salvamento dos Dados Processados.** Salva os dados de usuários processados em arquivos Parquet.

### 5. Treino e Validação de Modelo Neural: *NewsClusterNNPredictor*
A classe NewsClusterNNPredictor é responsável por prever os clusters de notícias usando uma rede neural. Quando o método de entrada execute é chamado, ele realiza as seguintes etapas:  
* **Verificação de Dados Previstos.** Verifica se os dados de teste de usuários já foram previstos anteriormente. Se existirem e o reprocessamento não for forçado, carrega esses dados e os retorna.
* **Preparação dos Dados de Treinamento e Teste.** Seleciona as colunas de características e o alvo (target_cluster) dos dados de treinamento e teste.
Converte os dados para tensores do PyTorch.
* **Inicialização dos Modelos.** Inicializa os modelos de rede neural definidos no módulo news_cluster_predictor_nn_model.
* **Configuração do DataLoader do PyTorch.** Configura o DataLoader do PyTorch para os dados de treinamento, permitindo o carregamento em mini-lotes.
* **Treinamento e Avaliação dos Modelos.** Treina e avalia cada modelo de rede neural, calculando métricas de desempenho (precisão, recall, F1 score, etc.) para três hipóteses:
    * Hipótese 1: A previsão corresponde ao alvo.
    * Hipótese 2: A previsão corresponde ao alvo ou aos 2 clusters mais similares.
    * Hipótese 3: A previsão corresponde ao alvo ou aos 3 clusters mais similares.
  * Seleciona o melhor modelo com base nas métricas de desempenho.
* **Definição das Previsões no Conjunto de Dados.** Atribui as previsões do melhor modelo ao conjunto de dados de teste.
* **Salvamento dos Dados Previstos e do Modelo.**
  * Salva os dados de teste previstos em um arquivo Parquet.
  * Salva o modelo PyTorch treinado.

### 6. Treino e Validação de Modelo Clássico: *NewsClusterClassicPredictor*
A classe NewsClusterClassicPredictor é responsável por prever os clusters de notícias usando modelos clássicos de aprendizado de máquina. Quando o método de entrada execute é chamado, ele realiza as seguintes etapas:  
* **Verificação de Dados Previstos.** Verifica se os dados de teste de usuários já foram previstos anteriormente. Se existirem e o reprocessamento não for forçado, carrega esses dados e os retorna.
* **Preparação dos Dados de Treinamento e Teste.** Seleciona as colunas de características e o alvo (target_cluster) dos dados de treinamento e teste, convertendo os dados para o formato Pandas.
* **Treinamento e Avaliação dos Modelos.** Treina e avalia cada modelo de aprendizado de máquina definido nos presets, calculando métricas de desempenho (precisão, recall, F1 score, etc.) para três hipóteses:
  * Hipótese 1: A previsão corresponde ao alvo.
  * Hipótese 2: A previsão corresponde ao alvo ou aos 2 clusters mais similares.
  * Hipótese 3: A previsão corresponde ao alvo ou aos 3 clusters mais similares.
  * Seleciona o melhor modelo com base nas métricas de desempenho.
* **Definição das Previsões no Conjunto de Dados.** Atribui as previsões do melhor modelo ao conjunto de dados de teste.
* **Salvamento dos Dados Previstos e do Modelo.**
  * Salva os dados de teste previstos em um arquivo Parquet.
  * Salva o modelo treinado do Scikit-Learn.

### 7. Validação de Acurácia Final: *NewsPagePredictor*
A classe NewsPagePredictor é responsável por prever as páginas de notícias que um usuário pode estar interessado, utilizando clusters de notícias previamente calculados. Quando o método de entrada predict é chamado, ele realiza as seguintes etapas:  
* **Processamento dos Dados de Teste.** Utiliza a biblioteca polars para manipular os dados de teste dos usuários, aplicando as funções auxiliares para prever as páginas de notícias:
  * Seleciona as colunas relevantes (target_page, target_timestamp, predicted_cluster_nn, predicted_cluster_classic).
  * Mapeia os clusters similares para as previsões de clusters (predicted_clusters_nn, predicted_clusters_classic).
  * Obtém as previsões de notícias para cada cluster similar.
  * Verifica se a página alvo está entre as previsões e marca como acerto ou erro.
* **Cálculo de Métricas de Desempenho.** Calcula métricas de desempenho (precisão, recall, F1 score, etc.) para diferentes hipóteses de acerto:
  * Hipótese 1: A previsão corresponde à página alvo.
  * Hipótese 2: A previsão corresponde à página alvo ou às 2 páginas mais similares.
  * Hipótese *N*: A previsão corresponde à página alvo ou às *N* páginas mais similares.
  * Armazena e imprime as métricas de desempenho.

---

## Resultados

### Acurácia de Predição de Clusters (Redes Neurais)
Os resultados a seguir apresentam a acurácia de predição de clusters para os modelos de redes neurais baseados nas seguintes hipóteses:
* O modelo infere exatamente o cluster ao qual a próxima notícia acessada pelo usuário pertence.
* A próxima notícia acessada pelo usuário pertence ao cluster inferido ou a um dos *N* clusters mais similares.
>Por similaridade, entende-se a distância entre os centróides dos clusters. Quanto menor esta distância, mais similar é o cluster.

#### Hypothesis 1: Prediction Matches Target

| Architecture                  | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|--------------------------------|-------------|-------------|--------------|------------|
| NewsClusterInferenceModel_0    | 37.89       | 39.78       | 47.34        | 37.89      |
| NewsClusterInferenceModel_1    | 38.00       | 39.77       | 48.46        | 38.00      |
| NewsClusterInferenceModel_2    | 7.26        | 9.71        | 51.91        | 7.26       |
| NewsClusterInferenceModel_3    | 52.23       | 45.46       | 45.45        | 52.23      |
| NewsClusterInferenceModel_4    | 31.24       | 34.99       | 43.62        | 31.24      |
| NewsClusterInferenceModel_5    | 53.21       | 43.49       | 39.39        | 53.21      |
| NewsClusterInferenceModel_6    | 32.54       | 35.90       | 45.39        | 32.54      |

#### Hypothesis 2: Prediction Matches Target or 1 Most Similar Cluster

| Architecture                  | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|--------------------------------|-------------|-------------|--------------|------------|
| NewsClusterInferenceModel_0    | 80.12       | 77.04       | 80.10        | 80.12      |
| NewsClusterInferenceModel_1    | 77.92       | 73.33       | 80.69        | 77.92      |
| NewsClusterInferenceModel_2    | 71.03       | 74.54       | 84.35        | 71.03      |
| NewsClusterInferenceModel_3    | 78.53       | 73.85       | 78.74        | 78.53      |
| NewsClusterInferenceModel_4    | 74.68       | 71.48       | 79.22        | 74.68      |
| NewsClusterInferenceModel_5    | 75.27       | 68.20       | 71.65        | 75.27      |
| NewsClusterInferenceModel_6    | 79.89       | 78.61       | 79.38        | 79.89      |

#### Hypothesis 3: Prediction Matches Target or 2 Most Similar Clusters

| Architecture                  | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|--------------------------------|-------------|-------------|--------------|------------|
| NewsClusterInferenceModel_0    | 90.46       | 88.14       | 89.23        | 90.46      |
| NewsClusterInferenceModel_1    | 90.43       | 87.57       | 89.28        | 90.43      |
| NewsClusterInferenceModel_2    | 80.24       | 83.10       | 92.54        | 80.24      |
| NewsClusterInferenceModel_3    | 91.00       | 89.06       | 89.29        | 91.00      |
| NewsClusterInferenceModel_4    | 83.97       | 84.35       | 90.74        | 83.97      |
| NewsClusterInferenceModel_5    | 88.92       | 85.13       | 85.75        | 88.92      |
| NewsClusterInferenceModel_6    | 88.25       | 87.43       | 88.35        | 88.25      |

#### Arquitetura com melhor performance 
```python
NewsClusterInferenceModel_3(
  (conv1): Conv1d(1, 16, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv2): Conv1d(16, 32, kernel_size=(3,), stride=(1,), padding=(1,))
  (fc1): Linear(in_features=320, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=10, bias=True)
  (softmax): Softmax(dim=1)
)
```
---

### Acurácia de Predição de Clusters (Algoritmos Clássicos)
Os resultados a seguir apresentam a acurácia de predição de clusters para os algoritmos clássicos nas seguintes hipóteses:
* O modelo infere exatamente o cluster ao qual a próxima notícia acessada pelo usuário pertence.
* A próxima notícia acessada pelo usuário pertence ao cluster inferido ou a um dos *N* clusters mais similares.
>Por similaridade, entende-se a distância entre os centróides dos clusters. Quanto menor esta distância, mais similar é o cluster.

#### Hypothesis 1: Prediction Matches Target

| Architecture              | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|---------------------------|-------------|-------------|--------------|------------|
| SupportVectorMachine      | 59.21       | 45.63       | 45.24        | 59.21      |
| DecisionTree             | 6.99        | 1.31        | 2.60         | 6.99       |
| RandomForest             | 38.67       | 41.58       | 47.29        | 38.67      |
| XGBClassifier            | 13.39       | 12.17       | 50.48        | 13.39      |

#### Hypothesis 2: Prediction Matches Target or 1 Most Similar Cluster

| Architecture              | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|---------------------------|-------------|-------------|--------------|------------|
| SupportVectorMachine      | 75.65       | 66.48       | 76.24        | 75.65      |
| DecisionTree             | 75.30       | 71.22       | 88.80        | 75.30      |
| RandomForest             | 80.78       | 78.62       | 78.58        | 80.78      |
| XGBClassifier            | 79.88       | 80.22       | 83.63        | 79.88      |

#### Hypothesis 3: Prediction Matches Target or 2 Most Similar Clusters

| Architecture              | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|---------------------------|-------------|-------------|--------------|------------|
| SupportVectorMachine      | 89.80       | 85.57       | 89.77        | 89.80      |
| DecisionTree             | 89.83       | 87.18       | 90.30        | 89.83      |
| RandomForest             | 89.49       | 88.22       | 88.89        | 89.49      |
| XGBClassifier            | 90.16       | 89.40       | 90.42        | 90.16      |

#### Algoritmo com melhor performance 
```python
LinearSVC(C=1, class_weight='balanced')
```

---

### Acurácia de Predição de Páginas (Hot/Warm Start)
Os resultados a seguir apresentam a acurácia de predição de páginas (notícias) para os algoritmos clássicos e redes neurais nas seguintes hipóteses:
* A próxima notícia acessada pelo usuário está dentre as *N* notícias sugeridas.
>A lista de notícias sugeridas considera as *N* mais recentes (*Hipothesis N*) dentro do cluster previsto e os 4 mais similares.

**Total samples:** 35,049  
**Clusters:** Primary + 4 secondary (5)  
**Selected news:** 1-10  
**Cluster Predictors:** NN (Neural Network), CLS (Classic)

| Criteria       | Match Count NN | Match Count CLS | Accuracy NN | Accuracy CLS | F1 Score NN | F1 Score CLS | Precision NN | Precision CLS | Recall NN | Recall CLS |
|---------------|---------------|-----------------|-------------|--------------|-------------|--------------|--------------|---------------|------------|-------------|
| Hypothesis 1  | 53            | 53              | 0.15%       | 0.15%        | 0.22%       | 0.22%        | 8.43%        | 8.43%         | 0.15%      | 0.15%       |
| Hypothesis 2  | 98            | 98              | 0.28%       | 0.28%        | 0.45%       | 0.44%        | 8.69%        | 8.69%         | 0.28%      | 0.28%       |
| Hypothesis 3  | 179           | 179             | 0.51%       | 0.51%        | 0.75%       | 0.75%        | 8.69%        | 8.69%         | 0.51%      | 0.51%       |
| Hypothesis 4  | 207           | 207             | 0.59%       | 0.59%        | 0.89%       | 0.89%        | 9.12%        | 9.12%         | 0.59%      | 0.59%       |
| Hypothesis 5  | 253           | 252             | 0.72%       | 0.72%        | 1.07%       | 1.06%        | 9.21%        | 9.21%         | 0.72%      | 0.72%       |
| Hypothesis 6  | 1165          | 1163            | 3.32%       | 3.32%        | 3.82%       | 3.81%        | 9.21%        | 9.21%         | 3.32%      | 3.32%       |
| Hypothesis 7  | 1292          | 1291            | 3.69%       | 3.68%        | 4.23%       | 4.22%        | 9.21%        | 9.21%         | 3.69%      | 3.68%       |
| Hypothesis 8  | 1333          | 1332            | 3.80%       | 3.80%        | 4.40%       | 4.40%        | 9.41%        | 9.40%         | 3.80%      | 3.80%       |
| Hypothesis 9  | 1885          | 1925            | 5.38%       | 5.49%        | 6.00%       | 6.06%        | 9.48%        | 9.48%         | 5.38%      | 5.49%       |
| Hypothesis 10 | 2375          | 2412            | 6.78%       | 6.88%        | 7.36%       | 7.42%        | 9.48%        | 9.48%         | 6.78%      | 6.88%       |

---

## Conclusão

### Predição de Clusters
O sistema apresentou resultados **satisfatórios** em relação à acurácia de predição de clusters. 
Os modelos de redes neurais e algoritmos clássicos obtiveram desempenho semelhante, com destaque para a hipótese 3 
(a notícia visualizada encontra-se no cluster recomendado ou nos dois mais semelhantes), alcançando até 91.00% de acurácia 
para os modelos de redes neurais e 90.16% para os algoritmos clássicos.

### Predição de Páginas (Notícias)
Já para a predição de notícias, o sistema apresentou resultados **muito pouco satisfatórios**, com acurácia máxima de 
6.88% para a hipótese 10, onde a notícia visualizada pelo usuário estava entre as 10 sugeridas.

### Pontos de Melhoria/Atenção
Foram identificados problemas fundamentais no dataset de teste/validação:
* Timestamps de visualização anteriores ao de publicação de notícias no dataset de validação/teste, invalidando os *datapoints*.
* Identificadores de notícias no dataset de validação não contidos no dataset de notícias.

Estes aspectos nos levam a crer que há **imprecisões generalizadas** no processo de aquisição de dados da solução,
gerando inconsistências nos resultados de predição. Em uma situação real, seria necessária uma ampla revisão nestes processos,
possivelmente invalidando completamente os dados obtidos até o momento. Algumas sugestões para esta revisão incluem:
* Garantir que os timestamps de visualização não sejam capturados na ponta do cliente e sim em uma plataforma especializada de *analytics*.
* Utilização de uma plataforma de *Data Quality* para monitoramento contínuo da qualidade dos dados.
* Atualização permanente do dataset de validação para que as notícias mais recentes estejam sempre presentes.
