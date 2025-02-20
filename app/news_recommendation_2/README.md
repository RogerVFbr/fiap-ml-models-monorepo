# News Recommendation System - Cold Start
Este arquivo descreve o sistema de recomendação de notícias, focando no processo de Cold Start. Ele inclui instruções 
para execução local, instalação de dependências e credenciamento AWS. O documento detalha etapas desde a carga e 
pré-processamento de dados até construção de dados analíticos, execução avaliação de previsões. 
Por fim, apresenta resultados de acurácia e uma conclusão sobre o desempenho do sistema.

---

## Índice
- [Execução local](#execução-local)
  - [Instalação de Dependências](#instalação-de-dependências)
  - [Credenciamento AWS](#credenciamento-aws)
- [Passo a Passo](#passo-a-passo)
  - [1. Carga de Dados](#1-carga-de-dados)
  - [2. Construção dos Dados Analíticos](#2-construção-dos-dados-analíticos)
  - [3. Execução das Previsões](#3-execução-das-previsões)
  - [4. Avaliação das Previsões](#4-avaliação-das-previsões)
- [Resultados](#resultados)
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

## Passo a Passo

### 1. Carga de Dados
O método **onboard** da classe DataService é chamado para carregar os dados de treinamento dos usuários (user_data_train), os dados de teste dos usuários (user_data_test) e os dados das notícias (news_data).

### 2. Construção dos Dados Analíticos
O método **build_mock_analytics_data** é chamado com os dados de treinamento dos usuários e os dados das notícias. Este método cria um conjunto de dados analíticos simulados que será usado para fazer previsões.

### 3. Execução das Previsões
O método **execute_predictions** é chamado com os dados analíticos, os dados de teste dos usuários e os dados das notícias. Este método realiza as previsões de quais notícias os usuários podem estar interessados, com base em um conjunto de regras e pesos definidos.

### 4. Avaliação das Previsões
O método **evaluate** é chamado com as previsões geradas. Este método calcula várias métricas de desempenho (como acurácia, F1 score, precisão e recall) para avaliar a qualidade das previsões feitas pelo sistema.

---

## Resultados
Os resultados abaixo representam a acurácia na predição de notícias levando em consideração os seguintes critérios:
* A notícia acessada pelo usuário está dentro de lista de recomendações de tamanho *N* (Hipótese *N*).
* O critério de recomendação é baseado em uma combinação de features:
  * Maior número de Visualizações
  * Maior Engajamento médio (clicks na página da notícia)
  * Maior Tempo médio na página por caractere 
  * Maior Porcentagem média de scroll.
  * Janela temporal de medição: contada para trás a partir do momento em que o usuário solicitou o acesso â notícia. 
* Em cada experimento foi atribuído uma combinação de pesos diferente para cada critério, assim como uma janela específica.
Foram testadas dezenas de combinações, apenas algumas foram apresentadas para referência.

### Cenário 1

**Total Samples:** 103,265  
**Window (Days):** 1  
**Weights:**  
- Views: **1**  
- Engagement: **0**  
- Time on Page per Character: **0**  
- Scroll Percentage: **1**  

| Criteria      | Match Count | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|--------------|------------|--------------|--------------|--------------|------------|
| Hypothesis 1 | 2          | 0.00         | 0.00         | 0.00         | 0.00       |
| Hypothesis 2 | 6          | 0.01         | 0.00         | 0.00         | 0.01       |
| Hypothesis 3 | 8          | 0.01         | 0.01         | 0.01         | 0.01       |
| Hypothesis 4 | 8          | 0.01         | 0.01         | 0.01         | 0.01       |
| Hypothesis 5 | 8          | 0.01         | 0.01         | 0.01         | 0.01       |
| Hypothesis 6 | 11         | 0.01         | 0.01         | 0.01         | 0.01       |
| Hypothesis 7 | 12         | 0.01         | 0.01         | 0.01         | 0.01       |
| Hypothesis 8 | 12         | 0.01         | 0.01         | 0.01         | 0.01       |
| Hypothesis 9 | 14         | 0.01         | 0.01         | 0.01         | 0.01       |
| Hypothesis 10| 14         | 0.01         | 0.01         | 0.01         | 0.01       |
⏳ **Elapsed Time:** 4 minutes 9 seconds 776 milliseconds  

### Cenário 2

**Total Samples:** 103,265  
**Window (Days):** 1  
**Weights:**  
- Views: **1**  
- Engagement: **1**  
- Time on Page per Character: **1**  
- Scroll Percentage: **2**  

| Criteria      | Match Count | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|--------------|------------|--------------|--------------|--------------|------------|
| Hypothesis 1 | 4          | 0.00         | 0.00         | 0.00         | 0.00       |
| Hypothesis 2 | 4          | 0.00         | 0.00         | 0.00         | 0.00       |
| Hypothesis 3 | 6          | 0.01         | 0.00         | 0.00         | 0.01       |
| Hypothesis 4 | 8          | 0.01         | 0.00         | 0.00         | 0.01       |
| Hypothesis 5 | 8          | 0.01         | 0.00         | 0.00         | 0.01       |
| Hypothesis 6 | 8          | 0.01         | 0.00         | 0.00         | 0.01       |
| Hypothesis 7 | 8          | 0.01         | 0.00         | 0.00         | 0.01       |
| Hypothesis 8 | 8          | 0.01         | 0.00         | 0.00         | 0.01       |
| Hypothesis 9 | 8          | 0.01         | 0.00         | 0.00         | 0.01       |
| Hypothesis 10| 8          | 0.01         | 0.00         | 0.00         | 0.01       |
⏳ **Elapsed Time:** 4 minutes 6 seconds 948 milliseconds  

### Cenário 3

**Total Samples:** 103,265  
**Window (Days):** 2  
**Weights:**  
- Views: **0**  
- Engagement: **0**  
- Time on Page per Character: **0**  
- Scroll Percentage: **1**  

| Criteria      | Match Count | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%) |
|--------------|------------|--------------|--------------|--------------|------------|
| Hypothesis 1 | 4          | 0.00         | 0.00         | 0.00         | 0.00       |
| Hypothesis 2 | 4          | 0.00         | 0.00         | 0.00         | 0.00       |
| Hypothesis 3 | 143        | 0.14         | 0.13         | 0.13         | 0.14       |
| Hypothesis 4 | 143        | 0.14         | 0.13         | 0.13         | 0.14       |
| Hypothesis 5 | 143        | 0.14         | 0.13         | 0.13         | 0.14       |
| Hypothesis 6 | 144        | 0.14         | 0.14         | 0.14         | 0.14       |
| Hypothesis 7 | 144        | 0.14         | 0.14         | 0.14         | 0.14       |
| Hypothesis 8 | 1096       | 1.06         | 1.06         | 1.06         | 1.06       |
| Hypothesis 9 | 1096       | 1.06         | 1.06         | 1.06         | 1.06       |
| Hypothesis 10| 1097       | 1.06         | 1.06         | 1.06         | 1.06       |
⏳ **Elapsed Time:** 5 minutes 59 seconds 809 milliseconds  

---

## Conclusão
Observa-se que a melhor acurácia foi obtida recomendando uma lista de 10 notícias em uma janela temporal de 2 dias, considerando
apenas aquelas com o maior percentual de scroll médio. A acurácia foi de apenas 1.06%, o que indica que a estratégia 
é pouco eficaz para o cenário proposto.

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
