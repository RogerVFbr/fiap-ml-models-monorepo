# News Recommendation System
Os entregáveis contidos neste arquivo ZIP referem-se ao projeto *Datathon* da **Fase 5** do curso de pós-graduação 
Engenharia de Machine Learning da Universidade FIAP. Este arquivo contém informações sobre como os requisitos foram 
abordados, bem como links diretos para os repositórios de código desenvolvidos.

## Índice
- [Abordagem](#abordagem)
- [Repositórios](#repositorios)
- [Tecnologias](#tecnologias)
- [Atendimento de Requisitos](#atendimento-de-requisitos)
  - [Cold Starts](#cold-starts)
  - [Recência](#recencia)
  - [Treinamento do Modelo](#treinamento-do-modelo)
  - [Salvamento do Modelo](#salvamento-do-modelo)
  - [API](#api)
  - [Docker](#docker)
  - [Testes](#testes)
  - [Deploy](#deploy)


## Abordagem
O projeto trata-se de um sistema para recomendações de notícias. A premissa utilizada neste projeto foi entendida sob 
duas óticas:

1. **Hot / Warm Starts**: Estratégia para lidar com usuários ou itens com algum histórico de acesso. Em alto 
nível, constitui-se em:
   * Classificação de notícias em classes (clusters).
   * Inferência do cluster de interesse a partir do histórico de acessos do usuário.
   * Recomendação de *N* notícias mais recentes do cluster inferido ou *top M* clusters similares.


2. **Cold Start**: Estratégias para lidar com usuário sobre os quais não se tem nenhuma informação histórica. Em linhas 
gerais:
    * Na ausência de histórico, o usuário receberá como recomendações as notícias mais relevantes em uma janela temporal 
medida a partir do momento do seu acesso até *N* horas antes.

A **métrica / critério de sucesso** foi inferida da seguinte forma:

* A notícia escolhida pelo usuário está dentro da lista de notícias recomendadas. A partir daí, foi criada uma medida de 
**acurácia**.

## Repositórios
O código fonte deste projeto foi dividido em dois repositórios:
1. **Monorepo de Modelos**
    * Link direto: [https://github.com/RogerVFbr/fiap-ml-models-monorepo/tree/develop](https://github.com/RogerVFbr/fiap-ml-models-monorepo/tree/develop)
    * Detalhamento sobre os procedimentos de treinamento, validação e publicação ([Hot/Warm Start](https://github.com/RogerVFbr/fiap-ml-models-monorepo/tree/develop/app/news_recommendation_1), [Cold Start](https://github.com/RogerVFbr/fiap-ml-models-monorepo/tree/develop/app/news_recommendation_2)).
    * Código fonte para treinamento e validação de modelos ([Hot/Warm Start](https://github.com/RogerVFbr/fiap-ml-models-monorepo/tree/develop/app/news_recommendation_1), [Cold Start](https://github.com/RogerVFbr/fiap-ml-models-monorepo/tree/develop/app/news_recommendation_2)).
    * Esteira de CI/CD com GitHub Actions para treinamento e publicação ([Workflows](https://github.com/RogerVFbr/fiap-ml-models-monorepo/tree/develop/.github/workflows)).
2. **API de Inferência**
    * Link direto: [https://github.com/RogerVFbr/fiap-api/tree/develop](https://github.com/RogerVFbr/fiap-api/tree/develop)
    * Detalhamento sobre o racional escolhido, testagem local e publicada (AWS) ([README.md](https://github.com/RogerVFbr/fiap-api/blob/develop/README.md)).
    * Código fonte e imagem Docker de API de inferência baseada em *Python Flask* ([App](https://github.com/RogerVFbr/fiap-api/tree/develop/app)).
    * Definição de IaC (infraestrutura como código) para API Serverless e estruturas auxiliares ([Infra](https://github.com/RogerVFbr/fiap-api/tree/develop/infra)).
    * Esteira de CI/CD com GitHub Actions para dockerização e publicação ([Workflows](https://github.com/RogerVFbr/fiap-api/tree/develop/.github/workflows)).

## Tecnologias
* **Modelagem**: Python, Pandas, Polars, Pytorch, Scikit-learn, Numpy, Spacy.
* **API**: Python, Flask, Docker.
* **AWS**: Lambda, API Gateway, Cognito, S3, ECR, KMS, IAM, Cloudwatch.
* **CI/CD**: Github, GitHub Actions, AWS CLI.
* **Infraestrutura**: Terraform.

## Atendimento de Requisitos

### Cold Starts
Dentro do escopo deste projeto, *Cold Start* foi entendido como o caso extremo onde não se tem absolutamente nenhuma
informação histórica ou contextual capaz de tipificar o usuário. Para lidar com este cenário, foi implementada uma
estratégia de recomendação baseada nas notícias mais *relevantes* em uma janela temporal de *N* horas antes do acesso do
usuário. O detalhamento da implementação desta estratégia pode ser vista neste [link](https://github.com/RogerVFbr/fiap-ml-models-monorepo/tree/develop/app/news_recommendation_2).

### Recência
O conceito de recência foi utilizado em alguns pontos específicos da estratégia.
* **Modelo de Inferência de Cluster de Notícias**: Durante a etapa de *feature engineering*, a recência da visualização 
da notícia possui um peso maior quando comparado ao peso de dados como *tempo de permanência no site* ou *percentual de scroll*.
* **Recomendação de Notícias**: Serão recomendadas as notícias mais recentes dentro dos clusters selecionados.

### Treinamento do Modelo
O treinamento do modelo de recomendação de clusters de notícias em seus sabores *Neural Network* e *Classic* foi feito 
no repositório de modelos e localizado nas classes 
[NewsClusterNNPredictor](https://github.com/RogerVFbr/fiap-ml-models-monorepo/blob/develop/app/news_recommendation_1/predictors/news_cluster_predictor_nn.py) e 
[NewsClusterClassicPredictor](https://github.com/RogerVFbr/fiap-ml-models-monorepo/blob/develop/app/news_recommendation_1/predictors/news_cluster_predictor_classic.py).

### Salvamento do Modelo
O salvamento do modelo é executado ao término da etapa de treinamento, podendo ser visualizado nas classes 
[NewsClusterNNPredictor](https://github.com/RogerVFbr/fiap-ml-models-monorepo/blob/develop/app/news_recommendation_1/predictors/news_cluster_predictor_nn.py) e 
[NewsClusterClassicPredictor](https://github.com/RogerVFbr/fiap-ml-models-monorepo/blob/develop/app/news_recommendation_1/predictors/news_cluster_predictor_classic.py).
Em tempo de esteira, estes arquivos são persistidos em um bucket S3 na AWS.

### API
Foi criada uma API de inferência para recomendação de notícias, baseada em *Python Flask*. O código fonte pode ser
encontrado [aqui](https://github.com/RogerVFbr/fiap-api/tree/develop).

### Docker
A dockerização foi aplicada na API Flask e sua definição pode ser avaliada [neste link](https://github.com/RogerVFbr/fiap-api/blob/develop/app/Dockerfile).
A imagem é construida de fato em tempo de esteira e publicada em um repositório AWS Elastic Container Registry (ECR), para
posteriormente ser utilizada na infra estrutura da API.

### Testes
Os testes foram realizados em ambiente local e em nuvem com chamadas HTTP executadas a partir da aplicação [Insomnia](https://insomnia.rest/download).
Foi disponibilizada uma *collection* para facilitar sua validação, já lidando com a parte de autenticação, disponível 
[aqui](https://github.com/RogerVFbr/fiap-api/tree/develop/docs) (arquivo *Insomnia_2025-xx-xx.json*).

### Deploy
O processo de publicação tanto do modelo quanto da API na AWS foi automatizado com o uso de *GitHub Actions*. Os workflows 
utilizados podem ser encontrados nos seguintes links: 
[Modelos](https://github.com/RogerVFbr/fiap-ml-models-monorepo/tree/develop/.github/workflows), 
[API](https://github.com/RogerVFbr/fiap-api/tree/develop/.github/workflows).