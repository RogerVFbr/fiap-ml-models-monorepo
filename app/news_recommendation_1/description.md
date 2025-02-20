Dataset Description
Dados
O conjunto de dados para este desafio foi dividido em treino, validação e teste. Como esperado, os usuários não terão acesso ao conjunto de teste. Este fica para gerar o ranking do Kaggle.

Antes de começar é bom deixar claro que: todos os usuários que estão no teste e validação também estão no treino. nem todos que aparecem no treino estão no teste/validação.

Conjunto de Treino
O conjunto de treino está disponibilizado em diferentes pastas, cada uma contendo informação complementar. Os arquivos treino_parte_X.csv, onde X é um valor de 1 até 6, consistem das colunas:

userId: id do usuário
userType: usuário logado ou anônimo
HistorySize: quantidade de notícias lidas pelo usuário
history: lista de notícias visitadas pelo usuário
TimestampHistory: momento em que o usuário visitou a página
timeOnPageHistory: quantidade de ms em que o usuário ficou na página
numberOfClicksHistory: quantidade de clicks na matéria
scrollPercentageHistory: quanto o usuário visualizou da matéria
pageVisitsCountHistory: quantidade de vezes que o usuário visitou a matéria

Além desses arquivos, a pasta de treino contém uma subpasta denominada de itens. Esta pasta tem a seguinte informação:

Page: id da matéria. Esse é o mesmo id que aparece na coluna history de antes.
Url: url da matéria
Issued: data em que a matéria foi criada
Modified: última data em que a matéria foi modificada
Title: título da matéria
Body: corpo da matéria
Caption: subtítulo da matéria

Este conjunto de treino consiste de dados de usuários reais da Globoplay. Eles forem coletados até uma data limite T (a maior data em todo o conjunto TimestampHistory).

Conjunto de Validação
Capturando informações de um período posterior ao treino, ou seja não existe sobreposição temporal com o treino, o conjunto de validação consiste de:

userId: id do usuário
userType: usuário logado ou anônimo
history: lista de páginas que devem ser recomendadas ao usuário

O seu objetivo é gerar um RANKING para a coluna history. Ou seja, quando um usuário loga, prever quais serão os próximos acessos do mesmo. Observe que o mesmo userId está na validação quanto no treino.