# TELECOM-X-PT2
Segunda parte do Challenge da Oracle One

Contexto do Desafio

•	Objetivo: Construir e comparar modelos de classificação para prever churn.
•	Principais tarefas:
o	Preparar os dados (tratamento de dados faltantes, codificação categórica, normalização).
o	Análise exploratória e seleção de variáveis.
o	Treinamento de pelo menos dois modelos (ex. regressão logística, random forest, k-NN).
o	Avaliação dos modelos via métricas (acurácia, precisão, recall, F1, matriz de confusão).
o	Interpretação da importância das variáveis e insights decisórios.
Descrição e Estrutura dos Dados
•	Base com 7.267 linhas e 6 colunas inicialmente.
•	Colunas principais:
o	customerID: identificador único (removido).
o	Churn: variável alvo (sim/não).
o	Quatro colunas categóricas contendo dicionários em formato string (como JSON) agrupando informações de cliente, telefone, internet e conta.
•	Após limpeza, o dataset fica com 5 colunas (sem customerID), e alvo convertido em binário (1 para churn "Yes", 0 para não churn "No").
Limpeza e Pré-processamento
•	Remoção de customerID e conversão de colunas numéricas de texto (como ‘TotalCharges’).
•	Eliminação de registros com valores ausentes.
•	Conversão da variável Churn para binária.
•	Identificação das colunas categóricas (todas as features restantes são categóricas com dados no formato string que representam dicionários).
•	Tratamento de dados categóricos para evitar DataFrame vazio (substituição de valores ausentes por "desconhecido" e garantia de que colunas são do tipo string).
Desafio: Tratamento das Colunas com Dicionários Embutidos
•	As colunas categóricas customer, phone, internet e account possuem strings que representam dicionários JSON.
•	Para modelagem, é necessário extrair as variáveis internas desses dicionários e desdobrá-las em colunas separadas.
•	Essas variáveis internas trazem características importantes como:
o	Gênero, SeniorCitizen, Partner, Dependents, tenure (cliente).
o	PhoneService, MultipleLines (telefone).
o	InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies (internet).
o	Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges (conta).
 
Passos recomendados para dar continuidade:
1.	Parsing das colunas JSON (strings de dicionários)
o	Aplicar json.loads() ou ast.literal_eval() para converter strings em dicionários.
o	Expandir os dicionários em colunas separadas para cada feature contida.
2.	Conversão dos tipos de dados:
o	Converter as novas colunas para tipos adequados (int, float, category).
o	Tratar valores ausentes se existirem.
3.	Codificação das variáveis categóricas:
o	Utilizar OneHotEncoder ou ordinal encoding conforme necessário.
4.	Normalização dos dados numéricos:
o	Por exemplo, aplicar StandardScaler para variáveis como tenure, MonthlyCharges, TotalCharges.
5.	Divisão entre treino e teste:
o	Fazer split após o pré-processamento completo.
6.	Treinamento de modelos preditivos:
o	Testar pelo menos dois modelos, por exemplo, Regressão Logística e Random Forest.
o	Usar técnicas para lidar com desequilíbrio, como SMOTE, visto que a variável alvo pode ser desbalanceada.
7.	Avaliação dos modelos e interpretação:
o	Métricas: acurácia, recall (importante para churn), precision, F1 score.
o	Extrair importância das variáveis dos modelos (feature importance, coeficientes).
Resumo
O notebook baseia-se num dataset de churn contendo dados em formatos complexos (dicionários em colunas), demandando um processo robusto de:
•	Extração e transformação das colunas internas;
•	Codificação e normalização;
•	Treinamento e avaliação de múltiplos modelos;
•	Análise dos fatores que mais impactam a evasão.

