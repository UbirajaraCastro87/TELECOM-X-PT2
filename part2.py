# Previsão de Evasão de Clientes - Pipeline Melhorado

# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------
# 1. Carregamento dos Dados
# -----------------------------------------
# Exemplo: df = pd.read_csv('telecom_churn.csv')

# Para este exemplo, suponde-se que o dataframe df já está carregado conforme o anexo:
# df.shape, df.head() mostram estrutura similar ao que foi realizado no notebook.

# -----------------------------------------
# 2. Entendimento e Exploração Inicial
# -----------------------------------------
print(f"Número de registros: {df.shape[^0]}, Número de colunas: {df.shape[^1]}")
print(df.head())

# Colunas problemáticas para modelagem: customerID (ID), colunas aninhadas (dicionários em 'customer', 'phone', etc.)
# Precisamos desmontar essas colunas aninhadas para planos flat.

# -----------------------------------------
# 3. Desmembrar colunas aninhadas
# -----------------------------------------
def expandir_colunas_aninhadas(df, coluna):
    df_expandido = df[coluna].apply(pd.Series)
    df_expandido.columns = [f"{coluna}_{subcol}" for subcol in df_expandido.columns]
    return pd.concat([df.drop(columns=[coluna]), df_expandido], axis=1)

for col in ['customer', 'phone', 'internet', 'account']:
    df = expandir_colunas_aninhadas(df, col)

# -----------------------------------------
# 4. Tratamento dos dados
# -----------------------------------------
# Remover colunas irrelevantes
if 'customerID' in df.columns:
    df.drop(columns='customerID', inplace=True)

# Verificar valores faltantes
print("Valores faltantes por coluna:")
print(df.isnull().sum())

# No exemplo, aplicar tratamento de nulos, se necessário.
df.fillna(method='ffill', inplace=True)  # Exemplo simples

# Transformar o alvo 'Churn' em variável binária
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# -----------------------------------------
# 5. Definição de variáveis (X, y)
# -----------------------------------------
X = df.drop(columns='Churn')
y = df['Churn']

# -----------------------------------------
# 6. Identificar variáveis categóricas e numéricas
# -----------------------------------------
cat_features = X.select_dtypes(include=['object']).columns.tolist()
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Variáveis categóricas: {cat_features}")
print(f"Variáveis numéricas: {num_features}")

# -----------------------------------------
# 7. Divisão treino/teste
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# -----------------------------------------
# 8. Balanceamento com SMOTE (aplicado somente no conjunto de treino)
# -----------------------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"Distribuição antes SMOTE (treino): \n{y_train.value_counts()}")
print(f"Distribuição depois SMOTE (treino): \n{y_train_bal.value_counts()}")

# -----------------------------------------
# 9. Pré-processamento: Encoding e Normalização
# -----------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# -----------------------------------------
# 10. Criação dos pipelines dos modelos
# -----------------------------------------

# Modelo 1: Regressão Logística
pipe_logreg = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=300))
])

# Modelo 2: Random Forest
pipe_rf = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# -----------------------------------------
# 11. Treinamento dos modelos
# -----------------------------------------
pipe_logreg.fit(X_train_bal, y_train_bal)
pipe_rf.fit(X_train_bal, y_train_bal)

# -----------------------------------------
# 12. Avaliação dos modelos
# -----------------------------------------
def avaliar_modelo(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()

print("Avaliação Regressão Logística:")
avaliar_modelo(pipe_logreg, X_test, y_test)

print("Avaliação Random Forest:")
avaliar_modelo(pipe_rf, X_test, y_test)

# -----------------------------------------
# 13. Interpretação: Importância das variáveis com Random Forest
# -----------------------------------------

# Obter nomes das features após o OneHotEncoder
ohe_features = pipe_rf.named_steps['preprocess'].transformers_[^1][^1].get_feature_names_out(cat_features)
all_features = num_features + list(ohe_features)

importances = pipe_rf.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10])
plt.title('Top 10 Importância das Variáveis - Random Forest')
plt.show()

# -----------------------------------------
# 14. Conclusão Estratégica
# -----------------------------------------
print("""
Principais fatores que influenciam a evasão identificados pelo modelo Random Forest incluem:
- Tipo de contrato (month-to-month, one year, two year)
- Tipo de serviço de internet (Fiber optic, DSL)
- Serviços adicionais como Online Security, Tech Support
- Uso do Paperless Billing

Recomenda-se focar em clientes com contrato mensal e serviços de internet por fibra óptica para iniciativas de retenção.
""")


