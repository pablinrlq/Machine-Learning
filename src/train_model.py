import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# 1. gerar os dados falsos de temperatura e venda (Simulando o ambiente do Gelato Mágico)
# aqui vou criar um dicionario mais simples
data = {
    'Temperatura': [20, 25, 30, 35, 22, 28, 40, 18, 26, 33],
    'Vendas': [200, 250, 320, 400, 210, 300, 450, 180, 270, 380]
}

df = pd.DataFrame(data)

# aqui vai so salvar como csv pra simular o arquivo de entrada
df.to_csv('inputs/sorvetes_sales.csv', index=False)
print("Dados gerados e salvos em inputs/sorvetes_sales.csv")

# 2. preparação dos Dados
X = df[['Temperatura']] # Variável independente (Features)
y = df['Vendas']        # Variável dependente (Target)

# dividir em treino e teste pra ficar bonitão
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Configuração do MLflow
mlflow.set_experiment("Gelato_Magico_Sales_Prediction")

with mlflow.start_run():
    # 4. treinamento do modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. previsões e métricas
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MSE: {mse}")
    print(f"R²: {r2}")

    # 6. faz as logs no MLflow
    # registra os parâmetros do modelo (coeficientes)
    mlflow.log_param("coeficiente_angular", model.coef_[0])
    mlflow.log_param("intercepto", model.intercept_)
    
    # registra as metricas de erro 
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    
    # vai registrar o modelo
    mlflow.sklearn.log_model(model, "modelo_sorvete")
    
    # 7. aqui vai gerar o grafico pra por no readme 
    plt.scatter(X, y, color='blue', label='Dados Reais')
    plt.plot(X, model.predict(X), color='red', label='Linha de Regressão')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Vendas de Sorvete')
    plt.legend()
    plt.title('Previsão de Vendas: Gelato Mágico')
    plt.savefig('grafico_previsao.png')
    print("Gráfico salvo como grafico_previsao.png")

print("Treinamento concluído e registrado no MLflow!")