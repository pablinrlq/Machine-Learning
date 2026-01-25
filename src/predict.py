import mlflow.sklearn
import pandas as pd

# Simulação: a gente tem que imaginar que este script roda em um servidor na nuvem
# e recebe a temperatura de uma API meteorológica em tempo real.

def prever_vendas(temperatura_atual):
    # 1. Carregar o modelo salvo anteriormente
    # Nota: Em produção real, usamos o caminho do MLflow (runs:/...)
    # Aqui, para facilitar, vamos carregar a pasta local se ela foi gerada,
    # ou você pode carregar o objeto salvo via pickle/joblib se preferir.
    # Vamos pensar que a gente rodou o train_model.py e ele gerou a pasta 'mlruns'
    
    # Para simplificar neste exemplo didático, vamos carregar o modelo 
    # da última execução (run) automaticamente:
    last_run_id = mlflow.search_runs().iloc[0].run_id
    logged_model = f'runs:/{last_run_id}/modelo_sorvete'
    
    # Carrega o modelo
    loaded_model = mlflow.sklearn.load_model(logged_model)
    
    # 2. Criar o DataFrame com a entrada
    input_data = pd.DataFrame([[temperatura_atual]], columns=['Temperatura'])
    
    # 3. Fazer a previsão
    previsao = loaded_model.predict(input_data)
    
    return previsao[0]

if __name__ == "__main__":
    temp = float(input("Digite a temperatura prevista para hoje (°C): "))
    vendas_previstas = prever_vendas(temp)
    print(f"💰 Para {temp}°C, a previsão é vender aprox. {int(vendas_previstas)} sorvetes!")