<div align="center">

# Gelato Mágico

**Previsão de vendas de sorvete pela temperatura, com regressão linear e MLflow.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)

<img src="grafico_previsao.png" alt="Dispersão de vendas por temperatura com a reta de regressão" width="560"/>

</div>

## O problema

A sorveteria Gelato Mágico produz todo dia sem saber quanto vai vender. Em dia frio sobra
sorvete; em dia quente falta. A ideia é usar a previsão do tempo para estimar as vendas
e planejar a produção.

## A abordagem

Vendas e temperatura têm uma relação claramente linear: quanto mais quente, mais sorvete.
Por isso escolhi uma **regressão linear simples** em vez de um modelo mais complexo.
É fácil de explicar para o dono da sorveteria e funciona bem para esse caso.

1. **Dados:** `train_model.py` gera um conjunto de exemplo e salva em `inputs/sorvetes_sales.csv`
2. **Treino:** separa 80% para treino e 20% para teste e ajusta a reta `vendas = a × temperatura + b`
3. **Registro:** o MLflow guarda coeficientes, métricas (MSE e R²) e o próprio modelo a cada execução
4. **Previsão:** `predict.py` carrega o último modelo registrado e estima as vendas para a temperatura informada

## Rodando

```bash
git clone https://github.com/pablinrlq/Machine-Learning.git
cd Machine-Learning
pip install -r requirements.txt

python src/train_model.py    # treina, registra no MLflow e gera o gráfico
python src/predict.py        # pede a temperatura e mostra a previsão
mlflow ui                    # opcional: compara as execuções em http://localhost:5000
```

## Próximos passos

- Treinar com dados reais de vendas em vez do conjunto de exemplo
- Incluir outras variáveis, como dia da semana e feriados, e comparar com uma regressão múltipla
- Expor a previsão numa API pequena que consulta a previsão do tempo sozinha

---

<div align="center">
<sub>Feito por <a href="https://github.com/pablinrlq">Pablo Moises</a> · desafio do bootcamp de Machine Learning da DIO</sub>
</div>
