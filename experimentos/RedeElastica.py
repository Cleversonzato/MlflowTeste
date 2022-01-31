import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse


def argumentos():
    parser=argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--tam_treino', default=0.7, type=float)
    parser.add_argument('--random_state_amostra', default=1, type=int)
    parser.add_argument('--random_state_modelo', default=42, type=int)
    parser.add_argument('--l1', default=0.5, type=float)
    args = parser.parse_args()

    return  args.tam_treino, args.random_state_amostra, args.alpha, args.l1, args.random_state_modelo


def metricas(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


def preparar_dados(tam_treino, random_state_amostra):
    pasta_dados_tratados = str(Path(os.path.dirname(os.path.realpath(__file__))).parent) + "/dados/tratados/"

    dados = pd.read_csv(pasta_dados_tratados + 'dados_prontos.csv')
    treino, teste = train_test_split(dados, train_size=tam_treino, random_state=random_state_amostra)
    treino_x = treino[["Review Date","Cocoa Percent"]]
    teste_x = teste[["Review Date","Cocoa Percent"]]
    treino_y = treino["Rating"]
    teste_y = teste["Rating"]    

    return treino_x, teste_x, treino_y, teste_y


def logs_resultados(alpha, l1, rmse, mae, r2):
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)


def executar_teste(tam_treino=0.7, random_state_amostra=42, alpha=0.5, l1=0.5, random_state_modelo=42):
    #dados
    (treino_x, teste_x, treino_y, teste_y) = preparar_dados(tam_treino, random_state_amostra)

    with mlflow.start_run():
        en = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=random_state_modelo)
        en.fit(treino_x, treino_y)

        previsoes = en.predict(teste_x)

        (rmse, mae, r2) = metricas(teste_y, previsoes)

        logs_resultados(alpha, l1, rmse, mae, r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(en, "model", registered_model_name="RedeElastica sem store?")
        else:
            mlflow.sklearn.log_model(en, "model")


#Em execuções pelo cli, para pegar os parâmetros
if __name__ == "__main__":
    (tam_treino, random_state_amostra, alpha, l1, random_state_modelo) = argumentos()
    executar_teste(tam_treino, random_state_amostra, alpha, l1, random_state_modelo)