import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse


pasta_dados_tratados = '../dados/tratados/'


def argumentos():
    parser=argparse.ArgumentParser()
    parser.add_argument('--alpha', defalt=0.5, type=float)
    parser.add_argument('--tam_treino', defalt=0.7, type=float)
    parser.add_argument('--random_state_amostra', defalt=1, type=int)
    parser.add_argument('--random_state_modelo', defalt=42, type=int)
    parser.add_argument('--l1', defalt=0.5, type=float)
    args = parser.parse_args()

    return args.apha, args.tam_treino, args.random_state_amostra, args.random_state_modelo, args.l1


def metricas(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


def preparar_dados(tam_treino, random_state_amostra):
    dados = pd.read_csv(pasta_dados_tratados + 'treino.csv')
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


def executar_teste(alpha, tam_treino, random_state_amostra, random_state_modelo, l1):
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
            mlflow.sklearn.log_model(en, "model", registered_model_name="RedeElastica")
        else:
            mlflow.sklearn.log_model(en, "model")


#Em execuções pelo cli, para pegar os parâmetros
if __name__ == "__main__":         
    executar_teste( 
        argumentos() 
    )