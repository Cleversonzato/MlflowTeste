""" Classe desnecessária, para exemplo de uso"""
import mlflow
from sklearn.linear_model import LinearRegression
from codigos.util.modelo import ModeloAbstrato, metricas_erro
from mlflow.models.signature import infer_signature


class SklearnLinearRegression(ModeloAbstrato):
    def __init__(self):
        self.modelo = LinearRegression()
        self.ultima_estimativa = None
        self.ultima_avaliacao = None
        self.assinatura_modelo = None


    def definir_schema(self, treino_x):
        self.assinatura_modelo = infer_signature(treino_x, self.modelo.predict(treino_x))


    def treinar(self, treino_x, treino_y):
        self.modelo.fit(treino_x, treino_y)
        self.definir_schema(treino_x)


    def estimar(self, teste_x):
         self.ultima_estimativa = self.modelo.predict(teste_x)


    def avaliar(self, esperado):
        self.ultima_avaliacao = metricas_erro(esperado, self.ultima_estimativa)


    def registrar(self):
        (mae, rmse, r2) = self.ultima_avaliacao

        print("LinearRegression:")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
  
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.set_tag("release.version", "teste?")

        #Se alterar a versão do modelo:
        mlflow.sklearn.log_model( self.modelo, artifact_path='VoltaDaLinha', registered_model_name="VoltaDaLinha", signature=self.assinatura_modelo )


