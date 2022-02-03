""" Classe mais para exemplo de uso"""
import mlflow
from sklearn.linear_model import ElasticNet
from codigos.util.modelo import metricas_erro
from mlflow.types.schema import Schema, ColSpec 
from mlflow.models.signature import ModelSignature

def treinar(modelo, variaveis, alvo):
    return modelo.fit(variaveis, alvo)


def estimar(modelo, teste_x):
    return modelo.predict(teste_x)


def avaliar(esperado, previsto):
    return metricas_erro(esperado, previsto) 
     

def assinatura_modelo():
    input_schema = Schema([
        ColSpec("integer", "Review Date"),
        ColSpec("float", "Cocoa Percent")
    ])
    output_schema = Schema([ColSpec("long", "Rating")])
    
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def sklearn_ElasticNet_ML(dados, alpha: float, l1:float, random_state_modelo:int):
    modelo = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=random_state_modelo)
    modelo_treinado = treinar(modelo, dados.treino_x, dados.treino_y)    
    estimativas = estimar(modelo_treinado, dados.teste_x)   
    (mae, rmse, r2) = avaliar(estimativas, dados.teste_y)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.set_tag("release.version", "teste!")


    #Se alterar a versão do modelo:
    info_modelo = mlflow.sklearn.log_model(
        modelo,
        artifact_path='RedeElastica',
        registered_model_name="RedeElastica",
        signature=assinatura_modelo(),
        pip_requirements=["-r requirements.txt"] #como se fosse um pip install -> []
       )

    # Abaixo, 
    # - Necessita do pip install SHAP
    # - Incompatível com pycaret:
    #           ERROR: pycaret 2.3.6 has requirement scikit-learn==0.23.2, but you'll have scikit-learn 0.24.0 which is incompatible.

    df_eval = dados.teste_x
    df_eval['Rating'] = dados.teste_y
    df_eval['Review Date'] = df_eval['Review Date'].astype('int32')
    df_eval['Cocoa Percent'] = df_eval['Cocoa Percent'].astype('float32')

    #este aparece no run
    result = mlflow.evaluate(
        model = info_modelo.model_uri,
        data = df_eval,
        targets = 'Rating',
        model_type="regressor", # ou classifier
        dataset_name="dados de teste",
        evaluators=["default"],
    )

   