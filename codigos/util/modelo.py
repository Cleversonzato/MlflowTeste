import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from abc import ABC, abstractmethod

class ModeloAbstrato(ABC):
    """ Classe abstrata para os modelos"""

    @abstractmethod
    def treinar(self, treino_x, treino_y):
        """ 
            Método definindo como é feito o treinamento do modelo             
        """
        ...


    @abstractmethod
    def estimar(self, teste_x):        
        """ 
            Método definindo como é feito o uso do modelo em dados não treinados (esimativa, predição...) 
        """
        ...
        

    @abstractmethod
    def avaliar(esperado, previsto):
        """ 
            Método definindo como é feito a avaliação do modelo
        """
        ...

    
    @abstractmethod
    def registrar():
        """ 
            Método definindo o que deve se logado, salvo ou registrado pelo mlflow, assim como qualquer outro procedimento ao final do experimento
        """
        ...





### Funções uteis para os modelos


def metricas_erro(esperado, previsto):
    """ 
    Algumas métricas de erro padrões, respectivametne: 
        - Média do erro absoluto (MAE)
        - Raiz quadrada da média do erro (RMSE)
        - Coeficiente de determinação (R2)
    """
    mae = mean_absolute_error(esperado, previsto)
    rmse = np.sqrt(mean_squared_error(esperado, previsto))    
    r2 = r2_score(esperado, previsto)

    return mae, rmse, r2