import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


def separar_treino_teste(
        dados: pd.DataFrame,
        alvo: str,
        percentual_treino: float,
        random_state_amostra: int
    ):
    """ Separa um dataframe dados em treino e teste de acordo com os parâmetros utilizados. Nada muito especial""" 
    treino, teste = train_test_split(dados, train_size=percentual_treino, random_state=random_state_amostra)
    treino_x = treino.drop(alvo, axis=1)
    teste_x = teste.drop(alvo, axis=1)
    treino_y = treino[alvo]
    teste_y = teste[alvo]    

    return treino_x, teste_x, treino_y, teste_y


class DadosMLAbstrato(ABC):
    """ 
        Classe abstrata para os dados de treino e teste. 
        Bascimente é uma classe com as variáveis de treino e teste (x) e seus "alvos" (y) separados
    """  

    @abstractmethod
    def treino_x(self):
        """" Propriedade. Variáveis de treino"""
        ...

    @abstractmethod
    def teste_x(self):
        """" Propriedade. Variáveis de teste"""
        ...

    @abstractmethod
    def treino_y(self):
        """" Propriedade. Alvo do treino"""
        ...

    @abstractmethod
    def teste_y(self):
        """" Propriedade. Alvo do teste"""
        ...



class DadosCSV(DadosMLAbstrato):
    """ Classe para simplificar o split de dados em treino e teste de um csv"""
    def __init__( self,
            pasta_csv: str,
            alvo: str,
            percentual_treino: float = 0.75,
            random_state_amostra: int = 42,
            index_col=0
        ):
        df = pd.read_csv(pasta_csv,  index_col=index_col)
        ( self.treino_x, self.teste_x, self.treino_y, self.teste_y ) = separar_treino_teste( df, alvo, percentual_treino, random_state_amostra)
            
    def treino_x(self):
        return self.treino_x

    def teste_x(self):
        return self.teste_x

    def treino_y(self):
        return self.treino_y

    def teste_y(self):
        return self.teste_y