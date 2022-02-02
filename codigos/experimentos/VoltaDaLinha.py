from codigos.modelos.SklearnLinearRegressionModel import SklearnLinearRegression
from configuracoes.definicoes import PASTA_DADOS
from codigos.util.dados import DadosCSV
from codigos.util.experimento import experimentar


def executar_experimento_volta_da_linha(tam_treino=0.7, random_state_amostra=42):
    dados = DadosCSV( 
        pasta_csv=  PASTA_DADOS + 'tratados/dados_prontos.csv', 
        alvo='Rating', 
        random_state_amostra=random_state_amostra, 
        percentual_treino=tam_treino
    )

    experimentar("Experimento Volta da Linha", SklearnLinearRegression(), dados)