import pandas as pd
import pycaret.regression as rs
from configuracoes.definicoes import PASTA_DADOS
from codigos.util.experimento import pycaret_experimentar_modelos


def executar_experimentos_pycaret(tam_treino=0.7, nome_experimento='Teste Pycaret'):
    dados =  pd.read_csv(PASTA_DADOS + 'tratados/dados_prontos.csv',  index_col=0)

    melhores = pycaret_experimentar_modelos(
        rs,
        num_modelos=2,
        data=dados,
        target='Rating',        
        train_size = tam_treino,
        log_experiment = True,
        experiment_name= nome_experimento,
        silent=True
    )

