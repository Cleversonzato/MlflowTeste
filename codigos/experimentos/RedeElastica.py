from codigos.modelos.SklearnElasticNet import sklearn_ElasticNet_ML
from configuracoes.definicoes import PASTA_DADOS
from codigos.util.dados import DadosCSV
from codigos.util.experimento import experimentar_funcional


def executar_experimento_rede_elastica(tam_treino=0.7, random_state_amostra=42, alpha=0.5, l1=0.5, random_state_modelo=42):
    dados = DadosCSV( 
        pasta_csv= PASTA_DADOS + 'tratados/dados_prontos.csv', 
        alvo='Rating', 
        random_state_amostra=random_state_amostra, 
        percentual_treino=tam_treino
    )

    experimentar_funcional(
        "Experimento Rede Elástica",
        dados, 
        sklearn_ElasticNet_ML,
        alpha=alpha,
        l1=l1,
        random_state_modelo=random_state_modelo
    )
