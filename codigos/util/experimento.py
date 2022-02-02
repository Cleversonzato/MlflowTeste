import argparse
import mlflow
from configuracoes.definicoes import MLFLOW_TRACKING_URI
from codigos.util.modelo import ModeloAbstrato
from codigos.util.dados import DadosMLAbstrato


def definir_argumentos_entrada(argumentos: list ) -> argparse.ArgumentParser:
    """ 
    Lista de um iterador (lista ou tupla) com informações na seguinte ordem: 
        - nome do argumento
        - tipo do argumento
        - valor padrão para o argumento
    que serão lidas na chamada do programa/script

    Ex: 
        definir_argumentos( [
            ( "--alfa", float, 0.5 ),            
            ( "--random_stats", integer, 42 )
        ])
    
    """
    parser=argparse.ArgumentParser()

    for arg in argumentos:
        parser.add_argument(arg[0], type=arg[1], default=arg[2], )

    return parser.parse_known_args()[0]


def experimentar(nome_experimento:str, modelo: ModeloAbstrato, dados: DadosMLAbstrato, run_name:None,):
    """ Função para fazer um experimento com um modelo"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experimento = mlflow.set_experiment(nome_experimento)
    print("Experimento: " + experimento.name )

    with mlflow.start_run(run_name=run_name):
        modelo.treinar(dados.treino_x, dados.treino_y)
        modelo.estimar(dados.teste_x)        
        modelo.avaliar(dados.teste_y)
        modelo.registrar()


def experimentar_funcional(nome_experimento:str, dados: DadosMLAbstrato, run_name:None, funcoes_ml: callable, *args, **kwargs):
    """ 
        Função para fazer um experimento com um modelo, mas com um paradigman mais funcional.
        basicamente encapsula a funcoes_ml dentro do mlflow run e passa DadosMLAbstrato para ela
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experimento = mlflow.set_experiment(nome_experimento)
    print("Experimento: " + experimento.name )

    with mlflow.start_run(run_name = run_name):
        funcoes_ml(dados, *args, **kwargs)
