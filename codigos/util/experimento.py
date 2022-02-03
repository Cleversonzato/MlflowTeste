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


def preprar_experimento(nome_experimento):
    """ Função para definir um experimento"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experimento = mlflow.set_experiment(nome_experimento)
    print("Experimento: " + experimento.name )


def experimentar(nome_experimento:str, modelo: ModeloAbstrato, dados: DadosMLAbstrato, run_name=None):
    """ Função para fazer um experimento com um modelo"""
    preprar_experimento(nome_experimento)

    modelo.treinar(dados.treino_x, dados.treino_y)
    modelo.estimar(dados.teste_x)
    modelo.avaliar(dados.teste_y)

    with mlflow.start_run(run_name=run_name):
        modelo.registrar()


def experimentar_funcional(nome_experimento:str, dados: DadosMLAbstrato, funcoes_ml: callable, run_name=None, *args, **kwargs):
    """ 
        Função para fazer um experimento com um modelo, mas com um paradigman mais funcional.
        basicamente encapsula a funcoes_ml dentro do mlflow run e passa DadosMLAbstrato para ela
    """
    preprar_experimento(nome_experimento)

    with mlflow.start_run(run_name = run_name):
        funcoes_ml(dados, *args, **kwargs)


def pycaret_setup(definicoesML, experiment_name, *args, **kwargs):
    """ 
    Faz o setup do pycaret com o ambiente do MLflow
    - 'definicoesML' se refere ao tipo de procedimento utilizado no setup (pycaret.regression ou pycaret.classication, por exemplo)
    - experiment_name é obrigatório e também se refere ao argumento que é utilizado no setup do pycaret 
    No mais, passe todos os argumentos que normalmente passaria ao Setup do pycaret
    

    Exemplo:
        import pycaret.regression as rs
        pycaret_setup(
            rs,
            data=dados,
            target='Rating',        
            train_size = tam_treino,
            log_experiment = True,
            experiment_name= "Experimento Teste",
            silent=True
        )   

    """
    preprar_experimento(experiment_name)
    definicoesML.setup(experiment_name=experiment_name, *args, **kwargs)


def pycaret_experimentar_modelos(definicoesML, experiment_name, num_modelos=1, *args, **kwargs):
    """
    Faz o setup do pycaret com o ambiente do MLflow e retorna os "num_modelos" melhores modelos da comparação de perfermance entre eles
    - 'definicoesML' se refere ao tipo de procedimento utilizado no setup (pycaret.regression ou pycaret.classication, por exemplo) 
    - experiment_name é obrigatório e também se refere ao argumento que é utilizado no setup do pycaret 
    No mais, passe todos os argumentos que normalmente passaria ao Setup do pycaret

    Exemplo:
        import pycaret.regression as rs
        pycaret_experimentar_modelos(
            rs,
            experiment_name= "Experimento Teste",
            data=dados,
            target='Rating',        
            train_size = tam_treino,
            log_experiment = True,           
            silent=True
        )   
    """
    pycaret_setup(definicoesML, experiment_name, *args, **kwargs) 
    melhores = definicoesML.compare_models(n_select = num_modelos)

    print( """
    Resultados:""" )
    print( definicoesML.pull() )
    
    return melhores