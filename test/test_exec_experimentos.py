from codigos.experimentos.RedeElastica import executar_experimento_rede_elastica
from codigos.experimentos.VoltaDaLinha import executar_experimento_volta_da_linha

def test_rodar_ui():
    """ Necessário alterar a configuração aqui se houver alterações nas configurações """
    import os
    os.system('mlflow ui --host 0.0.0.0 --port 5000 --default-artifact-root mlruns --backend-store-uri sqlite:///mlruns/sqlitle.db &')


def test_encerar_ui():
    import os
    os.system("killall gunicorn")
   

def test_experimento_linear():
    executar_experimento_volta_da_linha()


def test_teste_elastico():
    executar_experimento_rede_elastica()