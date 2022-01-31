from experimentos.RedeElastica import executar_teste as executar_elastico
from experimentos.VoltaDaLinha import executar_teste as executar_linear

def test_rodar_ui():
    import os
    os.system('mlflow ui --host 0.0.0.0 --port 5000 &')


def test_encerar_ui():
    import os
    os.system("killall gunicorn")
   

def test_experimento_linear():
    executar_linear()


def test_teste_elastico():
    executar_elastico()