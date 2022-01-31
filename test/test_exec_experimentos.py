from experimentos.RedeElastica import executar_teste as teste_elastico
from experimentos.VoltaDaLinha import executar_teste as teste_linear


def test_experimento_linear():
    teste_linear()


def test_teste_elastico():
    teste_elastico()