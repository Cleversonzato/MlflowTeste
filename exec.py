import sys
from codigos.experimentos.RedeElastica import executar_experimento_rede_elastica
from codigos.experimentos.VoltaDaLinha import executar_experimento_volta_da_linha
from codigos.experimentos.DoPycaret import executar_experimentos_pycaret
from codigos.util.experimento import definir_argumentos_entrada


#Em execuções pelo cli, para pegar os parâmetros
if __name__ == "__main__":
    if sys.argv[1] == "RedeElastica":
        args = definir_argumentos_entrada([
            ("--tam_treino", float, 0.7),        
            ("--random_state_amostra", float, 42),
            ("--alpha", float, 0.5),        
            ("--l1", float, 0.5),
            ("--random_state_modelo", float, 42)
        ])

        executar_experimento_rede_elastica( args.tam_treino, args.random_state_amostra, args.alpha, args.l1, args.random_state_modelo )
  
    else:
        if sys.argv[1] == "VoltaDaLinha":
            args = definir_argumentos_entrada([
                ("--tam_treino", float, 0.7),        
                ("--random_state_amostra", float, 42)
            ])

            executar_experimento_volta_da_linha( args.tam_treino, args.random_state_amostra )


        else:
            if sys.argv[1] == "DoPycaret":
                args = definir_argumentos_entrada([
                        ("--tam_treino", float, 0.7),
                        ("--nome_experimento", str, "Teste Pycaret")
                    ])

                executar_experimentos_pycaret( args.tam_treino, args.nome_experimento )


