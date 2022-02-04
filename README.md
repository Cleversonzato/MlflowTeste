# Teste de uso do MLflow

 - Atualizar o ambiente com as novas versões, futuramente 

 - ambiente:
```
python3 -m venv '.mlflow'
```
```
source .mlflow/bin/activate
```
```
pip install -r requirements.txt
```

 - ui do mlflow 
 ```
mlflow ui --host 0.0.0.0 --port 5000
# http://localhost:5000/
 ```

 - para executar os experimentos, executar no root
 ```
 python3 exec.py RedeElastica
 #ou 
 python3 exec.py VoltaDaLinha
 #ou
 python3 exec.py DoPycaret
 ```
