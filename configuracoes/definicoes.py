import os
from pathlib import Path

PASTA_DADOS = pasta_dados_tratados = str(Path(os.path.dirname(os.path.realpath(__file__))).parent) + "/dados/"
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI') or "sqlite:///mlruns/sqlitle.db"