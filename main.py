import pandas as pd
import os
import datetime
import logging

from src.loader import cargar_datos
from src.features import feature_engineering_lag, feature_engineering_min_max, feature_engineering_deltas, feature_engineering_medias_moviles, feature_engineering_cum_sum,         feature_engineering_ratios


## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
monbre_log = f"log_{fecha}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{monbre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

## Funcion principal
def main():
    logger.info("Inicio de ejecucion.")

    #00 Cargar datos
    os.makedirs("data", exist_ok=True)
    path = "data/competencia_01.csv"
    df = cargar_datos(path)   

    #01 Feature Engineering
    atributos_fe_lag = ["ctrx_quarter"] 
    atributo_fe_deltas = ["ctrx_quarter"]
    atributos_fe_medias_moviles = ["ctrx_quarter"]
    atributos_cum_sum = ["Master_mlimitecompra", "Visa_mlimitecompra"]
    atributos_min_max = ["Master_mlimitecompra", "Visa_mlimitecompra"]
    ratio_pairs = [
    ("Master_msaldototal", "Master_ctrx"),
    ("Visa_msaldototal", "Visa_ctrx"),
    ("Master_msaldototal", "Visa_msaldototal")
]
    

    cant_lag = 2
    df = feature_engineering_lag(df, columnas=atributos_fe_lag, cant_lag=cant_lag)
    df = feature_engineering_deltas(df, columnas=atributo_fe_deltas, cant_lag=cant_lag)
    df = feature_engineering_medias_moviles(df, columnas=atributos_fe_medias_moviles, cant_lag=cant_lag)
    df = feature_engineering_cum_sum(df, columnas=atributos_cum_sum)
    df = feature_engineering_min_max(df, columnas=atributos_min_max)
    df = feature_engineering_ratios(df, ratio_pairs=ratio_pairs)
  
    #02 Guardar datos
    path = "data/competencia_01_fe.csv"
    df.to_csv(path, index=False)
  
    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.{monbre_log}")

if __name__ == "__main__":
    main()