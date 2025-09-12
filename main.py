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
    atributos = [
        "mrentabilidad",
        "mcomisiones",
        "mpasivos_margen",
        "mcaja_ahorro",
        "mcaja_ahorro_dolares",
        "mcuentas_saldo",
        "ctarjeta_debito_transacciones",
        "mautoservicio",
        "ctarjeta_visa_transacciones",
        "mtarjeta_visa_consumo",
        "ctarjeta_master_transacciones",
        "mtarjeta_master_consumo",
        "mprestamos_personales",
        "mprestamos_prendarios",
        "mprestamos_hipotecarios",
        "mplazo_fijo_dolares",
        "mplazo_fijo_pesos",
        "cpayroll_trx",
        "cpayroll2_trx",
        "mcomisiones_mantenimiento",
        "ctrx_quarter",
        "Master_mlimitecompra",
        "Master_mconsumototal",
        "Visa_mlimitecompra",
        "Visa_mconsumototal"
    ]

    atributos_fe_lag = atributos 
    atributo_fe_deltas = atributos
    atributos_fe_medias_moviles = atributos
    atributos_cum_sum = atributos
    atributos_min_max = atributos

    ratio_pairs = [
        # Uso de tarjetas relativo al límite disponible
        ("Master_mconsumototal", "Master_mlimitecompra"),
        ("Visa_mconsumototal", "Visa_mlimitecompra"),

        # Consumo mensual relativo al límite
        ("mtarjeta_master_consumo", "Master_mlimitecompra"),
        ("mtarjeta_visa_consumo", "Visa_mlimitecompra"),

        # Préstamos en relación al saldo de cuentas
        ("mprestamos_personales", "mcuentas_saldo"),
        ("mprestamos_prendarios", "mcuentas_saldo"),
        ("mprestamos_hipotecarios", "mcuentas_saldo"),

        # Ahorros relativos al saldo total
        ("mcaja_ahorro", "mcuentas_saldo"),
        ("mcaja_ahorro_dolares", "mcuentas_saldo"),

        # Comisiones en relación a la rentabilidad
        ("mcomisiones", "mrentabilidad")
    ]

    

    cant_lag = 2
    window_size = 3
    df = feature_engineering_lag(df, columnas=atributos_fe_lag, cant_lag=cant_lag)
    df = feature_engineering_deltas(df, columnas=atributo_fe_deltas, cant_lag=cant_lag)
    df = feature_engineering_medias_moviles(df, columnas=atributos_fe_medias_moviles, window_size=window_size)
    df = feature_engineering_cum_sum(df, columnas=atributos_cum_sum)
    df = feature_engineering_min_max(df, columnas=atributos_min_max)
    df = feature_engineering_ratios(df, ratio_pairs=ratio_pairs)
  
    #02 Guardar datos
    path = "data/competencia_01_fe.csv"
    df.to_csv(path, index=False)
  
    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.{monbre_log}")

if __name__ == "__main__":
    main()