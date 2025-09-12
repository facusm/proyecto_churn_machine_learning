import pandas as pd
import duckdb
import logging

logger = logging.getLogger("__name__")

def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
  
    # Construir la consulta SQL
    sql = "SELECT *"
  
    # Agregar los lags para los atributos especificados. Si la columna ya existe, no se vuelve a generar.
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                col_name = f"{attr}_lag_{i}"
                if col_name not in df.columns:
                    sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {col_name}"
                else:
                    logger.warning(f"La columna {col_name} ya existe, no se vuelve a generar")

  
    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())
  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df




def feature_engineering_min_max(
    df: pd.DataFrame,
    columnas: list[str],
) -> pd.DataFrame:
    """
    Genera variables de min y max por cliente para los atributos especificados.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str]
        Lista de atributos para los cuales generar min y max.

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de min y max agregadas.
    """
    logger.info(f"Generando min y max para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar min y max")
        return df

    sql = "SELECT *"

    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"La columna {attr} no existe en el DataFrame")
            continue

        col_min = f"{attr}_min"
        col_max = f"{attr}_max"

        if col_min not in df.columns:
            sql += f", MIN({attr}) OVER (PARTITION BY numero_de_cliente) AS {col_min}"
        else:
            logger.warning(f"La columna {col_min} ya existe, no se vuelve a generar")

        if col_max not in df.columns:
            sql += f", MAX({attr}) OVER (PARTITION BY numero_de_cliente) AS {col_max}"
        else:
            logger.warning(f"La columna {col_max} ya existe, no se vuelve a generar")

    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df