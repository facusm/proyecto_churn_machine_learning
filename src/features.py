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




def feature_engineering_deltas(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera columnas de delta en SQL: valor actual menos lag correspondiente.
    Mantiene NULL si no hay información suficiente.
    """
    logger.info(f"Generando deltas (SQL) para {len(columnas)} columnas con {cant_lag} lags")

    sql = "SELECT *"

    for attr in columnas:
        for i in range(1, cant_lag + 1):
            lag_col = f"{attr}_lag_{i}"
            delta_col = f"{attr}_delta_{i}"

            if lag_col in df.columns and delta_col not in df.columns:
                sql += f", {attr} - {lag_col} AS {delta_col}"
            elif delta_col in df.columns:
                logger.warning(f"{delta_col} ya existe, no se genera nuevamente")
            else:
                logger.warning(f"{lag_col} no existe, no se puede generar {delta_col}")

    sql += " FROM df"

    con = duckdb.connect(":memory:")
    con.register("df", df)
    df_out = con.execute(sql).df()
    con.close()

    logger.info(f"Deltas generadas. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out





def feature_engineering_medias_moviles(df: pd.DataFrame, columnas: list[str], window_size: int = 3) -> pd.DataFrame:
    """
    Genera columnas de medias móviles en SQL incluyendo el valor actual y los anteriores
    según el tamaño de ventana especificado.
    """
    logger.info(f"Generando medias móviles (SQL) para {len(columnas)} columnas con ventana de {window_size}")

    sql = "SELECT *"
    for attr in columnas:
        ma_col_name = f"{attr}_ma_{window_size}"
        if ma_col_name not in df.columns:
            sql += f", AVG({attr}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW) AS {ma_col_name}"
        else:
            logger.warning(f"{ma_col_name} ya existe, no se vuelve a generar")

    sql += " FROM df"

    con = duckdb.connect(":memory:")
    con.register("df", df)
    df_out = con.execute(sql).df()
    con.close()

    return df_out


def feature_engineering_cum_sum(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
  """
  Genera columnas de suma acumulada por cliente para los atributos indicados.
  Los valores NULL se tratan como 0 solo para el cálculo de la suma acumulada,
  sin modificar las columnas originales.

  Parámetros
  ----------
  df : pd.DataFrame
      DataFrame con los datos originales
  columnas : list[str]
      Lista de columnas sobre las cuales generar la suma acumulada

  Retorna
  -------
  pd.DataFrame
  DataFrame con las nuevas columnas *_cumsum agregadas
  """
  if columnas is None or len(columnas) == 0:
    logger.warning("No se especificaron atributos para generar lags")
    return df

  sql = "select *"

  for attr in columnas:
    if attr in df.columns:
      if f"{attr}_cumsum" not in df.columns:
        sql += f", sum(coalesce({attr}, 0)) over (partition by numero_de_cliente order by foto_mes ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS {attr}_cumsum"
      else:
        logger.warning(f"{attr}_cumsum ya existe, no se vuelve a generar")

  sql += f" from df"

  logger.debug(f"Consulta SQL: {sql}")

  # Ejecutar la consulta SQL
  con = duckdb.connect(database=":memory:")
  con.register("df", df)
  df = con.execute(sql).df()
  con.close()

  print(df.head())

  logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

  return df


def feature_engineering_min_max(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
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



def feature_engineering_ratios(df: pd.DataFrame, ratio_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Genera columnas de ratios entre pares válidos de columnas.
    Maneja NULL y división por cero.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales
    ratio_pairs : list[tuple[str, str]]
        Lista de tuplas (numerador, denominador) indicando los ratios a generar

    Retorna
    -------
    pd.DataFrame
        DataFrame con las nuevas columnas *_ratio agregadas
    """
    if not ratio_pairs:
        logger.warning("No se especificaron pares de columnas para generar ratios")
        return df

    sql = "SELECT *"

    for numerador, denominador in ratio_pairs:
        ratio_col = f"{numerador}_over_{denominador}"

        if numerador in df.columns and denominador in df.columns:
            if ratio_col not in df.columns:
                # NULLIF evita división por cero
                sql += f", {numerador} / NULLIF({denominador}, 0) AS {ratio_col}"
            else:
                logger.warning(f"{ratio_col} ya existe, no se vuelve a generar")
        else:
            logger.warning(f"Columnas no encontradas: {numerador} o {denominador}, se omite el ratio {ratio_col}")

    sql += " FROM df"

    con = duckdb.connect(":memory:")
    con.register("df", df)
    df_out = con.execute(sql).df()
    con.close()

    logger.info(f"Ratios generados. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out
