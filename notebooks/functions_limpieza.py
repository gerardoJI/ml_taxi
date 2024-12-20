#----------------------------FUNCIONES PARA LA LIMPIEZA DE DATOS

def limpieza1(df):
    import pandas as pd
    # Se verifican los valores nulos en el dataframe.
    # Almacenamos la cantidad inicial de filas en el dataframe antes de eliminar los nulos
    datos_i = df.shape[0]  # variable para almacenar cantidad de datos iniciales.
    
    # Se elimina toda fila que contenga al menos un valor nulo
    df = df.dropna() 
    
    # Almacenamos la cantidad de filas restantes después de eliminar los nulos
    datos_f = df.shape[0]  # variable para almacenar cantidad de datos finales
    
    # Se imprime la cantidad de filas eliminadas por valores nulos y el porcentaje que representa sobre el total
    print(f'Se eliminaron {datos_i - datos_f} por valores nulos. Restan {datos_f} datos.')
    
    # Retorna el dataframe modificado y sin los valores nulos
    return df

#----------------------------FUNCIONES FEATURE ENGINEERING----------------------------------------

def ft_eng1(df):
    """
    Esta función realiza varias transformaciones y generaciones de nuevas características en el dataframe de taxis.
    
    Los pasos realizados son:
    1. Conversión de la columna "pickup_datetime" a formato datetime.
    2. Extracción de la hora y minuto del "pickup_datetime".
    3. Creación de una columna booleana para identificar si el viaje ocurrió durante las horas pico.
    4. Extracción de día, mes, año y día de la semana del "pickup_datetime".
    5. Cálculo de la distancia entre las coordenadas de recogida y entrega del viaje.
    6. Eliminación de la columna "pickup_datetime" original.

    Parámetros:
    - df: DataFrame que contiene la información de los viajes en taxi.

    Retorna:
    - df: DataFrame con las nuevas columnas generadas y la columna "pickup_datetime" eliminada.
    """
    
    import pandas as pd
    
    # Conversión de la columna "pickup_datetime" a tipo datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Se crea una nueva columna "pickup_hour" extrayendo la hora de la columna "pickup_datetime"
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    
    # Se crea una nueva columna "pickup_minutes" extrayendo los minutos de la columna "pickup_datetime"
    df['pickup_minutes'] = df['pickup_datetime'].dt.minute  # 'minutes' está mal, debería ser 'minute'

    # Genera columna booleana para definir si es hora pico (7-9 AM y 5-7 PM)
    df['peak_hour'] = df['pickup_hour'].isin([7, 8, 9, 17, 18, 19])

    # Crear las nuevas columnas con el día, mes y año de la fecha de recogida
    df['pickup_year'] = df['pickup_datetime'].dt.year  # Año del viaje
    df['pickup_month'] = df['pickup_datetime'].dt.month  # Mes del viaje
    df['pickup_day'] = df['pickup_datetime'].dt.day  # Día del mes del viaje
    
    # Genera columna con valores del 0 al 6 para indicar día de la semana
    # 0 = lunes, 6 = domingo
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  # Cambiado a 'pickup_datetime' (no 'date')

    # Se elimina la columna "pickup_datetime"
    df.drop(columns=["pickup_datetime"], inplace=True)  # 'inplace=True' asegura que el cambio se aplique directamente

    # Se calcula la distancia lineal del viaje y se crea la columna correspondiente
    from functions_limpieza import calcular_distancia
    df = calcular_distancia(df, "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude")

    # Mensaje indicando los resultados del proceso
    print(f"\n Se eliminó la columna 'pickup_datetime'. Las columnas en el dataframe son: {df.columns}.")
    
    return df



def calcular_distancia(df, lat1_col, lon1_col, lat2_col, lon2_col):
    """
    Función para calcular la distancia entre dos puntos geográficos utilizando la fórmula de Haversine.
    
    La fórmula de Haversine calcula la distancia en línea recta (sobre la superficie de una esfera) entre dos puntos
    dados por sus coordenadas geográficas (latitud y longitud). El resultado se devuelve en kilómetros.
    
    Parámetros:
    df (pandas.DataFrame): DataFrame que contiene las coordenadas de dos puntos geográficos.
    lat1_col (str): Nombre de la columna que contiene las latitudes del primer punto.
    lon1_col (str): Nombre de la columna que contiene las longitudes del primer punto.
    lat2_col (str): Nombre de la columna que contiene las latitudes del segundo punto.
    lon2_col (str): Nombre de la columna que contiene las longitudes del segundo punto.

    Retorna:
    pandas.DataFrame: DataFrame original con una nueva columna llamada 'distancia', que contiene la distancia en kilómetros
    entre los dos puntos para cada fila.

    """
    
    import pandas as pd
    import math

    # Función interna para calcular la distancia usando la fórmula de Haversine
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calcula la distancia entre dos puntos geográficos utilizando la fórmula de Haversine.
        
        Parámetros:
        lat1 (float): Latitud del primer punto en grados.
        lon1 (float): Longitud del primer punto en grados.
        lat2 (float): Latitud del segundo punto en grados.
        lon2 (float): Longitud del segundo punto en grados.
        
        Retorna:
        float: Distancia en kilómetros entre los dos puntos.
        """
        # Radio de la Tierra en km
        R = 6371.0
        
        # Convertir las coordenadas de grados a radianes
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Diferencias de coordenadas
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Aplicar la fórmula de Haversine
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # Distancia en km
        distance = R * c
        return distance
    
    # Calcular la distancia para cada fila del DataFrame y agregar la columna 'distancia'
    df['distance'] = df.apply(lambda row: haversine(row[lat1_col], row[lon1_col], row[lat2_col], row[lon2_col]), axis=1)
    
    return df

import numpy as np
import pandas as pd

def ft_eng2(df, variables_ciclicas):
    """
    Esta función aplica una transformación trigonométrica (seno y coseno) a las columnas cíclicas especificadas
    en la lista 'variables_ciclicas', y elimina las columnas originales del dataframe.

    Parámetros:
    - df: DataFrame que contiene las columnas a transformar.
    - variables_ciclicas: Lista de nombres de las columnas que contienen variables cíclicas a transformar.

    Retorna:
    - df: DataFrame con las nuevas columnas transformadas y las originales eliminadas.
    """
    
    # Para cada variable en la lista de variables cíclicas
    for var in variables_ciclicas:
        # Calcular las transformaciones trigonométricas de seno y coseno
        df[f'{var}_sin'] = np.sin(2 * np.pi * df[var] / df[var].max())  # Seno
        df[f'{var}_cos'] = np.cos(2 * np.pi * df[var] / df[var].max())  # Coseno

    # Eliminar las columnas originales
    df.drop(columns=variables_ciclicas, inplace=True)
    
    return df
