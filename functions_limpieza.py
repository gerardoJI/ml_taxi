#----------------------------FUNCIONES PARA LA LIMPIEZA DE DATOS Y ADICIÓN DE COLUMNAS AL DATAFRAME

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
    print(f'Se eliminaron {datos_i - datos_f} por valores nulos, es decir, el {round(1 - datos_i / datos_f * 100, 2)}% de los datos.')
    
    # Conversión de la columna "pickup_datetime" a tipo datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # Se crea una nueva columna "hour" extrayendo la hora de la columna "pickup_datetime"
    df['hour'] = df['pickup_datetime'].dt.hour
    
    # Se crea una nueva columna "date" extrayendo la fecha (sin la hora) de la columna "pickup_datetime"
    df['date'] = df['pickup_datetime'].dt.date
    
    # Mensaje indicando que las columnas "hour" y "date" han sido añadidas
    print("\n Se añadieron las columnas hour y date.")
    
    # Retorna el dataframe modificado con las nuevas columnas y sin los valores nulos
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