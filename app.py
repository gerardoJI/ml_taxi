# Importación de librerías necesarias
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import gdown

# ID del archivo de Google Drive
file_id = '19HP4mVoQXLVGImqRp88C0pWYlsubBsjy'

# URL para descargar el archivo
url = f'https://drive.google.com/uc?export=download&id={file_id}'

# Ruta de destino para guardar el archivo descargado
output_path = 'random_forest_model.joblib'

# Descargar el archivo
gdown.download(url, output_path, quiet=False)


# Cargar los archivos pkl y joblib con los modelos previamente entrenados
try:
    # Cargar el modelo de escalado (Scaler) y el modelo de predicción (Random Forest)
    scaler_model = joblib.load('scaler.pkl')
    rf_model = joblib.load(output_path)
except Exception as e:
    # Si ocurre un error en la carga de los modelos, mostrar un mensaje de error
    st.error(f"Error al cargar los modelos: {e}")


# Función para realizar el procesamiento de los datos (Feature Engineering)
def feature_engineering(df):
    try:
        # Importar las funciones para el procesamiento de características
        from functions_limpieza import ft_eng1, ft_eng2
        
        # Aplicar la primera función de procesamiento (ft_eng1) a los datos
        df = ft_eng1(df)  
        
        # Definir las variables cíclicas (hora, minuto, mes, día, y día de la semana)
        variables_ciclicas = ['pickup_hour', 'pickup_minutes', 'pickup_month', 'pickup_day', 'day_of_week']
        
        # Aplicar la segunda función de procesamiento (ft_eng2) a las variables cíclicas
        df = ft_eng2(df, variables_ciclicas)
    except Exception as e:
        # Si ocurre un error en el procesamiento de las características, mostrar un mensaje de error
        st.error(f"Error en el feature engineering: {e}")
    
    # Retornar el DataFrame procesado
    return df

# Título y descripción de la aplicación en Streamlit
st.title("Predicción del precio de taxi en Nueva York")
st.write("""
    Ingrese los datos a continuación para predecir el precio de un viaje en taxi. 
    Complete todos los campos para obtener el resultado.
""")

# Solicitar la fecha del viaje al usuario
pickup_date = st.date_input(
    "Seleccione la fecha del viaje", 
    datetime(2010, 9, 25).date(),  # Valor por defecto
    min_value=datetime(2010, 1, 1).date(),  # Fecha mínima permitida
    max_value=datetime(2024, 12, 31).date(),  # Fecha máxima permitida
    help="Elija la fecha para viajar."  # Información adicional para el usuario
)

# Solicitar la hora del viaje al usuario
pickup_time = st.time_input(
    "Seleccione la hora del viaje", 
    datetime(2010, 9, 25, 15, 25).time(),  # Valor por defecto
    help="Elija la hora programada para viajar."  # Información adicional para el usuario
)

# Combinar la fecha y la hora seleccionadas en un solo objeto datetime
pickup_datetime = datetime.combine(pickup_date, pickup_time)

# Solicitar la longitud del inicio del viaje
pickup_longitude = st.number_input(
    "Ingrese la longitud del inicio del viaje", 
    min_value=-180.0, max_value=180.0, step=0.0001, value=-73.9857  # Valores por defecto y rango permitido
)

# Solicitar la latitud del inicio del viaje
pickup_latitude = st.number_input(
    "Ingrese la latitud del inicio del viaje", 
    min_value=-90.0, max_value=90.0, step=0.0001, value=40.7484  # Valores por defecto y rango permitido
)

# Solicitar la longitud de la llegada del viaje
dropoff_longitude = st.number_input(
    "Ingrese la longitud de la llegada del viaje", 
    min_value=-180.0, max_value=180.0, step=0.0001, value=-73.9747  # Valores por defecto y rango permitido
)

# Solicitar la latitud de la llegada del viaje
dropoff_latitude = st.number_input(
    "Ingrese la latitud de la llegada del viaje", 
    min_value=-90.0, max_value=90.0, step=0.0001, value=40.7473  # Valores por defecto y rango permitido
)

# Solicitar el número de pasajeros
passenger_count = st.number_input(
    "Ingrese el número de pasajeros", 
    min_value=1, max_value=6, step=1, value=1  # Número de pasajeros (valor por defecto 1)
)

# Crear un DataFrame con los datos ingresados por el usuario
data = {
    'pickup_datetime': [pickup_datetime],  # Fecha y hora de recogida
    'pickup_longitude': [pickup_longitude],  # Longitud de la recogida
    'pickup_latitude': [pickup_latitude],  # Latitud de la recogida
    'dropoff_longitude': [dropoff_longitude],  # Longitud de la bajada
    'dropoff_latitude': [dropoff_latitude],  # Latitud de la bajada
    'passenger_count': [passenger_count]  # Número de pasajeros
}
df = pd.DataFrame(data)

# Mostrar los datos ingresados por el usuario en la interfaz
st.write("Datos ingresados:", df)

# Procesamiento de las características (Feature Engineering) con la función definida anteriormente
df_processed = feature_engineering(df)

# Escalar los datos utilizando el modelo de escalado previamente cargado
df_scaled = scaler_model.transform(df_processed)

# Realizar la predicción utilizando el modelo de Random Forest previamente cargado
prediccion = rf_model.predict(df_scaled)

# Mostrar el resultado de la predicción (precio estimado del viaje)
st.write("**Predicción del precio del viaje:**")
st.markdown(f"<h3 style='color:green;'>${prediccion[0]:.2f}</h3>", unsafe_allow_html=True)

# Agregar un botón para permitir al usuario hacer otra predicción
if st.button("Realizar otra predicción"):
    # Si el usuario presiona el botón, el script se reinicia (se vuelve a ejecutar)
    st.rerun()  # Usar st.rerun() para reiniciar la aplicación y permitir al usuario ingresar nuevos datos

