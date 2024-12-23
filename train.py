#--- Entraniemiento del modelo y generación de archivos pickle para su uso en streamlit

import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Cargar datos
df = pd.read_csv('data/df_final.csv')

#Se generan dataframes para características y targets
features = df.drop(columns = ["fare_amount"])
target = df["fare_amount"]



#------------------------------------ Estandarización de datos----------------------------------------------
# scaling the data 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler= StandardScaler()

scaler.fit(features)
features = scaler.transform(features)

import pickle
# Guardar el modelo entrenado en un archivo .pkl
with open('scaler.pkl', 'wb') as file: 
    pickle.dump(scaler, file) 


#---------------------------------Entranamiento del modelo-------------------------------------

#Ejecución de random forest con los mejores hiperparametros.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import joblib

start_time = time.time()
# Initialize RandomForestRegressor
forest = RandomForestRegressor(n_estimators=180, max_depth=24)

# Train the model
forest.fit(features, target)

# Guardar el modelo con compresión (nombre del modelo es 'forest')
joblib.dump(forest, 'random_forest_model.joblib', compress=3)
print("Modelo 'forest' guardado con compresión.")


