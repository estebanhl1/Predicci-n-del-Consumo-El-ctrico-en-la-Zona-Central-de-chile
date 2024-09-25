# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:08:55 2024

@author: mirko
"""


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf, acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from numpy.linalg import LinAlgError
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

#%% Definición de funciones


def detect_outliers(df, column_name):
        """
        Detecta outliers utilizando rango intercuartilico IQR. 
        Esta función toma dos argumentos, un dataframe y el nombre de la columna de interes y devuelve los indices de los
        outliers.
        
        Parameters
        ----------
        df : pandas.DataFrame
            El dataframe que contiene los datos.
        column_name : str
            El nombre de la columna sobre la cual se buscaran los indices de los outliers.

        Returns
        -------
        outliers_idx : pandas.Index
            Lista con los indices donde se encuentran los outliers de column_name.

        """
        # Calcular el IQR 
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir límites para detectar outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Detectar los índices de los outliers
        outliers_idx = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)].index
        return outliers_idx



def replace_outliers(df, column_name, outliers_idx):
    """
    Reemplaza outliers por el promedio de los 2 valores anteriores y siguientes en un dataframe.
    Esta función toma 3 argumentos, el dataframe df, la columna de interes column_name y la outliers_idx que contiene
    los indices de los outliers. Devuelve el dataframe df sin outliers.
    
    Utiliza la función auxiliar 'replace_with_avg' reemplaza el outlier con el promedio antes mencionado.

    Parameters
    ----------
    df : pandas.DataFrame
         El dataframe que contiene los datos.
    column_name : str
         El nombre de la columna sobre la cual se buscaran los indices de los outliers.
    outliers_idx : pandas.Index
         Lista con los indices donde se encuentran los outliers de column_name.
    Returns
    -------
   df : pandas.Dataframe
        El dataframe sin los outliers

    """

    
    
    # Copiar la columna original para evitar alterar los valores mientras reemplazamos
    original_series = df[column_name].copy()
    

    def replace_with_avg(series, idx):
        """
        Función auxiliar que calcula el promedio de los 2 valores anteriores y siguientes en un dataframe.
        

        """
        if idx > 1 and idx < len(series) - 2:
            prev_values = series[idx-2:idx]
            next_values = series[idx+1:idx+3]
            avg = np.mean(np.concatenate([prev_values, next_values]))
            return avg
        else:
            return series[idx]  
    
    # Reemplazar los outliers en los índices identificados
    for idx in outliers_idx:
        df.at[idx, column_name] = replace_with_avg(original_series, idx)
    
    return df

def plot_consumo_subestaciones(consumo_dfs, tipo_consumo, column_name , name_fig='consumo_subestaciones.png'):
    """
    Genera y guarda un gráfico con los lineplots del consumo mensual para varias subestaciones eléctricas.

    Esta función toma un diccionario donde las claves son los nombres de las subestaciones y los valores son
    DataFrames que contienen el consumo mensual. Se genera un lineplot para cada subestación y se organiza 
    en una cuadrícula de 3x3. Si hay menos subestaciones que ejes disponibles, los ejes sobrantes se ocultan.

    Parameters:
    ----------
    consumo_dfs : dict
        Diccionario donde las claves son los nombres de las subestaciones y los valores son DataFrames con 
        las columnas 'column_name' y 'consumption' (consumo).
    tipo_consumo : str
        Indica el tipo de consumo, diario o mensual.
    column_name : str
        Nombre de la columna en el dataframe, hace referencia a si es mensual o diario el consumo.
    output_path : str, optional
        La ruta/nombre donde se guardará el gráfico generado (por defecto es 'consumo_subestaciones.png').

    Returns:
    -------
    None
        La función no retorna ningún valor. El gráfico se guarda en la ruta especificada.
    
    """
    # Crear una figura con subplots 3x3
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    axs = axs.flatten()
    
    # Iterar sobre el diccionario para crear los lineplots
    for i, (nombre, df) in enumerate(consumo_dfs.items()):
        # Crear el lineplot
        sns.lineplot(data=df, x=column_name, y='consumption', ax=axs[i])
        axs[i].set_title(f'Consumo {tipo_consumo} - {nombre}', fontsize=20)
        axs[i].set_xlabel('Fecha', fontsize=20)
        axs[i].set_ylabel('Consumo', fontsize=20)
        axs[i].tick_params(axis='x', rotation=45)
    
    # Si hay menos subplots que ejes, ocultar los ejes restantes
    for j in range(len(consumo_dfs), len(axs)):
        axs[j].axis('off')
    
    # Ajustar la visualización y guardar la figura
    plt.tight_layout()
    plt.savefig(name_fig, dpi=300, bbox_inches='tight')
    plt.show()

def check_estacionaria(series, n_lags):
    """
    Verifica si la serie temporal es estacionaria utilizando la prueba de Dickey-Fuller aumentada.
    
    Parameters:
    -----------
    series (pd.Series): La serie temporal que se va a analizar.
    n_lags (int): El número de lags a utilizar en la prueba ADF.
    
    Returns:
    -------
    None
    
    """
    
    result = adfuller(series, maxlag=n_lags)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Número de lags utilizados:', result[2])
    print('Critical Values:')
    for key, value in result[4].items():
       print('\t%s: %.3f' % (key, value))
       
    return None   

def plot_consumo_acf_pacf(df, dft, tipo_consumo, y_train, y_test, estacion, n_lags):
    """
    Genera gráficos del consumo mensual y los gráficos ACF y PACF.

    Parameters:
    df (pd.DataFrame): DataFrame que contiene los datos de consumo.
    y_train (pd.Series): Serie temporal de entrenamiento.
    y_test (pd.Series): Serie temporal de prueba.
    estacion (str): Nombre de la estación para el título de los gráficos.
    n_lags (int): Número de lags a utilizar en los gráficos ACF y PACF.
    """
    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches(25, 5)

    # Gráfico del consumo mensual de entrenamiento
    axes[0].plot(df.index, y_train)
    axes[0].set_title(f'Consumo {tipo_consumo} - {estacion}', fontsize=20)
    axes[0].set_xlabel(f'{tipo_consumo}', fontsize=20)
    axes[0].set_ylabel('Consumo', fontsize=20)

    # Gráfico del consumo mensual de prueba
    axes[3].plot(dft.index, y_test)
    axes[3].set_title(f'Consumo {tipo_consumo} test - {estacion}', fontsize=20)
    axes[3].set_xlabel(f'{tipo_consumo}', fontsize=20)

    # Gráfico ACF
    plot_acf(y_train, lags=n_lags, ax=axes[1])
    axes[1].set_title('ACF', fontsize=20)

    # Gráfico PACF
    plot_pacf(y_train, lags=n_lags, ax=axes[2])
    axes[2].set_title('PACF', fontsize=20)
    plt.tight_layout()
   

    # Guardar la figura
    nombre = f'{estacion}_acf_pacf.png'
    plt.savefig(nombre, dpi=300, bbox_inches='tight')
    plt.show()

def best_p_q_mensual(y_train):
    """
    

    Parameters
    ----------
    y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    qOptimo : TYPE
        DESCRIPTION.
    pOptimo : TYPE
        DESCRIPTION.

    """
    pvalue = []
    qvalue = []
    MSEvalue    = []
    valoresP = []
    for i in range(1,20):
        valoresP.append(i) 

    valoresQ = valoresP    
    for p in valoresP:
        
        for q in valoresQ:
           try: 
            modelo = ARIMA (y_train, order = (p,0,q))
            predictor = modelo.fit()
            MSE = predictor.mse
            pvalue.append(p)
            qvalue.append(q)
            MSEvalue.append(MSE)
            
           except LinAlgError as e:
            
            pass   
           
    metricasARIMA = pd.DataFrame({
        'pvalue':pvalue,
        'qvalue':qvalue,
        'MSEvalue': MSEvalue        
        })
    menorMSE = metricasARIMA['MSEvalue'].idxmin()
    qOptimo = metricasARIMA.loc[menorMSE,'qvalue']
    pOptimo = metricasARIMA.loc[menorMSE,'pvalue']
    
    return qOptimo,pOptimo
    




    
#%% EDA 

# Cambiar al directorio de los datos
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Carga de los datos

df = pd.read_csv("train.csv")
dft = pd.read_csv("test.csv")

# Caracteristicas de los datos 

print(df.info())
print("")
print("Indico valores nulos por columnas")
print(df.isnull().sum()) # Sin valores nulos
print("")
print("Cantidad de subestaciones :", df['substation'].nunique())

subestaciones = df['substation'].unique().tolist()

#%% Divido los datos por subestaciones

dfs_by_substation = {substation: df[df['substation'] == substation] for substation in df['substation'].unique()}
dfst_by_substation = {substation: dft[dft['substation'] == substation] for substation in dft['substation'].unique()}

#%% Debido a que quiero predicir consumos mensuales y diarios, voy a generar 4 diccionarios para cada uno


consumo_diario_dfs = {}
consumo_diario_dfst = {}
consumo_mensual_dfs = {}
consumo_mensual_dfst = {}

# Consumos de entrenamiento

for substation, df_sub in dfs_by_substation.items():
    # Agrupar por mes o día y sumar el consumo total
    df_sub['date'] = pd.to_datetime(df_sub['date'], errors='coerce')
    
    df_sub['day'] = df_sub['date'].dt.to_period('D')  
    df_sub['month'] = df_sub['date'].dt.to_period('M') 
   
    
    monthly_consumption = df_sub.groupby('month')['consumption'].sum().reset_index()
    monthly_consumption['month'] = monthly_consumption['month'].dt.to_timestamp()
    
    daily_consumption = df_sub.groupby('day')['consumption'].sum().reset_index()
    daily_consumption['day'] = daily_consumption['day'].dt.to_timestamp()
    
    # Guardar en el diccionario el DataFrame con el consumo
    consumo_diario_dfs[substation] = daily_consumption
    consumo_mensual_dfs[substation] = monthly_consumption 

# Consumos de test
    
for substation, df_sub in dfst_by_substation.items():

    df_sub['date'] = pd.to_datetime(df_sub['date'], errors='coerce')
    
    df_sub['day'] = df_sub['date'].dt.to_period('D') 
    df_sub['month'] = df_sub['date'].dt.to_period('M')  
   
    
    monthly_consumption = df_sub.groupby('month')['consumption'].sum().reset_index()
    monthly_consumption['month'] = monthly_consumption['month'].dt.to_timestamp()
    
    daily_consumption = df_sub.groupby('day')['consumption'].sum().reset_index()
    daily_consumption['day'] = daily_consumption['day'].dt.to_timestamp()
    
    consumo_diario_dfst[substation] = daily_consumption
    consumo_mensual_dfst[substation] = monthly_consumption     

#%% Aplico la revisión de outliers

consumos = {
    'diario_train': consumo_diario_dfs,
    'diario_test': consumo_diario_dfst,
    'mensual_train': consumo_mensual_dfs,
    'mensual_test': consumo_mensual_dfst,  
    }

for tipo_consumo in consumos.values():
    for nombre_df, df in tipo_consumo.items():
        outliers_idx = detect_outliers(df, 'consumption')
        df = replace_outliers(df, 'consumption', outliers_idx)    

    



    
#%% Creo y guardo los gráficos correspondientes

plot_consumo_subestaciones(consumo_diario_dfs, 'diario', 'day', name_fig='consumo_subestaciones_diario_train.png')
plot_consumo_subestaciones(consumo_diario_dfst, 'diario', 'day', name_fig='consumo_subestaciones_diario_test.png')
plot_consumo_subestaciones(consumo_mensual_dfs, 'mensual', 'month', name_fig='consumo_subestaciones_mensual_train.png')
plot_consumo_subestaciones(consumo_mensual_dfst, 'mensual', 'month', name_fig='consumo_subestaciones_mensual_test.png')

#%% Predicciones mensuales

Nlags = 6
for estacion in subestaciones:
    df = consumo_mensual_dfs[estacion]
    dft = consumo_mensual_dfst[estacion]
    y_train = df['consumption'].to_numpy()
    y_train = y_train.reshape(len(y_train),1)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    y_test = dft['consumption'].to_numpy()
    y_test = y_test.reshape(len(y_test),1)
    sc_y_test = StandardScaler()
    y_test = sc_y_test.fit_transform(y_test)
    check_estacionaria(y_train, Nlags)
    plot_consumo_acf_pacf(df, dft,'mes', y_train, y_test, estacion, Nlags)
    p,q = best_p_q_mensual(y_train)
    modelo = ARIMA ( y_train, order = (p,0,q))
    predictor = modelo.fit()
    horizonte = len(y_test)
    x = list(range(0, horizonte))
    x_p = np.add(x,len(y_train))
    x_train = list(range(0,len(y_train)))
     

    modelo_pred=predictor.get_forecast(steps=horizonte)
    modelo_ci=modelo_pred.conf_int(0.05)

    y_pred=modelo_pred.predicted_mean
    y_true = y_test
    RMSE = np.mean(  np.sqrt((y_pred - y_true)*(y_pred - y_true)))

    nombre = 'consumo_predicho_mensual_subestacion'+ estacion + '.jpg'
    titulo = 'Predicción estacion '+ estacion

    plt.figure(figsize=(15, 10))
    plt.plot(x_train , y_train, label = 'TRAIN')
    plt.plot(x_p , y_pred, label = 'PREDICCIONES') 
    plt.plot(x_p , y_true, label = 'TEST')
    plt.xlabel('Meses', fontsize = 20)
    plt.ylabel('Consumo total mensual (Kw)' , fontsize = 20)
    plt.title(titulo , fontsize = 20)
    plt.fill_between(x_p,modelo_ci[:,0],modelo_ci[:,1],color="b",alpha=.15)
    plt.legend(title = f"RMSE : {RMSE :.2f}",  fontsize = 20)
    plt.savefig(nombre, dpi=500, bbox_inches='tight')
    plt.show()
    








