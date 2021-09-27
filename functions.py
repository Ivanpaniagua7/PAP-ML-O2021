"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def prueba_DickeyFuller(df: pd.DataFrame):
    result = adfuller(df.dropna())
    # result= df.diff()
    # Primero verificamos si la serie está estacionaria usando la prueba Augmented Dickey Fuller
    print('ADF Statistic: %f' % result[0])
    # si el valor p de la prueba es menor que el nivel de significancia (0.05), entonces rechaza la hipótesis nula
    # e infiere que la serie de tiempo es de hecho estacionaria.
    print('p-value: %f' % result[1])  # si el valor p de la prueba es menor que el nivel de significancia (0.05),
    # entonces rechaza la hipótesis nula e infiere que la serie de tiempo es de hecho estacionaria.
    if result[1] > 0.05:
        print('(', result[1], '> 0.05 )',
              'Dado que el valor P es mayor que el nivel de significancia, la serie NO es estacionaria, '
              'por lo que debemos de diferenciar la serie.')
    else:
        print('(', result[1], '< 0.05 )',
              'Dado que el valor es menor que el nivel de significancia (0.05), no es necesario diferenciar la serie')
    return result

def autocorrelacion(df, diff1, diff2):
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df["Adj Close"]);
    axes[0, 0].set_title('Original Series')
    plot_acf(df["Adj Close"], ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(diff1);
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(diff1.dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(diff2);
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(diff2.dropna(), ax=axes[2, 1])

    plt.show()


def resumen(Di_Fu0, Di_Fu1):
    datos_diferencia = list(zip(Di_Fu0, Di_Fu1))
    df_diferencias = pd.DataFrame(datos_diferencia)
    # Nombre de columnas
    df_diferencias.set_axis(['0 diferencia', '1ra diferencia'],
                            axis='columns', inplace=True)
    # Nombre de filas
    df_diferencias.rename(index={0: 'ADF estadístico ó prueba aumentada de Dickey-Fulle:',
                                 1: 'P-Value:',
                                 2: 'Rezagos usados:',
                                 3: 'Número de observaciones utilizadas para la regresión ADF '
                                    'y el cálculo de los valores críticos:',
                                 4: 'Valores críticos. Basado en MacKinnon:',
                                 5: 'El criterio de información maximizada si el autolag no es Ninguno:'},
                          inplace=True)

    return df_diferencias

def AR(df):
    # PACF plot of 1st differenced series -----MODIFIQUE LOS DATOS DE LA ARIMA-----
    plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})

    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(df);
    axes[0].set_title('2nd Differencing')
    axes[1].set(xlim=(0, 20), ylim=(0, 1))
    plot_pacf(df.dropna(), ax=axes[1])

    plt.show()

def MA(df):
    plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
    # Import data
    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(df);
    axes[0].set_title('2nd Differencing')
    axes[1].set(xlim=(0, 15), ylim=(0, 1))
    plot_acf(df.dropna(), ax=axes[1])

    plt.show()