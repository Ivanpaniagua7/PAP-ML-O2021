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
import yfinance as yf
import datetime
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import mse
#from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

#%% Ivan


#%% Edzna

def serie_weekdays(btc):
    fecha = pd.DataFrame(btc.index)
    fecha = pd.DataFrame(pd.to_datetime(fecha["Date"]).dt.weekday)
    btc = btc.reset_index(drop=True)
    btc["Weekday"] = fecha
    days = btc.groupby("Weekday")
    df = pd.DataFrame()
    df["Lunes"] = days.get_group(0)["Close"].reset_index(drop=True)
    df["Martes"] = days.get_group(1)["Close"].reset_index(drop=True)
    df["Miércoles"] = days.get_group(2)["Close"].reset_index(drop=True)
    df["Jueves"] = days.get_group(3)["Close"].reset_index(drop=True)
    df["Viernes"] = days.get_group(4)["Close"].reset_index(drop=True)
    df["Sábado"] = days.get_group(5)["Close"].reset_index(drop=True)
    df["Domingo"] = days.get_group(6)["Close"].reset_index(drop=True)

    return df


def conteo_calculos(bitcoin):
    btc = bitcoin.copy()
    df = pd.DataFrame()
    df['CO'] = (btc['Close'] - btc['Open'])
    df['CO pips'] = df['CO'] * 0.01
    df['HO'] = (btc['High'] - btc['Open'])
    df['HO pips'] = df['HO'] * 0.01
    df['OL'] = (btc['Open'] - btc['Low'])
    df['OL pips'] = df['OL'] * 0.01
    df['HL'] = (btc['High'] - btc['Low'])
    df['HL pips'] = df['HL'] * 0.01

    return df


def pruebaDickeyFuller(btc):
    dftest = adfuller(btc)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    return dfoutput


def model_fit(btc, p, d, q):
    model = SARIMAX(btc.reset_index()["Close"], order=(p, d, q), seasonal_order=(0, 0, 0, 0))
    modelo_fit = model.fit()

    return modelo_fit


def PredictTrain(data, rezagos, p, I, q):
    data = data.reset_index()
    PrediccionTrain = pd.DataFrame(columns=["Predicción"])
    for i in range(rezagos, len(data)-1):
        try:
            model = SARIMAX(pd.DataFrame(data["Close"][:i]), order=(p,I,q), seasonal_order=(0,0,0,0))
            model_fit = model.fit()
            nextday=i+1
            forecast = list(model_fit.predict(start=nextday,end=nextday,dynamic=True))
            PrediccionTrain.loc[data.loc[i,"Date"]]=forecast[0]
        except:
            pass
    return PrediccionTrain
