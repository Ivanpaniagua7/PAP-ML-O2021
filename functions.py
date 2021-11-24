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
from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

#%% Ivan
def PredictTrain(data, p, I, q, dias):
    if p>=q:
        rezagos=p+I
    else:
        rezagos=q+I
    data=data[:-rezagos].reset_index()
    PrediccionTrain=pd.DataFrame(columns=["Predicción"])
    for i in range(rezagos,len(data)-1):
        try:
            model = SARIMAX(pd.DataFrame(data["Close"][:i]), order=(p,I,q), seasonal_order=(0,0,0,0))
            model_fit = model.fit()
            diaspred=i+dias
            forecast = list(model_fit.predict(start=diaspred,end=diaspred,dynamic=True))
            PrediccionTrain.loc[data.loc[i+dias-1,"Date"]]=forecast[0] #-1 ya que aquí toma 1 dato más por el 0 del índice,
            #en predicción a 1 día, si i es 9, toma el decimo dato, es decir la fecha de la predicción.
        except:
            pass         
    return PrediccionTrain

def PredictTest(train, test, p, I, q, dias): #Este es con predicciones a diferentes dias
    if p>=q:
        rezagos=p+I
    else:
        rezagos=q+I
    data=train[-rezagos:].append(test)
    data=data.reset_index()
    PrediccionTrain=pd.DataFrame(columns=["Predicción"])
    for i in range(rezagos, len(data)):
        try:
            model = SARIMAX(pd.DataFrame(data["Close"][:i]), order=(p,I,q), seasonal_order=(0,0,0,0))
            model_fit = model.fit()
            diaspred=i+dias
            forecast = list(model_fit.predict(start=diaspred,end=diaspred,dynamic=True))
            PrediccionTrain.loc[data.loc[i+dias-1,"Date"]]=forecast[0] #-1 ya que aquí toma 1 dato más por el 0 del índice,
            #en predicción a 1 día, si i es 9, toma el decimo dato, es decir la fecha del de la predicción.
        except:
            pass         
    return PrediccionTrain

def MedidasDesempeño(PredTrain, btctrain, PredTest, btctest):
    MergeTrain=pd.merge(PredTrain,btctrain, left_index=True, right_index=True)
    MergeTest=pd.merge(PredTest,btctest, left_index=True, right_index=True)
    slope, intercept, r_value, p_value, std_err = stats.linregress(MergeTrain["Predicción"],MergeTrain["Close"])
    r2train=r_value**2
    slope, intercept, r_value, p_value, std_err = stats.linregress(MergeTest["Predicción"],MergeTest["Close"])
    r2test=r_value**2
    MSETrain=mse(MergeTrain["Predicción"],MergeTrain["Close"])
    MSETest=mse(MergeTest["Predicción"],MergeTest["Close"])
    return r2train, MSETrain, r2test, MSETest

def Desempeños(Desmp1, Desmp2, Desmp3, Desmp4, dias):
    
    ModelsResults=pd.DataFrame(columns=["R\u00b2 train","MSE train", "R\u00b2 test", "MSE test"])
    ModelsResults.loc["Model 8,1,8"]=Desmp1
    ModelsResults.loc["Model 10,1,10"]=Desmp2
    ModelsResults.loc["Model 12,1,7"]=Desmp3
    ModelsResults.loc["Model 12,1,12"]=Desmp4
    
    if dias>1:
        print("\033[1m Predicciones a %s días"%dias)
    else:
        print("\033[1m Predicciones a %s día"%dias)
    
    return ModelsResults


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


def estacionariedad(serie):
    plt.figure(figsize=(5, 2))
    serie.plot()
    plt.show()
    plot_acf(serie, lags=50)
    plt.show()
    plot_pacf(serie, lags=50)
    plt.show()


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


# PRUEBAS A LOS RESIDUALES
def prueba_normalidad(residuales):
    # Normalidad de los residuos Shapiro-Wilk test
    # ==============================================================================
    shapiro_test = stats.shapiro(residuales)
    print("statistic", shapiro_test[0])
    print("P-value=", shapiro_test[1])
    if shapiro_test[1] < 0.05:
        print("Los residuales se distrubuyen de forma normal.")
    else:
        print("Los residuales NO se distrubuyen de forma normal.")


def ljung_box(residuales):
    prueba = sm.stats.acorr_ljungbox(residuales, lags=[50], return_df=True)

    return prueba


def heterocedasticidad(residuales):
    prueba = breakvar_heteroskedasticity_test(residuales)

    return prueba







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


def medidas_train(PredYReales):
    slope, intercept, r_value, p_value, std_err = stats.linregress(PredYReales["Predicción"],
                                                                   PredYReales["Close"])
    r2train = r_value ** 2
    MSETrain = mse(PredYReales["Predicción"], PredYReales["Close"])

    return r2train, MSETrain


def medidas_test(test, forecast):
    slope, intercept, r_value, p_value, std_err = stats.linregress(test.squeeze(),forecast)
    r2test=r_value**2
    MSETest=mse(test.squeeze(),forecast)
    return r2test, MSETest
