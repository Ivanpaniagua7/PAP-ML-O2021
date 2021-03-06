"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
# Importar otros archivos
import data as dt
import functions as fn
import visualizations as vs

# Importar librerias
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
warnings.filterwarnings("ignore")


# %% Ivan


# %% Edzna

# Descargar los precios de cierre del Bitcoin
Cierres = dt.get_adj_closes('BTC-USD', "2019,01,09", "2021,28,11").shift().dropna()
Cierres = Cierres.loc["2019-09-27":]
btc = pd.DataFrame(Cierres["Close"].loc["2019-09-27":"2021-09-27"])

# Dividir la serie de tiempo por día de la semana
fechas = fn.serie_weekdays(btc)

# Diagramas de caja y bigotes para cada día de la semana


# Calcular el CO, HO, OL, HL así como su cambio en PIPs
conteo = fn.conteo_calculos(Cierres)

# METODOLOGÍA BOX JENKINS
# Paso 0: Identificar si la serie es estacionaria
# Graficar FAC y FACP para la serie original
btc = pd.DataFrame(btc["Close"])
plt.figure(figsize=(5, 2))
btc.plot()
plt.show()
plot_acf(btc, lags=30)
plt.show()
plot_pacf(btc, lags=30)
plt.show()

# Prueba Dickey Fuller
orig_DF = fn.pruebaDickeyFuller(btc)

# Calcular la primera diferencia de la serie
diff1 = btc.diff().dropna()
diff1_DF = fn.pruebaDickeyFuller(diff1)

# Paso 1: Identificación del modelo
# Graficar FAC y FACP para la primera diferencia
plot_acf(diff1, lags=50)
plt.show()
plot_pacf(diff1, lags=50)
plt.show()

# Paso 2: Estimación de parámetros del método elegido
modelo1 = fn.model_fit(btc, 8, 1, 8)

# Paso 3: Pruebas a los residuales
modelo1.plot_diagnostics()
plt.show()

# Paso 4: Pronóstico
btc = btc["Close"]
btc_train = btc.loc["2019-09-26":"2021-09-27"].shift()
btc_train = btc_train.dropna()

test = Cierres["Close"].reset_index()
testini = test[test['Date'] == "2021-09-27"].index.values[0]
testfin = test[test['Date'] == "2021-10-28"].index.values[0]
test = test["Close"].loc[testini:testfin]

forecast_serie = modelo1.predict(
       start=testini,
       end=testfin,
       dynamic=True)
forecast_grafica = vs.grafica_forecast(Cierres, btc, modelo1)

PrediccionesTry = fn.PredictTrain(btc_train, 9, 8, 1, 8)
# Predicciones y reales de train
PredYReales = pd.merge(PrediccionesTry, btc_train, left_index=True, right_index=True)

# Medidas train
medidas_train = fn.medidas_train(PredYReales)

# Medidas test
medidas_test = fn.medidas_test(test, forecast_serie)