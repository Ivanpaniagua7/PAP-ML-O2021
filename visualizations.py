
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
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

def grafica_forecast(data, btc, model_fit):
    test = data["Close"].reset_index()
    testini = test[test['Date'] == "2021-09-27"].index.values[0]
    testfin = test[test['Date'] == "2021-10-28"].index.values[0]
    test = test["Close"].loc[testini:testfin]
    train = btc.reset_index()["Close"]
    forecast = model_fit.predict(start=testini, end=testfin, dynamic=True)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(forecast, label='forecast')
    plt.legend(loc='upper left', fontsize=15)
    plt.show()






