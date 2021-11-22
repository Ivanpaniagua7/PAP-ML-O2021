
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import seaborn as sns
import matplotlib.pyplot as plt

#%% Ivan


#%% Edzna

def caja_bigotes(serie):
    plt.figure(figsize=(5, 2))
    sns.boxplot(serie)
    plt.show()


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






