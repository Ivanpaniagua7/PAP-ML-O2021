
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""


#%% Ivan
import pandas_datareader.data as web

# Función para descargar precios de cierre ajustados:
def get_adj_closes(tickers, start_date=None, end_date=None):
    # Fecha inicio por defecto (start_date='2010-01-01') y fecha fin por defecto (end_date=today)
    # Descargamos DataFrame con todos los datos
    closes = web.DataReader(name=tickers, data_source='yahoo', start=start_date, end=end_date)
    # Solo necesitamos los precios ajustados en el cierre
    closes = closes['Adj Close']
    # Se ordenan los índices de manera ascendente
    closes.sort_index(inplace=True)
    return closes

Closes=get_adj_closes(tickers=['BTC-USD'], start_date=("2020,09,01"))






#%% Edzna
import pandas as pd
import yfinance as yf


def get_adj_close(tickers: str, start_date: str, end_date: str):
    closes = yf.download(tickers,
                         start=start_date,
                         end=end_date,
                         progress=False)['Adj Close'].reset_index(drop=True)

    return pd.DataFrame(closes)

