
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import pandas_datareader.data as web
import pandas as pd


#%% Ivan


# Función para descargar precios de cierre ajustados:
def get_adj_closes(tickers, start_date=None, end_date=None):
    # Fecha inicio por defecto (start_date='2010-01-01') y fecha fin por defecto (end_date=today)
    # Descargamos DataFrame con todos los datos
    closes = web.DataReader(name=tickers, data_source='yahoo', start=start_date, end=end_date)
    # Solo necesitamos los precios ajustados en el cierre
    # Se ordenan los índices de manera ascendente
    closes.sort_index(inplace=True)

    return closes

def DatosIntradia():
    
    def is_nan(x):
        return (x != x)

    Datos=pd.read_csv("BTCUSDT.csv")
    for i in range(len(Datos)):
        if is_nan(Datos.iloc[i,3]):
            temp=Datos.iloc[:i,2:4]
            temp.columns = Datos.columns[:2]
            Datos=Datos.drop(Datos.columns[2:4],axis=1)
            Datos=Datos.append(temp).reset_index(drop=True)
            break
    return Datos





#%%



