
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

#%% Ivan











#%% Edzna
import data as dt
import numpy as np
import functions as fn
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})


# Download adjust close of Bitcoin from yahoo
btc = dt.get_adj_close("BTC-USD", start_date="2019-09-27", end_date="2021-09-27")
#print(btc)

# Calcular las diferencias de la serie de tiempo
diff1 = btc["Adj Close"].diff()
diff2 = diff1.diff()


# Realizar la prueba Dickey-Fuller a la serie de tiempo
Di_Fu0 = fn.prueba_DickeyFuller(btc)
Di_Fu1 = fn.prueba_DickeyFuller(diff1)

# Diferenciacion y autocorrelacion
#autocorr = fn.autocorrelacion(btc, diff1, diff2)

# Resumen de diferenciaciones de la serie de tiempo
resumen = fn.resumen(Di_Fu0, Di_Fu1)
#print(resumen)

# Encontrar el parametro "p"
#AR = fn.AR(diff2)

# Encontrar el parametro "q"
MA = fn.MA(diff1)