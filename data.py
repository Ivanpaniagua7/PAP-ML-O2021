
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import yfinance as yf


def get_adj_close(tickers: str, start_date: str, end_date: str):
    closes = yf.download(tickers,
                         start=start_date,
                         end=end_date,
                         progress=False)['Adj Close'].reset_index(drop=True)

    return pd.DataFrame(closes)
