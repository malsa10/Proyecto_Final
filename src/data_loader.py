# data_loader.py
import yfinance as yf
import pandas as pd
from datetime import datetime

def descargar_datos(ticker, start="2020-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data
