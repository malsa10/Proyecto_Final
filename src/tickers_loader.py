# tickers_loader.py
import pandas as pd


def cargar_tickers(ruta="data/tickers_ampliado.csv"):
    """Carga la lista de tickers desde un archivo CSV."""
    df = pd.read_csv(ruta)
    return df


def buscar_tickers(filtro, df_tickers):
    """Filtra tickers por coincidencia parcial en nombre o ticker."""
    filtro = filtro.lower()
    resultado = df_tickers[df_tickers.apply(lambda row:
        filtro in str(row['Ticker']).lower() or filtro in str(row['Nombre']).lower(), axis=1)]
    return resultado.reset_index(drop=True)
