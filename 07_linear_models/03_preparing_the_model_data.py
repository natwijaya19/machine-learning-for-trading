import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import seaborn as sns
import yfinance as yf

sns.set_style('darkgrid')

## Define parameters
MONTH: int = 21
YEAR: int = 12 * MONTH

START: str = '2015-01-01'
END: str = '2020-01-01'
idx = pd.IndexSlice

TICKERS: list[str] = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA',
                      'NVDA', 'PYPL', 'ADBE', 'NFLX', 'INTC', 'CMCSA',
                      'PEP', 'COST', 'TMUS', 'AVGO', 'TXN', 'CHTR', 'QCOM', ]

## Get the data
data: pd.DataFrame = yf.download(TICKERS, start=START, end=END)

## Copy the data
df: pd.DataFrame = data.copy()
print(f"df.info():\n{df.info()}")

## Plot close price of each ticker scaled to 100
(df['Adj Close'] / df['Adj Close'].iloc[0] * 100).plot(
    figsize=(12, 6), title='Close Price')
plt.legend()
plt.tight_layout()
plt.show()
