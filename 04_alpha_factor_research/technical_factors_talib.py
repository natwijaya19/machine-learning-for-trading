import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import talib
import yfinance as yf
from pandas import DataFrame

sns.set_style('darkgrid')

## Download historical data for required stocks
tickers = ["MSFT", "AAPL", ]

ohlcv_data = {}
start_date = '2015-01-01'
end_date = '2020-12-31'
for ticker in tickers:
    ohlcv_data[ticker] = yf.download(ticker, start_date, end_date)

# Create a dictionary to store all the technical indicators
aapl: pd.DataFrame = ohlcv_data['AAPL']

# bolinger bands
aapl['upper_band'], aapl['middle_band'], aapl['lower_band'] = talib.BBANDS(
    aapl['Close'], timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)

# Relative Strength Index
aapl['rsi'] = talib.RSI(aapl['Close'], timeperiod=14)

# Plot the charts and apply some styling: 2 rows, 1 column
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

## Plot the upper and lower bollinger bands
data: pd.DataFrame = aapl[['Close', 'upper_band', 'middle_band', 'lower_band']]
data.plot(ax=ax[0], title='Apple Close Price & Bollinger Bands', grid=True)

# Plot the RSI indicator
ax[1].plot(aapl['rsi'], color='black')
ax[1].set_title('Apple RSI Indicator')
ax[1].axhline(y=70, linestyle='--', linewidth=1, color='black')
ax[1].axhline(y=30, linestyle='--', linewidth=1, color='black')

plt.tight_layout()
plt.show()
