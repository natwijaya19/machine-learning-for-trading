"""
This script is used to test the installation of Python and the required
packages.
"""

# Download AMZN stock data from Yahoo Finance by using pandas_datareader
# and plot the data by using matplotlib

import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf

sns.set()

start: datetime = datetime.datetime(2000, 1, 1)
end: datetime = datetime.datetime.today()

data: pd.DataFrame = yf.download("AMZN", start=start, end=end)
amzn: pd.DataFrame = data

if __name__ == '__main__':
    # print(f"AMZN data: {amzn}")
    print(f"AMZN data: {amzn.head()}")

    # Save the data to a csv file
    file_path = r"stock_data.csv"
    amzn.to_csv(file_path)

    # Plot the Adj Close column
    amzn["Adj Close"].plot(title="AMZN Adj Close", figsize=(10, 5), grid=True)
    plt.semilogy()
    plt.tight_layout()
    plt.show()

