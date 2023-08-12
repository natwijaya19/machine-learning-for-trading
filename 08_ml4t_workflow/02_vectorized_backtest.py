import pandas_datareader.data as web
from matplotlib import pyplot as plt
import datetime as dt
name: str = "SP500"
datasource: str = "fred"
start: str = "2010-01-01"
end: str = dt.datetime.today().strftime("%Y-%m-%d")
sp500 = web.DataReader(
    name=name, data_source=datasource, start=start, end=end)

# Plot the SP500 data
sp500.plot(title="SP500", y="SP500")
plt.show()

## Get list of SP500 stocks from Wikipedia
import pandas as pd
url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables: pd.DataFrame = pd.read_html(url)
sp500_stocks: pd.DataFrame = tables[0]
sp500_stocks.to_csv("sp500_stocks.csv", index=False)
print(sp500_stocks.head())

# Get data for all SP500 stocks
import pandas_datareader.data as web
from matplotlib import pyplot as plt
import datetime as dt
import pandas as pd
sp500_stocks: pd.DataFrame = pd.read_csv("sp500_stocks.csv")
start: str = "2010-01-01"
end: str = dt.datetime.today().strftime("%Y-%m-%d")
for symbol in sp500_stocks["Symbol"]:
    try:
        df = web.DataReader(
            name=symbol, data_source="yahoo", start=start, end=end)
        df.to_csv(f"data/{symbol}.csv")
    except:
        print(f"Error getting data for {symbol}")

