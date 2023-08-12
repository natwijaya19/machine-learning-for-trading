import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import seaborn as sns

sns.set_style('whitegrid')

## Get data
# Get data from Fama French website
ff_factors: str = "F-F_Research_Data_5_Factors_2x3"
start: str = "2010-01-01"
end: str = "2017-12-31"
ff_factors_data: pd.DataFrame = web.DataReader(
    name=ff_factors, data_source="famafrench", start=start, end=end
)[0]
print(ff_factors_data.info())

# description
print(f"ff_factors_data.describe() = \n{ff_factors_data.describe()}\n")
print(f"ff_factors_data.head() = \n{ff_factors_data.head()}\n")

## Portfolio
# Get data from Ken French website
ff_portfolio: str = "17_Industry_Portfolios"
ff_portfolio_data: pd.DataFrame = web.DataReader(
    name=ff_portfolio, data_source="famafrench", start=start, end=end
)[0]

ff_portfolio_data_excess: pd.DataFrame = ff_portfolio_data.sub(
    ff_factors_data["RF"], axis=0
)
print(
    f"ff_portfolio_data_excess.head() = \n{ff_portfolio_data_excess.head()}\n")

# info
print(
    f"ff_portfolio_data_excess.info() = \n{ff_portfolio_data_excess.info()}\n")
