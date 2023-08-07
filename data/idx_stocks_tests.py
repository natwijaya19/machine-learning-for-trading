from typing import List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup


def scrape_indonesia_stock_exchange() -> None:
    url: str = "https://www.idx.co.id/en/listed-companies/company-profiles/"
    response: requests.Response = requests.get(url)

    soup: BeautifulSoup = BeautifulSoup(response.content, "html.parser")

    table: BeautifulSoup = soup.find("table", class_="table table-ellipsis")

    if table is not None:
        company_data: List[Dict[str, str]] = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            company_name: str = cols[1].text.strip()
            stock_symbol: str = cols[2].text.strip()
            sector: str = cols[3].text.strip()
            company_data.append(
                {"Company Name": company_name, "Stock Symbol": stock_symbol,
                 "Sector": sector})

        df: pd.DataFrame = pd.DataFrame(company_data)
        print(df)
        df.to_csv("indonesia_stock_exchange_companies.csv", index=False)
    else:
        print("Table not found. Please verify the class name.")


if __name__ == "__main__":
    scrape_indonesia_stock_exchange()
