import pandas as pd
from pathlib import Path
from functools import lru_cache


#Reading in the two main pre-processed ready to go datasets we will work with


_data_dir = Path(__file__).resolve().parent.parent / "data_csvs"




@lru_cache(maxsize = 2)
def load_final_time_df() -> pd.DataFrame:
    df = pd.read_csv(_data_dir / "final_time_df.csv", parse_dates = ["Date"])
    return df.set_index("Date").sort_index()



@lru_cache(maxsize = 2)
def load_fundamentals_df() -> pd.DataFrame:
    df = pd.read_csv(_data_dir / "fundamentals_df.csv")
    df.drop(columns = ["Unnamed: 0"], inplace = True)
    return df.set_index("Symbol")



def get_ticker_price(ticker: str) -> pd.Series:
    df = load_final_time_df()
    if ticker not in df:
        raise KeyError(f"{ticker} not included. Please try a ticker within the S&P 500 rotation of 2022.")
    return df[ticker]


def load_covered_tickers() -> list[str]:
    return sorted(set(load_final_time_df().columns[:-1]).intersection(load_fundamentals_df().index))

    

















