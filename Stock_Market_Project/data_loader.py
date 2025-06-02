import pandas as pd
from pathlib import Path


#Reading in the two main pre-processed ready to go datasets we will work with

final_time_directory = Path(__file__).resolve().parent / "data_csvs" / "final_time_df.csv"



fundamentals_df_directory = Path(__file__).resolve().parent / "data_csvs" / "fundamentals_df.csv"






def load_final_time_df() -> pd.DataFrame:
    return pd.read_csv(final_time_directory)



def load_fundamentals_df() -> pd.DataFrame:
    return pd.read_csv(fundamentals_df_directory)




