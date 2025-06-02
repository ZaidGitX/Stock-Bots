import pandas as pd
import os


#Reading in the two main pre-processed ready to go datasets we will work with
fundamentals_df = pd.read_csv("fundamentals_df.csv")
final_time_df = pd.read_csv("final_time_df.csv")


def load_final_time_df():
    filepath_final_time_df = os.path.join("data_csvs", final_time_df)
    return filepath_final_time_df


def load_fundamentals_df():
    filepath_fundamentals_df = os.path.join("data_csvs", fundamentals_df)
    return filepath_fundamentals_df






