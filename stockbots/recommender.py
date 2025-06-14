#Importing Dependencies needed to build entire Recommender System
from __future__ import annotations
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from stockbots import data_loader
import numpy as np
import pandas as pd

#Development dependencies 
from pathlib import Path
from typing import List
from joblib import dump, load

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODELS_DIR.mkdir(exist_ok = True)

_fundamentals_raw = data_loader.load_fundamentals_df()

class StockRecommender:
    #Instantiation
    def __init__(self, tfidf_model: TfidfVectorizer, tfidf_matrix, cos_sim: np.array, svd: SVD, fundamentals: pd.DataFrame) -> None:
        self.tfidf_model = tfidf_model
        self.tfidf_matrix = tfidf_matrix
        self.cos_sim = cos_sim
        self.svd = svd
        self.fundamentals = fundamentals.reset_index(drop = True)

    @classmethod
    def fit_model(cls, fundamentals: pd.DataFrame | None = None) -> "StockRecommender":
        fundamentals = fundamentals.copy() if fundamentals is not None else _fundamentals_raw.copy()

        if "Symbol" not in fundamentals.columns:
            fundamentals = fundamentals.reset_index()
            

        if "Analyst Rating" not in fundamentals.columns:
            rating_map = {"buy": 4, "hold": 3, "underperform": 2, "none": 1}
            fundamentals["Analyst Rating"] = fundamentals["Analyst Recommendation"].map(rating_map)

        if "user_id" not in fundamentals.columns:
            fundamentals["user_id"] = 1

        col_list = ["user_id", "Symbol", "Analyst Rating"]
        ratings_df = fundamentals.loc[:, col_list]

        train_df = ratings_df[ratings_df["Analyst Rating"] >= 1].copy()

        reader = Reader(rating_scale = (1, 4))
        collab_data = Dataset.load_from_df(train_df, reader)
        model = SVD()
        model.fit(collab_data.build_full_trainset())


        fundamentals["Analyst Rating"] = fundamentals["Analyst Rating"].astype(float)


        masked_data = fundamentals["Analyst Rating"] == 1.0
        
        if masked_data.any():
            for idx in fundamentals[masked_data].index:
                user = fundamentals.at[idx, "user_id"]
                stock = fundamentals.at[idx, "Symbol"]
                pred_unknown_ratings = model.predict(user, stock)
                
                fundamentals.at[idx, "Analyst Rating"] = pred_unknown_ratings.est

        content_df = fundamentals.drop(columns = ["user_id", "Analyst Rating", "Analyst Recommendation"])
        content_df["Stock Profile"] = content_df.apply(lambda row: ' '.join(str(x) for x in row.values), axis = 1)

        tfidf_model = TfidfVectorizer(stop_words = "english")
        tfidf_matrix = tfidf_model.fit_transform(content_df["Stock Profile"])
        cos_sim_stocks = cosine_similarity(tfidf_matrix, tfidf_matrix)

        return cls(tfidf_model, tfidf_matrix, cos_sim_stocks, model, fundamentals)
    
    def recommend(self, ticker: str, k: int) -> List[str]:
        if ticker not in self.fundamentals["Symbol"].values:
            raise KeyError(f"{ticker} not included. Please try a stock within the S&P500.")
        
        stock_symbol_index = int(self.fundamentals.index[self.fundamentals["Symbol"] == ticker].to_numpy()[0])

        content_similar_stocks = list(enumerate(self.cos_sim[stock_symbol_index]))
        top_contnet_similar_stocks = sorted(content_similar_stocks, key = lambda x: x[1], reverse = True)

        top_k_content_stocks = [i for i, _ in top_contnet_similar_stocks[1 : k + 1]]

        combined_preds = []

        for i in top_k_content_stocks:
            symbol = self.fundamentals.iloc[i]["Symbol"]
            stock_latent_scores = self.svd.predict(1, symbol)
            combined_preds.append((i, stock_latent_scores.est))

        final_preds = sorted(combined_preds, key = lambda x: x[1], reverse = True)
        top_k_final_recs = [i for i, _ in final_preds]

        return self.fundamentals.iloc[top_k_final_recs]["Symbol"].to_list()
    
    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or _MODELS_DIR) / "recommender.pkl"
        dump(self, path)

        return path
    
    @classmethod
    def load(cls, path: str | Path) -> "StockRecommender":
        return load(Path(path))
