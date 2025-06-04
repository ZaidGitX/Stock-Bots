#Importing Dependencies needed to build entire Recommender System
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from stockbots.data_loader import load_fundamentals_df
import seaborn as sns
import matplotlib.pyplot as plt


#If you will load final dataframe in with all the necessary data, you can type the following:
#fundamentals_df = pd.read_csv("fundamentals_df.csv") ## Load in your file path from wherever you stored the fundamentals_df csv
#fundamentals_df = pd.read_csv("/Users/zaid/Desktop/StockMarketProject/fundamentals_df.csv")

fundamentals_df = load_fundamentals_df()

### Constructing the Latent-Factor (Collaborative-filter) Model

ratings_dict = {"buy": 4, "hold": 3, "underperform": 2, "none": 1}
fundamentals_df["Analyst Rating"] = fundamentals_df["Analyst Recommendation"].map(ratings_dict)

fundamentals_df["user_id"] = 1
fundamentals_df.reset_index(inplace = True)

reader = Reader(rating_scale = (1, 4))
collab_data = Dataset.load_from_df(fundamentals_df[["user_id", "Symbol", "Analyst Rating"]], reader)

model = SVD()
model_eval = cross_validate(model, collab_data, measures = ["RMSE", "MAE"], cv = 5, verbose = True)
print(model_eval)


### Constructing the Content-Based Filtering (Cosine Similarity) Model

#Feature engineering the Stock Profile variable by aggregating all the string values into one value for each stock. We will then convert each 
#of the values for this newly feature engineered variable into a vector space with each value (all the characteristics for one stock) into a 
#single vector within a vector space. Finally, after completing this process, we can begin to determine cosine similairty between each vector
#representation of each stock's attributes (all features in dataset) and determine stocks that are most similar to one another. 

fundamentals_df.drop(columns = ["Analyst Recommendation", "Analyst Rating", "user_id"], inplace = True)

fundamentals_df["Stock Profile"] = fundamentals_df.apply(lambda row: ' '.join(str(x) for x in row), axis = 1)
tfidf_vectorizor = TfidfVectorizer(stop_words = "english")
vectorized_stock_profile = tfidf_vectorizor.fit_transform(fundamentals_df["Stock Profile"])

cos_sim_stocks = cosine_similarity(vectorized_stock_profile, vectorized_stock_profile)


#This function defines how we generate final stock recommendations. Initially, we identify the index for each stock ticker in fundamentals_df 
#and store these indices in a list. We then calculate the cosine similarity for each stock utilizing the derived index from the indices list, 
#obtaining a similarity score that compares every stock in the dataset to the target stock at the specified index. These scores are formatted 
#as tuples and compiled into a list, where each tuple indicates the similarity score between the target stock and another stock in the dataset. 
#Finally, we sort these scores in descending then order and select the top 10 to determine the stocks most similar to the target.


#After identifying the top 10 stocks most similar to a specified stock based on cosine similarity scores, we integrate these stocks into our 
#existing latent factor model (which incorporates elements of content-based filtering). This process involves utilizing the analyst ratings 
#for each of these top 10 stocks, thereby adding an additional layer of predictive power that combines both the intrinsic features of the 
#stocks and the insights from analysts' ratings. We then predict new scores for these relevant stocks, which are subsequently sorted in 
#descending order based on their scores. The final output of the model presents the top 10 stocks related to the specified stock, now evaluated
#based not only on their inherent characteristics but also influenced by the ratings provided by analysts.


def get_recommendations(stock, model):
    index = fundamentals_df.index[fundamentals_df["Symbol"] == stock].to_list()[0]
    stock_sim_scores = list(enumerate(cos_sim_stocks[index]))
    sorted_stock_sim_scores = sorted(stock_sim_scores, key = lambda x: x[1], reverse = True)
    top_10_stocks = [i[0] for i in sorted_stock_sim_scores[1:10]]

    latent_factor_preds = []

    for i in top_10_stocks:
        symbol = fundamentals_df.iloc[i]["Symbol"]
        stock_latent_scores = model.predict(1, symbol)
        latent_factor_preds.append((i, stock_latent_scores.est))

    final_recs = sorted(latent_factor_preds, key = lambda x: x[1], reverse = True)
    top_10_final_recs = [i[0] for i in final_recs]

    return fundamentals_df["Symbol"].iloc[top_10_final_recs]


model = SVD()
model.fit(collab_data.build_full_trainset())
get_recommendations("AAPL", model)




#Building a visualization of bar graphs that represent the similarities of our target stock feature valueswith their top 5 most relavent stock
#feature values. AAPL is our target stock in our case scenario. We will explore 3 key features we believe are of utmost importance:
#Earnings/Share, Price/Earnings, and 

listOfCompanies = ["AAPL", "ADSK", "MAS", "PHM", "IT", "HPQ"]

daCompanies = fundamentals_df[fundamentals_df["Symbol"].isin(listOfCompanies)]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))  # Adjust figsize as needed

sns.barplot(data = daCompanies, x = "Symbol", y = "Earnings/Share", ax=axes[0])
axes[0].set_title('Profit Margin Per Company')

sns.barplot(data = daCompanies, x = "Symbol", y = "Price/Earnings", ax=axes[1])
axes[1].set_title('Price/earnings Ratio Per Company')

sns.barplot(data = daCompanies, x = "Symbol", y = "Beta", ax=axes[2])
axes[2].set_title('Beta value per Company')

plt.tight_layout()
plt.show()



#These are the top 10 recommendations for a stock soley based off of the content-based filtering model predictions EXCLUDING 
#the latent factor model estimates

def get_recommendations(stock):
    index = fundamentals_df.index[fundamentals_df["Symbol"] == stock].to_list()[0]
    stock_sim_scores = list(enumerate(cos_sim_stocks[index]))
    sorted_stock_sim_scores = sorted(stock_sim_scores, key = lambda x: x[1], reverse = True)
    top_10_stocks = [i[0] for i in sorted_stock_sim_scores[1:10]]

    return fundamentals_df["Symbol"].iloc[top_10_stocks]

get_recommendations("AAPL")



#These are the top 10 recommendations for a stock soley based off of the latent factor model predictions EXCLUDING the content-based filtering
#model estimates

def get_recommendations_latent(stock, model):
    #index = fundamentals_df.index[fundamentals_df["Symbol"] == stock].to_list()[0]
    #stock_sim_scores = list(enumerate(cos_sim_stocks[index]))
    #sorted_stock_sim_scores = sorted(stock_sim_scores, key = lambda x: x[1], reverse = True)
    #top_10_stocks = [i[0] for i in sorted_stock_sim_scores[1:10]]

    latent_factor_preds = []

    for index, row in fundamentals_df.iterrows():
        stock = row["Symbol"]
        stock_latent_scores = model.predict(1, stock)
        latent_factor_preds.append((index, stock_latent_scores.est))

    final_recs = sorted(latent_factor_preds, key = lambda x: x[1], reverse = True)
    top_10_final_recs = [i[0] for i in final_recs[1:10]]

    return fundamentals_df["Symbol"].iloc[top_10_final_recs]


model = SVD()
model.fit(collab_data.build_full_trainset())
get_recommendations_latent("AAPL", model)
