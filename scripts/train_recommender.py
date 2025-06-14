from stockbots.recommender import StockRecommender


if __name__ == "__main__":
    main_model = StockRecommender.fit_model()

    path = main_model.save()

    


