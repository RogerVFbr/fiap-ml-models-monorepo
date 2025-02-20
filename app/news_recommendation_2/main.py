from app import ARGS
from app.news_recommendation_2.news_reccomendation_system import NewsRecommenderSystem

if __name__ == "__main__":
    NewsRecommenderSystem.OUTPUT_PATH = ARGS.output
    train = NewsRecommenderSystem()
    train.run()