import os

from app.news_recommendation_1 import time_it
from app.news_recommendation_1.data_repository import DataRepository
import polars as pl

class DataDeliverablesBuilder:

    def __init__(self, output_path: str):
        self.data_repo = DataRepository(output_path)

        self.DATA_PATH = "data-betelgeuse"
        self.NEWS_FILE = os.path.join(self.DATA_PATH, 'news_data')
        self.MATRIX_FILE = os.path.join(self.DATA_PATH, 'similarity_matrix')
        self.WEIGHTS_FILE = os.path.join(self.DATA_PATH, 'feature_weights')

    def execute(self, user_data_test: pl.DataFrame, user_data_train: pl.DataFrame, news_data: pl.DataFrame, similarity_matrix: pl.DataFrame, feature_weights: dict):
        news_data = self.filter_news_data(news_data)

        # self.explore(user_data_train, user_data_test, news_data, similarity_matrix)

        self.data_repo.save_polars_df_to_parquet(news_data, self.NEWS_FILE)
        self.data_repo.save_polars_df_to_parquet(similarity_matrix, self.MATRIX_FILE)
        self.data_repo.save_polars_df_to_parquet(pl.DataFrame(feature_weights), self.WEIGHTS_FILE)

    @staticmethod
    def filter_news_data(news_data: pl.DataFrame):
        return (news_data
        .select(
            pl.col("page"),
            pl.col("issued"),
            pl.col("cluster")
        )
        .filter(
            pl.col("issued") > pl.datetime(2022, 6, 1, 0, 0, 0, time_zone='UTC')
        ))

    @time_it
    def explore(self, df):
        print("EXPLORE")
        print("HEAD")
        print(df.head(5))
        print()
        print("TAIL")
        print(df.tail(1))
        print()

        return None
