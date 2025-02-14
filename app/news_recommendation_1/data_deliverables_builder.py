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


    def execute(self, user_data_test: pl.DataFrame, user_data_train: pl.DataFrame, news_data: pl.DataFrame, similarity_matrix: pl.DataFrame):

        news_data = (news_data
        .select(
            pl.col("page"),
            pl.col("issued"),
            pl.col("cluster")
        )
        .filter(
            pl.col("issued") > pl.datetime(2022, 6, 1, 0, 0, 0, time_zone='UTC')
        ))

        self.explore(user_data_train, user_data_test, news_data, similarity_matrix)

        self.data_repo.save_polars_df_to_parquet(news_data, self.NEWS_FILE)
        self.data_repo.save_polars_df_to_parquet(similarity_matrix, self.MATRIX_FILE)

    @time_it
    def explore(self, user_data_train, user_data_test, news_data, similarity_matrix):
        # print()
        # print('+=========================+')
        # print('|          TRAIN          |')
        # print('+=========================+')
        # # print(user_data_train.describe())
        # # print(f'Number of unique user ids: {user_data_test["userId"].n_unique()} (Total rows: {user_data_test.shape[0]})')
        # # print(f'historySize Max/Min : {user_data_train["historySize"].min()} / {user_data_train["historySize"].max()}')
        # print()
        # print("HEAD")
        # print(user_data_train.head(5))
        # # print()
        # # print("TAIL")
        # # print(user_data_train.tail(5))

        # print()
        # print('+=========================+')
        # print('|           TEST          |')
        # print('+=========================+')
        # print(user_data_test.describe())
        # print(f'Number of unique user ids: {user_data_test["userId"].n_unique()} (Total rows: {user_data_test.shape[0]})')
        # print()
        # print("HEAD")
        # print(user_data_test.head(5))
        # print()
        # print("TAIL")
        # print(user_data_test.tail(5))

        print()
        print('+=========================+')
        print('|        NEWS DATA        |')
        print('+=========================+')
        # print(news_data.describe())
        # print()
        print("HEAD")
        print(news_data.head(5))
        print()
        print("TAIL")
        print(news_data.tail(1))
        print()

        return None
