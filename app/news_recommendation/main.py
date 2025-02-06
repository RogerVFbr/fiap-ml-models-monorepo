import polars as pl
import pandas as pd

from app.news_recommendation import time_it
from app.news_recommendation.data_repository import DataRepository
from app.news_recommendation.news_cluster_inference_train import NewsClusterInferenceTrain
from app.news_recommendation.news_processor import NewsProcessor
from app.news_recommendation.news_classifier import NewsClassifier
from app.news_recommendation.user_formatter import UserFormatter


class NewsRecommenderSystem:
    FORCE_REPROCESS_NEWS_PROCESSOR = False
    FORCE_REPROCESS_NEWS_CLASSIFIER = False
    FORCE_REPROCESS_USER_FORMATTER = False

    NO_OF_CLUSTERS = 10

    def __init__(self):
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.width', 1000)

        pl.Config.set_tbl_cols(1000)
        pl.Config.set_tbl_width_chars(230)
        pl.Config.set_fmt_str_lengths(700)
        pl.Config.set_tbl_rows(10)
        pl.Config.set_fmt_table_cell_list_len(700)
        pl.Config.set_tbl_column_data_type_inline(True)

        self.data_repo = DataRepository()
        self.news_processor = NewsProcessor(self.FORCE_REPROCESS_NEWS_PROCESSOR)
        self.news_classifier = NewsClassifier(self.FORCE_REPROCESS_NEWS_CLASSIFIER, self.NO_OF_CLUSTERS)
        self.user_formatter = UserFormatter(self.FORCE_REPROCESS_USER_FORMATTER, self.NO_OF_CLUSTERS)
        self.news_cluster_inference_train = NewsClusterInferenceTrain()

    @time_it
    def run(self):
        # user_data_train, user_data_test, news_data = self.dataset_repo.load_dataset_from_csv()
        # self.dataset_repo.save_dataset_to_parquet(user_data_train, user_data_test, news_data)

        user_data_train, user_data_test, news_data = self.data_repo.load_dataset_from_parquet()
        user_data_train, user_data_test, news_data = self.adjust_datatypes(user_data_train, user_data_test, news_data)
        user_data_train, user_data_test, news_data = self.filter(user_data_train, user_data_test, news_data)
        user_data_train, user_data_test, news_data = self.sanitize(user_data_train, user_data_test, news_data)

        news_data = self.news_processor.execute(news_data)
        news_data, similarity_matrix = self.news_classifier.execute(news_data)

        user_data_train = self.user_formatter.execute(user_data_train, news_data)

        self.news_cluster_inference_train.execute(user_data_train)

        # self.explore(user_data_train, user_data_test, news_data)

        # self.train(user_data_train, user_data_test, news_data)

        # self.plot_exploratory_analysis(user_data_train)

    @time_it
    def adjust_datatypes(self, user_data_train, user_data_test, news_data):
        user_data_train = user_data_train.with_columns([
            pl.col('userType').cast(pl.Categorical),
            pl.col('historySize').cast(pl.Int32),
            pl.col('history').str.split(', '),
            # (pl.col('timestampHistory').str.split(', ').cast(pl.List(pl.Int64)) * 1000).cast(pl.List(pl.Datetime)),
            pl.col('timestampHistory').str.split(', ').cast(pl.List(pl.Int64)),
            pl.col('numberOfClicksHistory').str.split(', ').cast(pl.List(pl.Int32)),
            pl.col('timeOnPageHistory').str.split(', ').cast(pl.List(pl.Int32)),
            pl.col('scrollPercentageHistory').str.split(', ').cast(pl.List(pl.Float64)),
            pl.col('pageVisitsCountHistory').str.split(', ').cast(pl.List(pl.Int32)),
        ])

        user_data_test = user_data_test.with_columns([
            pl.col('userType').cast(pl.Categorical),
            pl.col('history').str.replace_all(r"[^a-zA-Z0-9-\s]", "").str.split(' '),
            (pl.col('timestampHistory').str.replace_all(r"[^a-zA-Z0-9-\s]", "").str.split(' ').cast(pl.List(pl.Int64)) * 1000).cast(pl.List(pl.Datetime))
        ])

        news_data = news_data.with_columns([
            pl.col('issued').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z"),
            pl.col('modified').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        ])

        return user_data_train, user_data_test, news_data

    @time_it
    def filter(self, user_data_train, user_data_test, news_data):
        # TRAIN
        initial_count = user_data_train.shape[0]

        user_data_train = user_data_train.filter(pl.col('historySize') >= 2)
        user_data_train = user_data_train.filter(pl.col('history').list.len() == pl.col('historySize'))
        user_data_train = user_data_train.filter(pl.col('timestampHistory').list.len() == pl.col('historySize'))
        user_data_train = user_data_train.filter(pl.col('numberOfClicksHistory').list.len() == pl.col('historySize'))
        user_data_train = user_data_train.filter(pl.col('timeOnPageHistory').list.len() == pl.col('historySize'))
        user_data_train = user_data_train.filter(pl.col('scrollPercentageHistory').list.len() == pl.col('historySize'))
        user_data_train = user_data_train.filter(pl.col('pageVisitsCountHistory').list.len() == pl.col('historySize'))

        final_count = user_data_train.shape[0]
        discarded_count = initial_count - final_count
        print(f'TRAIN ...: Number of discarded rows: {discarded_count} (Initial: {initial_count}, Final: {final_count})')

        # TEST
        initial_count = user_data_test.shape[0]

        user_data_test = user_data_test.filter(pl.col('history').list.len() >= 2)

        final_count = user_data_test.shape[0]
        discarded_count = initial_count - final_count
        print(f'TEST ....: Number of discarded rows: {discarded_count} (Initial: {initial_count}, Final: {final_count})')

        # NEWS
        initial_count = news_data.shape[0]

        news_data = news_data.unique(subset=['page'])

        final_count = news_data.shape[0]
        discarded_count = initial_count - final_count
        print(f'NEWS ....: Number of discarded rows: {discarded_count} (Initial: {initial_count}, Final: {final_count})')

        return user_data_train, user_data_test, news_data

    @time_it
    def sanitize(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, news_data: pl.DataFrame):
        user_data_train.drop_in_place('timestampHistory_new')
        return user_data_train, user_data_test, news_data

    @time_it
    def format_user_data(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, news_data: pl.DataFrame):

        news_data_dict = news_data.select(['page', 'cluster']).to_dicts()
        news_data_dict = {x['page']: x['cluster'] for x in news_data_dict}

        def replace_page_by_cluster(pages: pl.Series):
            data = []
            for page in pages:
                if page in news_data_dict:
                    data.append(news_data_dict[page])
                else:
                    data.append(None)
            return data

        user_data_train = user_data_train.with_columns([
            pl.col('history').map_elements(lambda x: replace_page_by_cluster(x), return_dtype=pl.List(pl.Int32)).alias('history_cluster')
        ])


        print(user_data_train.head(5))

        return user_data_train, user_data_test

    @time_it
    def explore(self, user_data_train, user_data_test, news_data):
        print()
        print('+=========================+')
        print('|          TRAIN          |')
        print('+=========================+')
        # print(user_data_train.describe())
        # print(f'Number of unique user ids: {user_data_test["userId"].n_unique()} (Total rows: {user_data_test.shape[0]})')
        # print(f'historySize Max/Min : {user_data_train["historySize"].min()} / {user_data_train["historySize"].max()}')
        print()
        print("HEAD")
        print(user_data_train.head(5))
        # print()
        # print("TAIL")
        # print(user_data_train.tail(5))

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
        #
        print()
        print('+=========================+')
        print('|        NEWS DATA        |')
        print('+=========================+')
        # print(news_data.describe())
        # print()
        print("HEAD")
        print(news_data.head(1))
        print()
        print("TAIL")
        print(news_data.tail(1))
        print()


if __name__ == "__main__":
    train = NewsRecommenderSystem()
    train.run()