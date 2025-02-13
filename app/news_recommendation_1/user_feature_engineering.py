import time
from datetime import datetime

import polars as pl
from sympy.polys.matrices.sdm import unop_dict

from app.news_recommendation_1 import time_it
from app.news_recommendation_1.data_repository import DataRepository


class UserFeatureEngineering:

    FEATURE_COUNT = 10
    FORCE_REPROCESS = False

    FEATURE_WEIGHTS = {
        'timeOnPageHistory_norm': 0,
        'scrollPercentageHistory_norm': 0,
        'pageVisitsCountHistory_norm': 0,
        'timestampHistory_norm': 10
    }

    def __init__(self, force_reprocess, feature_count):
        self.FORCE_REPROCESS = force_reprocess
        self.FEATURE_COUNT = feature_count
        self.data_repo = DataRepository()

    @time_it
    def execute(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, news_data: pl.DataFrame):
        if self.data_repo.feature_engineered_user_data_parquet_exists() and not self.FORCE_REPROCESS:
            user_data_train, user_data_test = self.data_repo.load_feature_engineered_user_data_from_parquet()
            print(user_data_train.head(5))
            print(user_data_test.head(5))
            return user_data_train, user_data_test

        # user_data_train = user_data_train.select(pl.all().gather(range(50)))
        # user_data_test = user_data_test.select(pl.all().gather(range(50)))

        user_data_train, user_data_test = self.identify_history_clusters(user_data_train, user_data_test, news_data)
        user_data_train, user_data_test = self.treat_timestamp_history(user_data_train, user_data_test)
        user_data_train, user_data_test = self.normalize_numeric_list_columns(user_data_train, user_data_test)
        user_data_train = self.filter_irrelevant_rows(user_data_train)
        user_data_train, user_data_test = self.create_feature_and_target_columns(user_data_train, user_data_test)
        user_data_train = self.populate_train_feature_and_target_columns(user_data_train)
        user_data_test = self.populate_test_feature_and_target_columns(user_data_test)

        self.data_repo.save_feature_engineered_user_data_to_parquet(user_data_train, user_data_test)

        return user_data_train, user_data_test

    @time_it
    def identify_history_clusters(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, news_data: pl.DataFrame):
        news_data_dict = news_data.select(['page', 'cluster']).to_dicts()
        news_data_dict = {x['page']: x['cluster'] for x in news_data_dict}
        unknown_pages = []

        def replace_page_by_cluster(pages: pl.Series):
            data = []
            for page in pages:
                if page in news_data_dict:
                    data.append(news_data_dict[page])
                else:
                    unknown_pages.append(page)
                    data.append(None)
            return data

        user_data_train = user_data_train.with_columns([
            pl.col('history').map_elements(lambda x: replace_page_by_cluster(x), return_dtype=pl.List(pl.Int32)).alias('clusterHistory')
        ])

        user_data_test = user_data_test.with_columns([
            pl.col('history').map_elements(lambda x: replace_page_by_cluster(x), return_dtype=pl.List(pl.Int32)).alias('clusterHistory')
        ])

        print(f"Unknown pages: {len(unknown_pages)}")
        for p in unknown_pages[:20]:
            print(f"--> {p} <--")

        return user_data_train, user_data_test

    @time_it
    def treat_timestamp_history(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame):
        def treat(pages: pl.Series):
            data = []

            for page in pages:
                time_diff_seconds = time.mktime(time.localtime(pages[-1] / 1000)) - time.mktime(time.localtime(page / 1000))
                hours, remainder = divmod(time_diff_seconds, 3600)
                index = 100 - hours*10
                index = index if index > 10 else 10
                data.append(index)

            return data

        user_data_train = user_data_train.with_columns([
            pl.col('timestampHistory').map_elements(lambda x: treat(x), return_dtype=pl.List(pl.Float32)).alias('timestampHistory_treated')
        ])

        user_data_test = user_data_test.with_columns([
            pl.col('timestampHistory').map_elements(lambda x: treat(x), return_dtype=pl.List(pl.Float32)).alias('timestampHistory_treated')
        ])

        return user_data_train, user_data_test

    @time_it
    def normalize_numeric_list_columns(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame):
        user_data_train = user_data_train.with_columns([
            pl.col('numberOfClicksHistory').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('numberOfClicksHistory_norm'),
            pl.col('timeOnPageHistory').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('timeOnPageHistory_norm'),
            pl.col('scrollPercentageHistory').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('scrollPercentageHistory_norm'),
            pl.col('pageVisitsCountHistory').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('pageVisitsCountHistory_norm'),
            pl.col('timestampHistory_treated').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('timestampHistory_norm'),
        ])

        user_data_test = user_data_test.with_columns([
            pl.col('timestampHistory_treated').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('timestampHistory_norm'),
        ])

        return user_data_train, user_data_test

    @time_it
    def filter_irrelevant_rows(self, user_data_train: pl.DataFrame):
        initial_count = user_data_train.shape[0]

        threshold = 0.5

        user_data_train = user_data_train.with_columns([
            pl.col('timestampHistory_norm').list.reverse().alias('max_timestampHistory_treated')
                .list.slice(1, -1).alias('max_timestampHistory_treated')
        ])

        user_data_train = user_data_train.filter(
            pl.col('max_timestampHistory_treated').list.max() >= threshold
        )

        final_count = user_data_train.shape[0]
        discarded_count = initial_count - final_count
        print(f'Number of discarded rows: {discarded_count} (Initial: {initial_count}, Final: {final_count})')

        user_data_train.drop_in_place('max_timestampHistory_treated')
        return user_data_train

    @time_it
    def create_feature_and_target_columns(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame):
        cols_train = [pl.zeros(user_data_train.shape[0], pl.Float32, eager=True).alias(f'cluster_{i}') for i in range(self.FEATURE_COUNT)]
        cols_test = [pl.zeros(user_data_test.shape[0], pl.Float32, eager=True).alias(f'cluster_{i}') for i in range(self.FEATURE_COUNT)]

        targets_train = [
            pl.zeros(user_data_train.shape[0], pl.Int32, eager=True).alias(f'target_cluster'),
            pl.lit("None").alias("target_page")
        ]

        targets_test = [
            pl.zeros(user_data_test.shape[0], pl.Int32, eager=True).alias(f'target_cluster'),
            pl.lit("None").alias("target_page")
        ]

        user_data_train = user_data_train.with_columns(cols_train + targets_train)
        user_data_test = user_data_test.with_columns(cols_test + targets_test)

        # print(user_data_train.head(5))
        return user_data_train, user_data_test

    # @time_it
    # def populate_train_feature_and_target_columns(self, user_data_train: pl.DataFrame):
    #     # Define a function to calculate the weight for each history entry
    #     def calculate_weight(row):
    #         weights = [
    #             row['timeOnPageHistory_norm'] * self.FEATURE_WEIGHTS['timeOnPageHistory_norm'],
    #             row['scrollPercentageHistory_norm'] * self.FEATURE_WEIGHTS['scrollPercentageHistory_norm'],
    #             row['pageVisitsCountHistory_norm'] * self.FEATURE_WEIGHTS['pageVisitsCountHistory_norm'],
    #             row['timestampHistory_norm'] * self.FEATURE_WEIGHTS['timestampHistory_norm'],
    #         ]
    #         return sum(weights) / sum(self.FEATURE_WEIGHTS.values())
    #
    #     # Apply the weight calculation and update the cluster columns
    #     for i in range(self.FEATURE_COUNT):
    #         user_data_train = user_data_train.with_columns([
    #             (pl.col('clusterHistory').list.eval(
    #                 pl.element().map_elements(lambda history: calculate_weight(history), return_dtype=pl.Float32)
    #             ) / pl.col('clusterHistory').list.eval(
    #                 pl.element().map_elements(lambda history: pl.col('clusterHistory').list.count(history), return_dtype=pl.Float32)
    #             )).alias(f'cluster_{i}')
    #         ])
    #
    #     # Set the target cluster and target page
    #     user_data_train = user_data_train.with_columns([
    #         pl.col('clusterHistory').list.last().alias('target_cluster'),
    #         pl.col('history').list.last().alias('target_page'),
    #         pl.col('timestampHistory').list.last().map_elements(lambda x: datetime.fromtimestamp(x / 1000),
    #                                                            return_dtype=pl.Datetime).alias('target_timestamp')
    #     ])
    #
    #     print()
    #     print(user_data_train.head(5))
    #
    #     return user_data_train

    @time_it
    def populate_train_feature_and_target_columns(self, user_data_train: pl.DataFrame):
        # get number of rows on datafram
        number_of_rows = user_data_train.shape[0]
        times = []
        print()

        for row_index, row in enumerate(user_data_train.iter_rows(named=True)):
            # measure iteration time
            start = time.time()
            for history_index, history in enumerate(row['clusterHistory']):
                if history_index == len(row['clusterHistory']) - 1:
                    user_data_train[row_index, f'target_cluster'] = history
                    user_data_train[row_index, f'target_page'] = row['history'][history_index]
                    continue

                data = [
                    row['timeOnPageHistory_norm'][history_index]*self.FEATURE_WEIGHTS['timeOnPageHistory_norm'],
                    row['scrollPercentageHistory_norm'][history_index]*self.FEATURE_WEIGHTS['scrollPercentageHistory_norm'],
                    row['pageVisitsCountHistory_norm'][history_index]*self.FEATURE_WEIGHTS['pageVisitsCountHistory_norm'],
                    row['timestampHistory_norm'][history_index]*self.FEATURE_WEIGHTS['timestampHistory_norm'],
                ]

                weight = sum(data) / sum(self.FEATURE_WEIGHTS.values())
                occurrences = row['clusterHistory'][:-1].count(history)

                user_data_train[row_index, f"cluster_{history}"] += 1*weight/occurrences

            end = time.time()
            times.append(end - start)

            print_every = 1000
            if row_index % print_every == 0:
                # print rows index, numer of rows, percentage progress and average time
                print(f"Row {row_index} / {number_of_rows} processed. Progress: {row_index/number_of_rows*100:.2f}%. Elapsed per 1000 rows: {sum(times)*1000/len(times):.2f}s")
                times = []

        user_data_train = user_data_train.with_columns([
            pl.col('timestampHistory').map_elements(lambda x: datetime.fromtimestamp(x[-1]/1000), return_dtype=pl.Datetime).alias('target_timestamp')
        ])

        print()
        print(user_data_train.head(5))

        return user_data_train

    @time_it
    def populate_test_feature_and_target_columns(self, user_data_test: pl.DataFrame):
        # get number of rows on datafram
        number_of_rows = user_data_test.shape[0]
        times = []
        print()

        for row_index, row in enumerate(user_data_test.iter_rows(named=True)):
            # measure iteration time
            start = time.time()
            for history_index, history in enumerate(row['clusterHistory']):
                if history_index == len(row['clusterHistory']) - 1:
                    user_data_test[row_index, f'target_cluster'] = history
                    user_data_test[row_index, f'target_page'] = row['history'][history_index]
                    continue

                if history is None:
                    continue

                occurrences = row['clusterHistory'][:-1].count(history)
                user_data_test[row_index, f"cluster_{history}"] += 1 * row['timestampHistory_norm'][history_index] / occurrences

            end = time.time()
            times.append(end - start)

            print_every = 1000
            if row_index % print_every == 0:
                # print rows index, numer of rows, percentage progress and average time
                print(f"Row {row_index} / {number_of_rows} processed. Progress: {row_index/number_of_rows*100:.2f}%. Elapsed per 1000 rows: {sum(times)*1000/len(times):.2f}s")
                times = []

        user_data_test = user_data_test.with_columns([
            pl.col('timestampHistory').map_elements(lambda x: datetime.fromtimestamp(x[-1]/1000000), return_dtype=pl.Datetime).alias('target_timestamp')
        ])

        print()
        print(user_data_test.head(20))

        invalid_target_cluster_count = user_data_test.filter(
            pl.col('target_cluster').is_null() | pl.col('target_cluster').cast(pl.Utf8).str.contains(r'\D')
        ).shape[0]
        total_rows = user_data_test.shape[0]
        percent_invalid = (invalid_target_cluster_count / total_rows) * 100
        print(f"Percentage of rows with invalid target_cluster: {percent_invalid:.2f}%")

        user_data_test = user_data_test.filter(
            ~(pl.col('target_cluster').is_null() | pl.col('target_cluster').cast(pl.Utf8).str.contains(r'\D'))
        )
        final_rows = user_data_test.shape[0]
        discarded_count = total_rows - final_rows
        print(f"Number of discarded rows (invalid target_cluster): {discarded_count} (Initial: {total_rows}, Final: {final_rows})")

        return user_data_test

    def normalize(self, data: pl.Series):
        result = []
        # max_value = data[:-1].max()
        max_value = data.max()
        for value in data:
            if max_value > 0:
                result.append(value / max_value)
            else:
                result.append(0)
        return result