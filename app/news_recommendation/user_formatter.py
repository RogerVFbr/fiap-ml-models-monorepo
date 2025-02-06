import time

import polars as pl

from app.news_recommendation import time_it
from app.news_recommendation.data_repository import DataRepository


class UserFormatter:

    FEATURE_COUNT = 10
    FORCE_REPROCESS = False

    def __init__(self, force_reprocess, feature_count):
        self.FORCE_REPROCESS = force_reprocess
        self.FEATURE_COUNT = feature_count
        self.data_repo = DataRepository()

    @time_it
    def execute(self, user_data_train: pl.DataFrame, news_data: pl.DataFrame):
        if self.data_repo.formatted_user_data_parquet_exists() and not self.FORCE_REPROCESS:
            data = self.data_repo.load_formatted_user_data_from_parquet()
            print(data.head(5))
            return data

        # user_data_train = user_data_train.select(pl.all().gather(range(100000)))
        # user_data_train = user_data_train.select(pl.all().gather([2, 3, 4]))
        # user_data_train = user_data_train.select(pl.all().gather([3]))

        user_data_train = self.identify_history_clusters(user_data_train, news_data)
        user_data_train = self.normalize_numeric_list_columns(user_data_train)
        user_data_train = self.create_feature_and_target_columns(user_data_train)
        user_data_train = self.populate_feature_and_target_columns(user_data_train)

        self.data_repo.save_formatted_user_data_to_parquet(user_data_train)

        return user_data_train

    @time_it
    def identify_history_clusters(self, user_data_train: pl.DataFrame, news_data: pl.DataFrame):
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
            pl.col('history').map_elements(lambda x: replace_page_by_cluster(x), return_dtype=pl.List(pl.Int32)).alias('clusterHistory')
        ])

        # print(user_data_train.head(5))

        return user_data_train

    @time_it
    def normalize_numeric_list_columns(self, user_data_train: pl.DataFrame):
        user_data_train = user_data_train.with_columns([
            pl.col('numberOfClicksHistory').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('numberOfClicksHistory_norm'),
            pl.col('timeOnPageHistory').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('timeOnPageHistory_norm'),
            pl.col('scrollPercentageHistory').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('scrollPercentageHistory_norm'),
            pl.col('pageVisitsCountHistory').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('pageVisitsCountHistory_norm'),
            pl.col('timestampHistory').map_elements(lambda x: self.normalize(x), return_dtype=pl.List(pl.Float32)).alias('timestampHistory_norm'),
        ])

        print(user_data_train.head(5))

        return user_data_train

    @time_it
    def create_feature_and_target_columns(self, user_data_train: pl.DataFrame):
        cols = [pl.zeros(user_data_train.shape[0], pl.Float32, eager=True).alias(f'cluster_{i}') for i in range(self.FEATURE_COUNT)]

        targets = [
            pl.zeros(user_data_train.shape[0], pl.Int32, eager=True).alias(f'target_cluster'),
            pl.lit("None").alias("target_page")
        ]

        user_data_train = user_data_train.with_columns(cols + targets)

        print(user_data_train.head(5))
        return user_data_train

    @time_it
    def populate_feature_and_target_columns(self, user_data_train: pl.DataFrame):
        # get number of rows on dataframe
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
                    # row['numberOfClicksHistory_norm'][history_index],
                    row['timeOnPageHistory_norm'][history_index],
                    row['scrollPercentageHistory_norm'][history_index],
                    row['pageVisitsCountHistory_norm'][history_index],
                    row['timestampHistory_norm'][history_index]
                ]

                weight = sum(data) / len(data)
                occurrences = row['clusterHistory'][:-1].count(history)

                user_data_train[row_index, f"cluster_{history}"] += 1*weight/occurrences

            end = time.time()
            times.append(end - start)

            print_every = 1000
            if row_index % print_every == 0:
                # print rows index, numer of rows, percentage progress and average time
                print(f"Row {row_index} / {number_of_rows} processed. Progress: {row_index/number_of_rows*100:.2f}%. Elapsed per 1000 rows: {sum(times)*1000/len(times):.2f}s")
                times = []

        print()
        print(user_data_train.head(5))

        return user_data_train

    # import concurrent.futures

    # @time_it
    # def populate_feature_and_target_columns(self, user_data_train: pl.DataFrame):
    #     import concurrent.futures
    #     number_of_rows = user_data_train.shape[0]
    #     print()
    #
    #     def process_row(row_index, row):
    #         for history_index, history in enumerate(row['clusterHistory']):
    #             if history_index == len(row['clusterHistory']) - 1:
    #                 user_data_train[row_index, f'target_cluster'] = history
    #                 user_data_train[row_index, f'target_page'] = row['history'][history_index]
    #                 continue
    #
    #             data = [
    #                 row['numberOfClicksHistory_norm'][history_index],
    #                 row['timeOnPageHistory_norm'][history_index],
    #                 row['scrollPercentageHistory_norm'][history_index],
    #                 row['pageVisitsCountHistory_norm'][history_index],
    #                 row['timestampHistory_norm'][history_index]
    #             ]
    #
    #             weight = sum(data) / len(data)
    #             occurrences = row['clusterHistory'][:-1].count(history)
    #
    #             user_data_train[row_index, f"cluster_{history}"] += 1 * weight / occurrences
    #
    #         if row_index % 1000 == 0:
    #             print(
    #                 f"Row {row_index} / {number_of_rows} processed. Progress: {row_index / number_of_rows * 100:.2f}%")
    #
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = [executor.submit(process_row, row_index, row) for row_index, row in
    #                    enumerate(user_data_train.iter_rows(named=True))]
    #         for future in concurrent.futures.as_completed(futures):
    #             future.result()
    #
    #     print()
    #     print(user_data_train.head(5))
    #
    #     return user_data_train

    def normalize(self, data: pl.Series):
        result = []
        max_value = data[:-1].max()
        for value in data:
            if max_value > 0:
                result.append(value / max_value)
            else:
                result.append(0)
        return result