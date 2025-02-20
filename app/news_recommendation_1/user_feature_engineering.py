import time
from datetime import datetime
import polars as pl
from app.news_recommendation_1 import time_it
from app.news_recommendation_1.data_repository import DataRepository


class UserFeatureEngineering:
    """
    A class to perform feature engineering on user data for the news recommendation system.

    Attributes
    ----------
    FEATURE_COUNT : int
        The number of features to generate.
    FORCE_REPROCESS : bool
        A flag to force reprocessing of the user data.
    FEATURE_WEIGHTS : dict
        A dictionary containing weights for different features.
    data_repo : DataRepository
        An instance of DataRepository to manage data loading and saving.

    Methods
    -------
    __init__(self, force_reprocess: bool, feature_count: int)
        Initializes the UserFeatureEngineering with the given parameters.
    execute(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, news_data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]
        Executes the feature engineering process on the user data.
    identify_history_clusters(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, news_data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]
        Identifies clusters in the user's history based on the news data.
    treat_timestamp_history(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]
        Treats the timestamp history to create a new feature.
    normalize_numeric_list_columns(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]
        Normalizes numeric list columns in the user data.
    filter_irrelevant_rows(self, user_data_train: pl.DataFrame) -> pl.DataFrame
        Filters out irrelevant rows from the training data.
    create_feature_and_target_columns(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]
        Creates feature and target columns in the user data.
    populate_train_feature_and_target_columns(self, user_data_train: pl.DataFrame) -> pl.DataFrame
        Populates the feature and target columns in the training data.
    populate_test_feature_and_target_columns(self, user_data_test: pl.DataFrame) -> pl.DataFrame
        Populates the feature and target columns in the test data.
    normalize(self, data: pl.Series) -> list
        Normalizes a list of numeric values.

    Usage Examples
    --------------
    >>> feature_engineering = UserFeatureEngineering(force_reprocess=True, feature_count=10)
    >>> user_data_train, user_data_test = feature_engineering.execute(user_data_train, user_data_test, news_data)
    """

    FEATURE_COUNT = 10
    FORCE_REPROCESS = False

    FEATURE_WEIGHTS = {
        'timeOnPageHistory_norm': 0,
        'scrollPercentageHistory_norm': 0,
        'pageVisitsCountHistory_norm': 0,
        'timestampHistory_norm': 1
    }

    def __init__(self, force_reprocess, feature_count):
        self.FORCE_REPROCESS = force_reprocess
        self.FEATURE_COUNT = feature_count
        self.data_repo = DataRepository()

    @time_it
    def execute(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, news_data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Executes the feature engineering process on the user data.

        This method checks if feature-engineered user data already exists and loads it if available and reprocessing is not forced.
        Otherwise, it applies a series of feature engineering steps to the user data.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training user data to be processed.
        user_data_test : pl.DataFrame
            The test user data to be processed.
        news_data : pl.DataFrame
            The news data to be used for clustering.

        Returns
        -------
        tuple
            A tuple containing the processed training and test user data.

        Example
        -------
        >>> feature_engineering = UserFeatureEngineering(force_reprocess=True, feature_count=10)
        >>> user_data_train, user_data_test = feature_engineering.execute(user_data_train, user_data_test, news_data)
        """
        if self.data_repo.feature_engineered_user_data_parquet_exists() and not self.FORCE_REPROCESS:
            user_data_train, user_data_test = self.data_repo.load_feature_engineered_user_data_from_parquet()
            print(user_data_train.head(5))
            print(user_data_test.head(5))
            return user_data_train, user_data_test

        # user_data_train = user_data_train.select(pl.all().gather(range(100, 130)))
        # user_data_test = user_data_test.select(pl.all().gather(range(1)))

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
    def identify_history_clusters(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, news_data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Identifies clusters in the user's history based on the news data.

        This method replaces the pages in the user's history with their corresponding clusters from the news data.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training user data to be processed.
        user_data_test : pl.DataFrame
            The test user data to be processed.
        news_data : pl.DataFrame
            The news data to be used for clustering.

        Returns
        -------
        tuple
            A tuple containing the processed training and test user data.

        Example
        -------
        >>> feature_engineering = UserFeatureEngineering(force_reprocess=True, feature_count=10)
        >>> user_data_train, user_data_test = feature_engineering.identify_history_clusters(user_data_train, user_data_test, news_data)
        """

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
    def treat_timestamp_history(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Treats the timestamp history to create a new feature.

        This method processes the timestamp history by calculating the time difference between each timestamp and the last timestamp, and creating a new feature based on this difference.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training user data to be processed.
        user_data_test : pl.DataFrame
            The test user data to be processed.

        Returns
        -------
        tuple
            A tuple containing the processed training and test user data.

        Example
        -------
        >>> feature_engineering = UserFeatureEngineering(force_reprocess=True, feature_count=10)
        >>> user_data_train, user_data_test = feature_engineering.treat_timestamp_history(user_data_train, user_data_test)
        """
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
    def normalize_numeric_list_columns(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Normalizes numeric list columns in the user data.

        This method normalizes the numeric list columns in the user data by dividing each value by the maximum value in the list.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training user data to be processed.
        user_data_test : pl.DataFrame
            The test user data to be processed.

        Returns
        -------
        tuple
            A tuple containing the processed training and test user data.

        Example
        -------
        >>> feature_engineering = UserFeatureEngineering(force_reprocess=True, feature_count=10)
        >>> user_data_train, user_data_test = feature_engineering.normalize_numeric_list_columns(user_data_train, user_data_test)
        """
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
    def filter_irrelevant_rows(self, user_data_train: pl.DataFrame) -> pl.DataFrame:
        """
        Filters out irrelevant rows from the training data.

        This method filters out rows from the training data where the maximum value in the 'timestampHistory_norm' column is below a certain threshold.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training user data to be processed.

        Returns
        -------
        pl.DataFrame
            The filtered training user data.

        Example
        -------
        >>> feature_engineering = UserFeatureEngineering(force_reprocess=True, feature_count=10)
        >>> user_data_train = feature_engineering.filter_irrelevant_rows(user_data_train)
        """
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
    def create_feature_and_target_columns(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Creates feature and target columns in the user data.

        This method initializes feature columns for each cluster and target columns for the cluster and page in both training and test data.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training user data to be processed.
        user_data_test : pl.DataFrame
            The test user data to be processed.

        Returns
        -------
        tuple
            A tuple containing the user data with the new feature and target columns.

        Example
        -------
        >>> feature_engineering = UserFeatureEngineering(force_reprocess=True, feature_count=10)
        >>> user_data_train, user_data_test = feature_engineering.create_feature_and_target_columns(user_data_train, user_data_test)
        """
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

        return user_data_train, user_data_test

    @time_it
    def populate_train_feature_and_target_columns(self, user_data_train: pl.DataFrame) -> pl.DataFrame:
        """
        Populates the feature and target columns in the training data.

        This method iterates through the user's history and assigns weights to the feature columns based on the user's interaction with each cluster.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training user data to be processed.

        Returns
        -------
        pl.DataFrame
            The training user data with populated feature and target columns.

        Example
        -------
        >>> feature_engineering = UserFeatureEngineering(force_reprocess=True, feature_count=10)
        >>> user_data_train = feature_engineering.populate_train_feature_and_target_columns(user_data_train)
        >>> print(user_data_train)
        """
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
                print(f"Row {row_index} / {number_of_rows} processed. Progress: {row_index/number_of_rows*100:.2f}%. Elapsed per 1000 rows: {sum(times)*1000/len(times):.2f}s")
                times = []

        user_data_train = user_data_train.with_columns([
            pl.col('timestampHistory').map_elements(lambda x: datetime.fromtimestamp(x[-1]/1000), return_dtype=pl.Datetime).alias('target_timestamp')
        ])

        print()
        print(user_data_train.head(5))

        return user_data_train

    @time_it
    def populate_test_feature_and_target_columns(self, user_data_test: pl.DataFrame) -> pl.DataFrame:
        """
        Populates the feature and target columns in the test data.

        This method iterates through the user's history and assigns weights to the feature columns based on the user's interaction with each cluster.

        Parameters
        ----------
        user_data_test : pl.DataFrame
            The test user data to be processed.

        Returns
        -------
        pl.DataFrame
            The test user data with populated feature and target columns.

        Example
        -------
        >>> feature_engineering = UserFeatureEngineering(force_reprocess=True, feature_count=10)
        >>> user_data_test = feature_engineering.populate_test_feature_and_target_columns(user_data_test)
        >>> print(user_data_test)
        """
        number_of_rows = user_data_test.shape[0]
        times = []
        print()

        for row_index, row in enumerate(user_data_test.iter_rows(named=True)):
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
                print(f"Row {row_index} / {number_of_rows} processed. Progress: {row_index/number_of_rows*100:.2f}%. Elapsed per 1000 rows: {sum(times)*1000/len(times):.2f}s")
                times = []

        user_data_test = user_data_test.with_columns([
            pl.col('timestampHistory').map_elements(lambda x: datetime.fromtimestamp(x[-1]/1000000), return_dtype=pl.Datetime).alias('target_timestamp')
        ])

        print()
        print(user_data_test.head(5))

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

    def normalize(self, data: pl.Series) -> list:
        """
        Normalizes a list of numeric values.

        This method normalizes the numeric values in the list by dividing each value by the maximum value in the list.

        Parameters
        ----------
        data : pl.Series
            The list of numeric values to be normalized.

        Returns
        -------
        list
            The normalized list of numeric values.

        Example
        -------
        >>> feature_engineering = UserFeatureEngineering(force_reprocess=True, feature_count=10)
        >>> normalized_data = feature_engineering.normalize(pl.Series([1, 2, 3, 4, 5]))
        """
        result = []
        max_value = data.max()
        for value in data:
            if max_value > 0:
                result.append(value / max_value)
            else:
                result.append(0)
        return result