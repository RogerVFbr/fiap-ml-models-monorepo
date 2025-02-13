from app.news_recommendation_2 import time_it
import polars as pl
from app.news_recommendation_2.data_repository import DataRepository

class DataService:
    """
    A service class to handle data operations for the news recommendation system.

    Attributes
    ----------
    data_repo : DataRepository
        An instance of DataRepository to manage data loading and saving.
    """

    def __init__(self):
        """
        Initializes the DataService with an instance of DataRepository.
        """
        self.data_repo = DataRepository()

    @time_it
    def onboard(self):
        """
        Loads, adjusts, filters, and sanitizes the datasets.

        Returns
        -------
        tuple
            A tuple containing the processed training data, testing data, and news data as Polars DataFrames.

        Example
        -------
        >>> service = DataService()
        >>> train_data, test_data, news_data = service.onboard()
        """
        user_data_train, user_data_test, news_data = self.data_repo.load_dataset_from_parquet()
        user_data_train, user_data_test, news_data = self.adjust_datatypes(user_data_train, user_data_test, news_data)
        user_data_train, user_data_test, news_data = self.filter(user_data_train, user_data_test, news_data)
        user_data_train, user_data_test, news_data = self.sanitize(user_data_train, user_data_test, news_data)

        return user_data_train, user_data_test, news_data

    @time_it
    def adjust_datatypes(self, user_data_train, user_data_test, news_data):
        """
        Adjusts the data types of columns in the datasets.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training data.
        user_data_test : pl.DataFrame
            The testing data.
        news_data : pl.DataFrame
            The news data.

        Returns
        -------
        tuple
            A tuple containing the adjusted training data, testing data, and news data as Polars DataFrames.
        """
        return (
            self.adjust_user_data_train_datatypes(user_data_train),
            self.adjust_user_data_test_datatypes(user_data_test),
            self.adjust_news_data_datatypes(news_data)
        )

    @time_it
    def adjust_user_data_train_datatypes(self, user_data_train: pl.DataFrame) -> pl.DataFrame:
        """
        Adjusts the data types of columns in the training data.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training data.

        Returns
        -------
        pl.DataFrame
            The adjusted training data.
        """
        user_data_train = user_data_train.with_columns([
            pl.col('userType').cast(pl.Categorical),
            pl.col('historySize').cast(pl.Int32),
            pl.col('history').str.split(', '),
            (pl.col('timestampHistory').str.split(', ').cast(pl.List(pl.Int64)) * 1000).cast(pl.List(pl.Datetime)),
            pl.col('numberOfClicksHistory').str.split(', ').cast(pl.List(pl.Int32)),
            pl.col('timeOnPageHistory').str.split(', ').cast(pl.List(pl.Int32)),
            pl.col('scrollPercentageHistory').str.split(', ').cast(pl.List(pl.Float64)),
            pl.col('pageVisitsCountHistory').str.split(', ').cast(pl.List(pl.Int32)),
        ])
        return user_data_train

    @time_it
    def adjust_user_data_test_datatypes(self, user_data_test: pl.DataFrame) -> pl.DataFrame:
        """
        Adjusts the data types of columns in the testing data.

        Parameters
        ----------
        user_data_test : pl.DataFrame
            The testing data.

        Returns
        -------
        pl.DataFrame
            The adjusted testing data.
        """
        user_data_test = user_data_test.with_columns([
            pl.col('userType').cast(pl.Categorical),
            pl.col('history').str.replace_all(r"[^a-zA-Z0-9-\s]", "").str.split('\n '),
            (pl.col('timestampHistory').str.replace_all(r"[^a-zA-Z0-9-\s]", "").str.split(' ').cast(
                pl.List(pl.Int64)) * 1000).cast(pl.List(pl.Int64)).cast(pl.List(pl.Datetime))
        ])
        return user_data_test

    @time_it
    def adjust_news_data_datatypes(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Adjusts the data types of columns in the news data.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data.

        Returns
        -------
        pl.DataFrame
            The adjusted news data.
        """
        news_data = news_data.with_columns([
            pl.col('issued').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z"),
            pl.col('modified').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        ])
        return news_data

    @time_it
    def filter(self, user_data_train, user_data_test, news_data):
        """
        Filters the datasets to ensure data consistency.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training data.
        user_data_test : pl.DataFrame
            The testing data.
        news_data : pl.DataFrame
            The news data.

        Returns
        -------
        tuple
            A tuple containing the filtered training data, testing data, and news data as Polars DataFrames.
        """
        return (
            self.filter_user_data_train(user_data_train),
            self.filter_user_data_test(user_data_test),
            self.filter_news_data(news_data)
        )

    @time_it
    def filter_user_data_train(self, user_data_train: pl.DataFrame) -> pl.DataFrame:
        """
        Filters the training data to ensure data consistency.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training data.

        Returns
        -------
        pl.DataFrame
            The filtered training data.
        """
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
        print(
            f'TRAIN ...: Number of discarded rows: {discarded_count} (Initial: {initial_count}, Final: {final_count})')

        return user_data_train

    @time_it
    def filter_user_data_test(self, user_data_test: pl.DataFrame) -> pl.DataFrame:
        """
        Filters the testing data to ensure data consistency.

        Parameters
        ----------
        user_data_test : pl.DataFrame
            The testing data.

        Returns
        -------
        pl.DataFrame
            The filtered testing data.
        """
        initial_count = user_data_test.shape[0]
        user_data_test = user_data_test.filter(pl.col('history').list.len() >= 2)
        final_count = user_data_test.shape[0]
        discarded_count = initial_count - final_count
        print(
            f'TEST ....: Number of discarded rows: {discarded_count} (Initial: {initial_count}, Final: {final_count})')

        return user_data_test

    @time_it
    def filter_news_data(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Filters the news data to remove duplicates.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data.

        Returns
        -------
        pl.DataFrame
            The filtered news data.
        """
        initial_count = news_data.shape[0]
        news_data = news_data.unique(subset=['page'])
        final_count = news_data.shape[0]
        discarded_count = initial_count - final_count
        print(
            f'NEWS ....: Number of discarded rows: {discarded_count} (Initial: {initial_count}, Final: {final_count})')

        return news_data

    @time_it
    def sanitize(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, news_data: pl.DataFrame):
        """
        Sanitizes the datasets by removing unnecessary columns.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training data.
        user_data_test : pl.DataFrame
            The testing data.
        news_data : pl.DataFrame
            The news data.

        Returns
        -------
        tuple
            A tuple containing the sanitized training data, testing data, and news data as Polars DataFrames.
        """
        user_data_train.drop_in_place('timestampHistory_new')
        return user_data_train, user_data_test, news_data