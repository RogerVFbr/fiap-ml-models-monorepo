import os

from app.news_recommendation_1.data_repository import DataRepository
import polars as pl

class DataDeliverablesBuilder:
    """
    A class to build and save data deliverables for the news recommendation system.

    Attributes
    ----------
    data_repo : DataRepository
        An instance of DataRepository to manage data loading and saving.
    DATA_PATH : str
        The base path for data storage.
    NEWS_FILE : str
        The file path for saving filtered news data.
    MATRIX_FILE : str
        The file path for saving the similarity matrix.
    WEIGHTS_FILE : str
        The file path for saving feature weights.

    Methods
    -------
    __init__(self, output_path: str)
        Initializes the DataDeliverablesBuilder with the specified output path.
    execute(self, user_data_test: pl.DataFrame, user_data_train: pl.DataFrame, news_data: pl.DataFrame, similarity_matrix: pl.DataFrame, feature_weights: dict) -> None
        Executes the process of filtering news data and saving the deliverables.
    filter_news_data(news_data: pl.DataFrame) -> pl.DataFrame
        Filters the news data to include only relevant columns and rows based on the issued date.

    Usage Examples
    --------------
    >>> builder = DataDeliverablesBuilder(output_path='output/path')
    >>> builder.execute(user_data_test, user_data_train, news_data, similarity_matrix, feature_weights)
    """
    def __init__(self, output_path: str):
        """
        Initializes the DataDeliverablesBuilder with the specified output path.

        Parameters
        ----------
        output_path : str
            The base path for saving the output data.
        """
        self.data_repo = DataRepository(output_path)

        self.DATA_PATH = "data-betelgeuse"
        self.NEWS_FILE = os.path.join(self.DATA_PATH, 'news_data')
        self.MATRIX_FILE = os.path.join(self.DATA_PATH, 'similarity_matrix')
        self.WEIGHTS_FILE = os.path.join(self.DATA_PATH, 'feature_weights')

    def execute(self, user_data_test: pl.DataFrame, user_data_train: pl.DataFrame, news_data: pl.DataFrame, similarity_matrix: pl.DataFrame, feature_weights: dict) -> None:
        """
        Executes the process of filtering news data and saving the deliverables.

        This method filters the news data to include only relevant columns and rows based on the issued date, and then saves the filtered news data, similarity matrix, and feature weights to Parquet files.

        Parameters
        ----------
        user_data_test : pl.DataFrame
            The test user data.
        user_data_train : pl.DataFrame
            The training user data.
        news_data : pl.DataFrame
            The news data to be filtered and saved.
        similarity_matrix : pl.DataFrame
            The similarity matrix to be saved.
        feature_weights : dict
            The feature weights to be saved.

        Returns
        -------
        None

        Example
        -------
        >>> builder = DataDeliverablesBuilder(output_path='output/path')
        >>> builder.execute(user_data_test, user_data_train, news_data, similarity_matrix, feature_weights)
        """
        news_data = self.filter_news_data(news_data)
        self.data_repo.save_polars_df_to_parquet(news_data, self.NEWS_FILE)
        self.data_repo.save_polars_df_to_parquet(similarity_matrix, self.MATRIX_FILE)
        self.data_repo.save_polars_df_to_parquet(pl.DataFrame(feature_weights), self.WEIGHTS_FILE)

    @staticmethod
    def filter_news_data(news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Filters the news data to include only relevant columns and rows based on the issued date.

        This method selects the 'page', 'issued', and 'cluster' columns from the news data and filters the rows to include only those issued after June 1, 2022.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be filtered.

        Returns
        -------
        pl.DataFrame
            The filtered news data.

        Example
        -------
        >>> filtered_news_data = DataDeliverablesBuilder.filter_news_data(news_data)
        """
        return (news_data
        .select(
            pl.col("page"),
            pl.col("issued"),
            pl.col("cluster")
        )
        .filter(
            pl.col("issued") > pl.datetime(2022, 6, 1, 0, 0, 0, time_zone='UTC')
        ))
