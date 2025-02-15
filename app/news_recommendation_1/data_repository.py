import os
import polars as pl
from app.news_recommendation_1 import time_it
import torch
from datetime import datetime
import numpy as np
import joblib

from app.news_recommendation_1.s3_client import S3Client


class DataRepository:
    """
    A class to manage the loading, saving, and processing of datasets-origin-bkp for a news recommendation system.

    Attributes
    ----------
    ORIGIN_DATA_PATH : str
        Base path for the dataset.
    PROCESSED_DATA_PATH : str
        Path for datasets-origin-bkp-preprocessed data.
    ORIGIN_BASE_DATA_PATH : str
        Full path to the base data directory.
    PRE_PROCESSED_BASE_PATH : str
        Full path to the base datasets-origin-bkp-preprocessed data directory.
    TRAIN_DIR : str
        Directory for training data files.
    TEST_DIR : str
        Directory for testing data files.
    NEWS_DIR : str
        Directory for news data files.
    TRAIN_FILE : str
        Path to the training data Parquet file.
    TEST_FILE : str
        Path to the testing data Parquet file.
    NEWS_FILE : str
        Path to the news data Parquet file.
    NEWS_PRE_PROCESSED_FILE : str
        Path to the datasets-origin-bkp-preprocessed news data Parquet file.
    NEWS_CLASSIFIED_FILE : str
        Path to the classified news data Parquet file.
    SIMILARITY_MATRIX_FILE : str
        Path to the similarity matrix Parquet file.
    FORMATTED_USER_DATA_TRAIN_FILE : str
        Path to the formatted user data Parquet file.
    """
    OUTPUT_PATH = None
    CURRENT_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def __init__(self, output_path: str = None):
        """
        Initializes the DataRepository with paths for data storage.
        """
        self.s3_client = S3Client()

        self.OUTPUT_PATH = output_path
        self.BUCKET_NAME = "fiap-ml-models-monorepo-bucket-develop"
        self.DATASETS_REMOTE_PREFIX = "datasets"

        self.ORIGIN_DATA_PATH = "challenge-webmedia-e-globo-2023"
        self.JOINED_DATA_PATH = "datasets-origin"
        self.PROCESSED_DATA_PATH = "datasets-preprocessed"

        self.ORIGIN_BASE_DATA_PATH = os.path.join(os.getcwd(), self.ORIGIN_DATA_PATH)
        self.JOINED_BASE_DATA_PATH = os.path.join(os.getcwd(), self.JOINED_DATA_PATH)
        self.PRE_PROCESSED_BASE_PATH = os.path.join(os.getcwd(), self.PROCESSED_DATA_PATH)

        self.TRAIN_DIR = os.path.join(self.ORIGIN_BASE_DATA_PATH, 'files', 'treino')
        self.TEST_DIR = os.path.join(self.ORIGIN_BASE_DATA_PATH, 'files', 'validacao')
        self.NEWS_DIR = os.path.join(self.ORIGIN_BASE_DATA_PATH, 'itens', 'itens')

        self.TRAIN_FILE = os.path.join(self.JOINED_DATA_PATH, 'user_data_train.parquet')
        self.TEST_FILE = os.path.join(self.JOINED_DATA_PATH, 'user_data_test.parquet')
        self.NEWS_FILE = os.path.join(self.JOINED_DATA_PATH, 'item_data.parquet')

        self.NEWS_PRE_PROCESSED_FILE = os.path.join(self.PRE_PROCESSED_BASE_PATH, 'news_data_preprocessed.parquet')
        self.NEWS_CLASSIFIED_FILE = os.path.join(self.PRE_PROCESSED_BASE_PATH, 'news_data_classified.parquet')
        self.SIMILARITY_MATRIX_FILE = os.path.join(self.PRE_PROCESSED_BASE_PATH, 'similarity_matrix.parquet')
        self.FORMATTED_USER_DATA_TRAIN_FILE = os.path.join(self.PRE_PROCESSED_BASE_PATH, 'user_data_train_formatted.parquet')
        self.FORMATTED_USER_DATA_TEST_FILE = os.path.join(self.PRE_PROCESSED_BASE_PATH, 'user_data_test_formatted.parquet')
        self.PREDICTED_CLASSIC_USER_DATA_TEST_FILE = os.path.join(self.PRE_PROCESSED_BASE_PATH, 'user_data_test_predicted_classic.parquet')
        self.PREDICTED_NN_USER_DATA_TEST_FILE = os.path.join(self.PRE_PROCESSED_BASE_PATH, 'user_data_test_predicted_nn.parquet')

    # +===============================+
    # |         FULL DATASETS         |
    # +===============================+

    def load_dataset_from_csv(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Loads datasets-origin-bkp from CSV files and concatenates them into single dataframes.

        Returns
        -------
        tuple
            A tuple containing the training data, testing data, and news data as Polars DataFrames.

        Example
        -------
        >>> repo = DataRepository()
        >>> train_data, test_data, news_data = repo.load_dataset_from_csv()
        """
        user_files_train = [f for f in os.listdir(self.TRAIN_DIR) if f.startswith('treino_parte') and f.endswith('.csv')]
        user_data_train = pl.concat([pl.read_csv(os.path.join(self.TRAIN_DIR, f), infer_schema=False) for f in user_files_train])
        train_size = sum(os.path.getsize(os.path.join(self.TRAIN_DIR, f)) for f in user_files_train)

        user_files_test = [f for f in os.listdir(self.TEST_DIR) if f.startswith('validacao') and f.endswith('.csv')]
        user_data_test = pl.concat([pl.read_csv(os.path.join(self.TEST_DIR, f), infer_schema=False) for f in user_files_test])
        test_size = sum(os.path.getsize(os.path.join(self.TEST_DIR, f)) for f in user_files_test)

        item_files = [f for f in os.listdir(self.NEWS_DIR) if f.startswith('itens-parte') and f.endswith('.csv')]
        item_data = pl.concat([pl.read_csv(os.path.join(self.NEWS_DIR, f)) for f in item_files])
        item_size = sum(os.path.getsize(os.path.join(self.NEWS_DIR, f)) for f in item_files)

        print(f"Total train files size: {train_size / (1024 * 1024):.2f} MB")
        print(f"Total test files size: {test_size / (1024 * 1024):.2f} MB")
        print(f"Total item files size: {item_size / (1024 * 1024):.2f} MB")

        return user_data_train, user_data_test, item_data

    @time_it
    def save_dataset_to_parquet(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, item_data: pl.DataFrame) -> None:
        """
        Saves the datasets-origin-bkp to Parquet files.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The training data to be saved.
        user_data_test : pl.DataFrame
            The testing data to be saved.
        item_data : pl.DataFrame
            The news data to be saved.

        Example
        -------
        >>> repo = DataRepository()
        >>> repo.save_dataset_to_parquet(train_data, test_data, news_data)
        """
        user_data_train.write_parquet(self.TRAIN_FILE)
        user_data_test.write_parquet(self.TEST_FILE)
        item_data.write_parquet(self.NEWS_FILE)

        self.print_file_size(self.TRAIN_FILE)
        self.print_file_size(self.TEST_FILE)
        self.print_file_size(self.NEWS_FILE)

    def parquet_files_exist(self) -> bool:
        if os.path.exists(self.TRAIN_FILE) and os.path.exists(self.TEST_FILE) and os.path.exists(self.NEWS_FILE):
            print(f"User data train parquet exists.")
            print(f"User data test parquet exists.")
            print(f"News data parquet exists.")
            return True
        return False

    def download_parquet_files_from_s3(self):
       self.s3_client.download_folder_from_s3(self.BUCKET_NAME, self.DATASETS_REMOTE_PREFIX, self.JOINED_BASE_DATA_PATH)

    @time_it
    def load_dataset_from_parquet(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Loads the datasets-origin-bkp from Parquet files.

        Returns
        -------
        tuple
            A tuple containing the training data, testing data, and news data as Polars DataFrames.

        Example
        -------
        >>> repo = DataRepository()
        >>> train_data, test_data, news_data = repo.load_dataset_from_parquet()
        """
        user_data_train = pl.read_parquet(self.TRAIN_FILE)
        user_data_test = pl.read_parquet(self.TEST_FILE)
        item_data = pl.read_parquet(self.NEWS_FILE)

        self.print_file_size(self.TRAIN_FILE)
        self.print_file_size(self.TEST_FILE)
        self.print_file_size(self.NEWS_FILE)

        return user_data_train, user_data_test, item_data

    # +===============================+
    # |   NEWS PREPROCESSED PARQUET   |
    # +===============================+

    def preprocessed_news_parquet_exists(self) -> bool:
        """
        Checks if the datasets-origin-bkp-preprocessed news Parquet file exists.

        Returns
        -------
        bool
            True if the datasets-origin-bkp-preprocessed news Parquet file exists, False otherwise.

        Example
        -------
        >>> repo = DataRepository()
        >>> exists = repo.preprocessed_news_parquet_exists()
        """
        if os.path.exists(self.NEWS_PRE_PROCESSED_FILE):
            print(f"Pre processed news parquet exists.")
            return True
        return False

    @time_it
    def save_preprocessed_news_to_parquet(self, news_data: pl.DataFrame) -> None:
        """
        Saves the datasets-origin-bkp-preprocessed news data to a Parquet file.

        Parameters
        ----------
        news_data : pl.DataFrame
            The datasets-origin-bkp-preprocessed news data to be saved.

        Example
        -------
        >>> repo = DataRepository()
        >>> repo.save_preprocessed_news_to_parquet(news_data)
        """
        if not os.path.exists(self.PRE_PROCESSED_BASE_PATH):
            os.makedirs(self.PRE_PROCESSED_BASE_PATH)

        news_data.write_parquet(self.NEWS_PRE_PROCESSED_FILE)
        self.print_file_size(self.NEWS_PRE_PROCESSED_FILE)

    @time_it
    def load_preprocessed_news_from_parquet(self) -> pl.DataFrame:
        """
        Loads the datasets-origin-bkp-preprocessed news data from a Parquet file.

        Returns
        -------
        pl.DataFrame
            The datasets-origin-bkp-preprocessed news data.

        Example
        -------
        >>> repo = DataRepository()
        >>> news_data = repo.load_preprocessed_news_from_parquet()
        """
        self.print_file_size(self.NEWS_PRE_PROCESSED_FILE)
        return pl.read_parquet(self.NEWS_PRE_PROCESSED_FILE)

    # +===============================+
    # |    NEWS CLASSIFIED PARQUET    |
    # +===============================+

    def classified_news_parquet_exists(self) -> bool:
        """
        Checks if the classified news and similarity matrix Parquet files exist.

        Returns
        -------
        bool
            True if both the classified news and similarity matrix Parquet files exist, False otherwise.

        Example
        -------
        >>> repo = DataRepository()
        >>> exists = repo.classified_news_parquet_exists()
        """
        if os.path.exists(self.NEWS_CLASSIFIED_FILE) and os.path.exists(self.SIMILARITY_MATRIX_FILE):
            print(f"Classified news parquet exists.")
            print(f"Similarity matrix parquet exists.")
            return True
        return False

    @time_it
    def save_classified_news_to_parquet(self, news_data: pl.DataFrame, similarity_matrix: pl.DataFrame) -> None:
        """
        Saves the classified news data and similarity matrix to Parquet files.

        Parameters
        ----------
        news_data : pl.DataFrame
            The classified news data to be saved.
        similarity_matrix : pl.DataFrame
            The similarity matrix to be saved.

        Example
        -------
        >>> repo = DataRepository()
        >>> repo.save_classified_news_to_parquet(news_data, similarity_matrix)
        """
        if not os.path.exists(self.PRE_PROCESSED_BASE_PATH):
            os.makedirs(self.PRE_PROCESSED_BASE_PATH)

        news_data.write_parquet(self.NEWS_CLASSIFIED_FILE)
        similarity_matrix.write_parquet(self.SIMILARITY_MATRIX_FILE)
        self.print_file_size(self.NEWS_CLASSIFIED_FILE)
        self.print_file_size(self.SIMILARITY_MATRIX_FILE)

    @time_it
    def load_classified_news_from_parquet(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Loads the classified news data and similarity matrix from Parquet files.

        Returns
        -------
        tuple
            A tuple containing the classified news data and similarity matrix as Polars DataFrames.

        Example
        -------
        >>> repo = DataRepository()
        >>> news_data, similarity_matrix = repo.load_classified_news_from_parquet()
        """
        news = pl.read_parquet(self.NEWS_CLASSIFIED_FILE)
        similarity_matrix = pl.read_parquet(self.SIMILARITY_MATRIX_FILE)
        self.print_file_size(self.NEWS_CLASSIFIED_FILE)
        self.print_file_size(self.SIMILARITY_MATRIX_FILE)
        return news, similarity_matrix

    # +===============================+
    # |      USER DATA FORMATTED      |
    # +===============================+

    def feature_engineered_user_data_parquet_exists(self) -> bool:
        """
        Checks if the formatted user data Parquet file exists.

        Returns
        -------
        bool
            True if the formatted user data Parquet file exists, False otherwise.

        Example
        -------
        >>> repo = DataRepository()
        >>> exists = repo.feature_engineered_user_data_parquet_exists()
        """
        if os.path.exists(self.FORMATTED_USER_DATA_TRAIN_FILE) and os.path.exists(self.FORMATTED_USER_DATA_TEST_FILE):
            print(f"Formatted user data parquet exists.")
            return True
        return False

    @time_it
    def save_feature_engineered_user_data_to_parquet(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame) -> None:
        """
        Saves the formatted user data to a Parquet file.

        Parameters
        ----------
        user_data_train : pl.DataFrame
            The formatted user data to be saved.

        Example
        -------
        >>> repo = DataRepository()
        >>> repo.save_feature_engineered_user_data_to_parquet(user_data_train)
        """
        if not os.path.exists(self.PRE_PROCESSED_BASE_PATH):
            os.makedirs(self.PRE_PROCESSED_BASE_PATH)

        user_data_train.write_parquet(self.FORMATTED_USER_DATA_TRAIN_FILE)
        user_data_test.write_parquet(self.FORMATTED_USER_DATA_TEST_FILE)
        self.print_file_size(self.FORMATTED_USER_DATA_TRAIN_FILE)
        self.print_file_size(self.FORMATTED_USER_DATA_TEST_FILE)

    @time_it
    def load_feature_engineered_user_data_from_parquet(self) -> (pl.DataFrame, pl.DataFrame):
        """
        Loads the formatted user data from a Parquet file.

        Returns
        -------
        pl.DataFrame
            The formatted user data.

        Example
        -------
        >>> repo = DataRepository()
        >>> user_data = repo.load_feature_engineered_user_data_from_parquet()
        """
        self.print_file_size(self.FORMATTED_USER_DATA_TRAIN_FILE)
        self.print_file_size(self.FORMATTED_USER_DATA_TEST_FILE)
        train = pl.read_parquet(self.FORMATTED_USER_DATA_TRAIN_FILE)
        test = pl.read_parquet(self.FORMATTED_USER_DATA_TEST_FILE)
        return train, test

    # +===============================+
    # |   CLUSTER PREDICTOR CLASSIC   |
    # +===============================+

    def predicted_classic_user_data_test_parquet_exists(self) -> bool:
        if os.path.exists(self.PREDICTED_CLASSIC_USER_DATA_TEST_FILE):
            print(f"Predicted classic user data test parquet exists.")
            return True
        return False

    @time_it
    def save_predicted_classic_user_data_test_to_parquet(self, user_data_test: pl.DataFrame) -> None:
        if not os.path.exists(self.PRE_PROCESSED_BASE_PATH):
            os.makedirs(self.PRE_PROCESSED_BASE_PATH)

        user_data_test.write_parquet(self.PREDICTED_CLASSIC_USER_DATA_TEST_FILE)
        self.print_file_size(self.PREDICTED_CLASSIC_USER_DATA_TEST_FILE)

    @time_it
    def load_predicted_classic_user_data_test_from_parquet(self) -> pl.DataFrame:
        self.print_file_size(self.PREDICTED_CLASSIC_USER_DATA_TEST_FILE)
        return pl.read_parquet(self.PREDICTED_CLASSIC_USER_DATA_TEST_FILE)

    # +===============================+
    # |      CLUSTER PREDICTOR NN     |
    # +===============================+

    def predicted_nn_user_data_test_parquet_exists(self) -> bool:
        if os.path.exists(self.PREDICTED_NN_USER_DATA_TEST_FILE):
            print(f"Predicted neural network user data test parquet exists.")
            return True
        return False

    @time_it
    def save_predicted_nn_user_data_test_to_parquet(self, user_data_test: pl.DataFrame) -> None:
        if not os.path.exists(self.PRE_PROCESSED_BASE_PATH):
            os.makedirs(self.PRE_PROCESSED_BASE_PATH)

        user_data_test.write_parquet(self.PREDICTED_NN_USER_DATA_TEST_FILE)
        self.print_file_size(self.PREDICTED_NN_USER_DATA_TEST_FILE)

    @time_it
    def load_predicted_nn_user_data_test_from_parquet(self) -> pl.DataFrame:
        self.print_file_size(self.PREDICTED_NN_USER_DATA_TEST_FILE)
        return pl.read_parquet(self.PREDICTED_NN_USER_DATA_TEST_FILE)

    # +===============================+
    # |            HELPERS            |
    # +===============================+

    def save_pytorch_model(self, model_name: str, model: torch.nn.Module, test_x: np.ndarray) -> None:
        print(f"Saving PyTorch model to path '{self.OUTPUT_PATH}'")

        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)

        file_name = self.__build_file_base_name(model_name) + ".pth"
        tensor = torch.FloatTensor(test_x)
        traced_cell = torch.jit.trace(model, tensor)
        torch.jit.save(traced_cell, file_name)

        print(f"Model saved -> {file_name}")
        self.print_file_size(file_name)

    def save_sklearn_model(self, model_name: str, model: torch.nn.Module) -> None:
        print(f"Saving SkLearn model to path '{self.OUTPUT_PATH}'")

        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)

        file_name = self.__build_file_base_name(model_name) + ".pkl"
        joblib.dump(model, file_name)

        print(f"Model saved -> {file_name}")
        self.print_file_size(file_name)

    @time_it
    def save_polars_df_to_parquet(self, df: pl.DataFrame, name: str) -> None:
        file_name = self.__build_file_base_name(name) + ".parquet"

        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        df.write_parquet(file_name)
        self.print_file_size(file_name)

    def __build_file_base_name(self, file_name: str) -> str:

        filename_with_datetime = f"{file_name}-{self.CURRENT_DATE_TIME}"
        return os.path.join(self.OUTPUT_PATH, filename_with_datetime)

    def print_file_size(self, file_path: str) -> None:
        """
        Prints the size of a file in a human-readable format.

        Parameters
        ----------
        file_path : str
            The path to the file.

        Example
        -------
        >>> repo = DataRepository()
        >>> repo.print_file_size(repo.TRAIN_FILE)
        """
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        if file_size < 1024 * 1024:
            print(f"File '{file_name}' size: {file_size} bytes")
        else:
            print(f"File '{file_name}' size: {file_size / (1024 * 1024):.2f} MB")