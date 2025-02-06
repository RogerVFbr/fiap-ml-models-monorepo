import os
import polars as pl

from app.news_recommendation import time_it


class DataRepository:

    def __init__(self):
        self.DATA_PATH = "challenge-webmedia-e-globo-2023"

        self.TRAIN_DIR = os.path.join(os.getcwd(), self.DATA_PATH, 'files', 'treino')
        self.TEST_DIR = os.path.join(os.getcwd(), self.DATA_PATH, 'files', 'validacao')
        self.NEWS_DIR = os.path.join(os.getcwd(), self.DATA_PATH, 'itens', 'itens')

        self.PRE_PROCESSED_DIR = os.path.join(os.getcwd(), self.DATA_PATH, 'preprocessed')

        self.TRAIN_FILE = os.path.join(self.TRAIN_DIR, 'user_data_train.parquet')
        self.TEST_FILE = os.path.join(self.TEST_DIR, 'user_data_test.parquet')
        self.NEWS_FILE = os.path.join(self.NEWS_DIR, 'item_data.parquet')

        self.NEWS_PRE_PROCESSED_FILE = os.path.join(os.getcwd(), self.PRE_PROCESSED_DIR, 'news_data_preprocessed.parquet')
        self.NEWS_CLASSIFIED_FILE = os.path.join(os.getcwd(), self.PRE_PROCESSED_DIR, 'news_data_classified.parquet')
        self.SIMILARITY_MATRIX_FILE = os.path.join(os.getcwd(), self.PRE_PROCESSED_DIR, 'similarity_matrix.parquet')
        self.FORMATTED_USER_DATA_FILE = os.path.join(os.getcwd(), self.PRE_PROCESSED_DIR, 'user_data_train_formatted.parquet')

    # +===============================+
    # |         FULL DATASETS         |
    # +===============================+

    def load_dataset_from_csv(self):
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
    def save_dataset_to_parquet(self, user_data_train, user_data_test, item_data):
        user_data_train.write_parquet(self.TRAIN_FILE)
        user_data_test.write_parquet(self.TEST_FILE)
        item_data.write_parquet(self.NEWS_FILE)

        self.print_file_size(self.TRAIN_FILE)
        self.print_file_size(self.TEST_FILE)
        self.print_file_size(self.NEWS_FILE)

    @time_it
    def load_dataset_from_parquet(self):
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

    def preprocessed_news_parquet_exists(self):
        if os.path.exists(self.NEWS_PRE_PROCESSED_FILE):
            print(f"Pre processed news parquet exists.")
            return True
        return False

    @time_it
    def save_preprocessed_news_to_parquet(self, news_data):
        if not os.path.exists(self.PRE_PROCESSED_DIR):
            os.makedirs(self.PRE_PROCESSED_DIR)

        news_data.write_parquet(self.NEWS_PRE_PROCESSED_FILE)
        self.print_file_size(self.NEWS_PRE_PROCESSED_FILE)

    @time_it
    def load_preprocessed_news_from_parquet(self):
        self.print_file_size(self.NEWS_PRE_PROCESSED_FILE)
        return pl.read_parquet(self.NEWS_PRE_PROCESSED_FILE)

    # +===============================+
    # |    NEWS CLASSIFIED PARQUET    |
    # +===============================+

    def classified_news_parquet_exists(self):
        if os.path.exists(self.NEWS_CLASSIFIED_FILE) and os.path.exists(self.SIMILARITY_MATRIX_FILE):
            print(f"Classified news parquet exists.")
            print(f"Similarity matrix parquet exists.")
            return True
        return False

    @time_it
    def save_classified_news_to_parquet(self, news_data, similarity_matrix):
        if not os.path.exists(self.PRE_PROCESSED_DIR):
            os.makedirs(self.PRE_PROCESSED_DIR)

        news_data.write_parquet(self.NEWS_CLASSIFIED_FILE)
        similarity_matrix.write_parquet(self.SIMILARITY_MATRIX_FILE)
        self.print_file_size(self.NEWS_CLASSIFIED_FILE)
        self.print_file_size(self.SIMILARITY_MATRIX_FILE)

    @time_it
    def load_classified_news_from_parquet(self):
        news = pl.read_parquet(self.NEWS_CLASSIFIED_FILE)
        similarity_matrix = pl.read_parquet(self.SIMILARITY_MATRIX_FILE)
        self.print_file_size(self.NEWS_CLASSIFIED_FILE)
        self.print_file_size(self.SIMILARITY_MATRIX_FILE)
        return news, similarity_matrix

    # +===============================+
    # |      USER DATA FORMATTED      |
    # +===============================+

    def formatted_user_data_parquet_exists(self):
        if os.path.exists(self.FORMATTED_USER_DATA_FILE):
            print(f"Formatted user data parquet exists.")
            return True
        return False

    @time_it
    def save_formatted_user_data_to_parquet(self, user_data_train):
        if not os.path.exists(self.PRE_PROCESSED_DIR):
            os.makedirs(self.PRE_PROCESSED_DIR)

        user_data_train.write_parquet(self.FORMATTED_USER_DATA_FILE)
        self.print_file_size(self.FORMATTED_USER_DATA_FILE)

    @time_it
    def load_formatted_user_data_from_parquet(self):
        self.print_file_size(self.FORMATTED_USER_DATA_FILE)
        return pl.read_parquet(self.FORMATTED_USER_DATA_FILE)

    # +===============================+
    # |            HELPERS            |
    # +===============================+

    def print_file_size(self, file_path):
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        if file_size < 1024 * 1024:
            print(f"File '{file_name}' size: {file_size} bytes")
        else:
            print(f"File '{file_name}' size: {file_size / (1024 * 1024):.2f} MB")
