import re
import string
import nltk
import spacy
import unidecode
from nltk.corpus import stopwords
import polars as pl
from spacy.tokens import Doc

from app.news_recommendation_1 import time_it
from app.news_recommendation_1.data_repository import DataRepository


class NewsProcessor:
    """
    A class to process news data for the news recommendation system.

    Attributes
    ----------
    FORCE_REPROCESS : bool
        A flag to force reprocessing of the news data.
    data_repo : DataRepository
        An instance of DataRepository to manage data loading and saving.
    steps : list
        A list of processing steps to be applied to the news data.
    """

    FORCE_REPROCESS = False

    def __init__(self, force_reprocess: bool):
        """
        Initializes the NewsProcessor with a flag to force reprocessing.

        Parameters
        ----------
        force_reprocess : bool
            A flag to force reprocessing of the news data.
        """
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('mac_morpho')
        nltk.download('averaged_perceptron_tagger_eng')

        self.FORCE_REPROCESS = force_reprocess

        self.data_repo = DataRepository()

        self.steps = [
            self.prepare_soup_and_tokenize,
            self.remove_punctuation_and_lowercase,
            self.remove_stopwords,
            self.remove_if_contains_numbers,
            self.lemmatize,
            # self.remove_non_nouns,
            self.remove_accents,
            self.join_soup_lists
        ]

    @time_it
    def execute(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Executes the processing steps on the news data.

        This method checks if preprocessed news data already exists and loads it if available and reprocessing is not forced.
        Otherwise, it sorts the news data by the 'modified' column in descending order and applies a series of processing steps.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be processed.

        Returns
        -------
        pl.DataFrame
            The processed news data.

        Example
        -------
        >>> processor = NewsProcessor(force_reprocess=True)
        >>> processed_data = processor.execute(news_data)
        """
        if self.data_repo.preprocessed_news_parquet_exists() and not self.FORCE_REPROCESS:
            data = self.data_repo.load_preprocessed_news_from_parquet()
            print(data.head(5))
            return data

        news_data = news_data.sort('modified', descending=True)

        for step in self.steps:
            news_data = step(news_data)
            self.print_soup(news_data)

        self.data_repo.save_preprocessed_news_to_parquet(news_data)
        return news_data

    @time_it
    def prepare_soup_and_tokenize(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Prepares the soup column by concatenating title and caption, and tokenizes it.

        This method processes the news data by:
        1. Concatenating the 'title' and 'caption' columns into a new 'soup' column.
        2. Removing extra whitespace and splitting the 'soup' column into a list of words.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be processed.

        Returns
        -------
        pl.DataFrame
            The news data with the prepared soup column.
        """
        news_data = news_data.with_columns([
            pl.concat_str([
                pl.col('title'),
                pl.col('caption')
            ], separator=' ').alias('soup')
        ])

        news_data = news_data.with_columns([
            pl
            .col('soup')
            .map_elements(lambda x: re.sub(r'\s+', ' ', x).strip(), return_dtype=pl.Utf8)
            .str
            .split(' ')
            .alias('soup_clean')
        ])

        news_data.drop_in_place('url')
        news_data.drop_in_place('title')
        news_data.drop_in_place('caption')
        news_data.drop_in_place('body')

        return news_data

    @time_it
    def remove_punctuation_and_lowercase(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Removes punctuation and converts text to lowercase.

        This method processes the 'soup_clean' column of the news data by:
        1. Removing all punctuation characters.
        2. Converting all text to lowercase.
        3. Stripping leading and trailing whitespace from each word.
        4. Filtering out words that are digits.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be processed.

        Returns
        -------
        pl.DataFrame
            The news data with punctuation removed and text in lowercase.
        """
        punctuation_table = str.maketrans('', '', string.punctuation)

        def remove(text: pl.Series):
            words = text.to_list()
            words = [word.translate(punctuation_table) for word in words]
            words = [word.lower().strip() for word in words if not word.isdigit()]
            return words

        news_data = news_data.with_columns([
            pl.col('soup_clean').map_elements(lambda x: remove(x), return_dtype=pl.List(pl.Utf8))
        ])

        return news_data

    @time_it
    def remove_stopwords(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Removes stopwords from the text.

        This method processes the 'soup_clean' column of the news data by:
        1. Removing common stopwords in Portuguese.
        2. Filtering out words with a length of 1 or less.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be processed.

        Returns
        -------
        pl.DataFrame
            The news data with stopwords removed.
        """
        stop_words = set(stopwords.words('portuguese'))

        def remove(text: pl.Series):
            words = text.to_list()
            return [word for word in words if word not in stop_words and len(word) > 1]

        return news_data.with_columns([
            pl.col('soup_clean').map_elements(lambda x: remove(x), return_dtype=pl.List(pl.Utf8))
        ])

    @time_it
    def remove_if_contains_numbers(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Removes words that contain numbers.

        This method processes the 'soup_clean' column of the news data by:
        1. Filtering out words that contain any numeric characters.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be processed.

        Returns
        -------
        pl.DataFrame
            The news data with words containing numbers removed.
        """
        def remove(text: pl.Series):
            words = text.to_list()
            return [word for word in words if not any(char.isdigit() for char in word)]

        return news_data.with_columns([
            pl.col('soup_clean').map_elements(lambda x: remove(x), return_dtype=pl.List(pl.Utf8))
        ])

    @time_it
    def remove_non_nouns(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Removes non-noun words from the text.

        This method processes the 'soup_clean' column of the news data by:
        1. Filtering out words that are not classified as nouns.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be processed.

        Returns
        -------
        pl.DataFrame
            The news data with non-noun words removed.
        """
        classes = self.prepare_classes()

        def remove(text: pl.Series):
            words = text.to_list()
            return [word for word in words if word not in classes]

        return news_data.with_columns([
            pl.col('soup_clean').map_elements(lambda x: remove(x), return_dtype=pl.List(pl.Utf8))
        ])

    @time_it
    def prepare_classes(self) -> dict:
        """
        Prepares a set of non-noun classes.

        This method retrieves tagged words from the mac_morpho corpus and filters out non-noun classes.

        Returns
        -------
        set
            A set of non-noun classes.
        """
        mac_morpho_tagged = nltk.corpus.mac_morpho.tagged_words()
        return {key.lower(): value for key, value in mac_morpho_tagged if value not in ['N', 'NPROP']}

    @time_it
    def lemmatize(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Lemmatizes the text in the soup column.

        This method processes the 'soup_clean' column of the news data by:
        1. Loading the spaCy model for Portuguese.
        2. Lemmatizing each word in the 'soup_clean' column.
        3. Converting each word to lowercase.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be processed.

        Returns
        -------
        pl.DataFrame
            The news data with lemmatized text.
        """
        nlp = spacy.load('pt_core_news_sm', exclude=[
            "ner",
            "parser",
            "tagger",
            "attribute_ruler",
            "morphologizer",
            "senter",
            "transformer",
        ])

        soup = [Doc(nlp.vocab, words=x) for x in news_data['soup_clean'].to_list()]

        total_rows = news_data.shape[0]
        counter = 0
        lemma_text_list = []
        docs = list(nlp.pipe(soup, n_process=6))

        for doc in docs:
            lemma_text_list.append([token.lemma_.lower() for token in doc])
            if counter % 10000 == 0:
                print(f'Rows lemmatized: {counter} / {total_rows}')
            counter += 1

        return news_data.with_columns(pl.Series(name="soup_clean", values=lemma_text_list))

    @time_it
    def remove_accents(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Removes accents from the text.

        This method processes the 'soup_clean' column of the news data by:
        1. Removing accents from each word.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be processed.

        Returns
        -------
        pl.DataFrame
            The news data with accents removed.
        """
        def remove(text: pl.Series):
            words = text.to_list()
            return [unidecode.unidecode(word) for word in words]

        news_data = news_data.with_columns([
            pl.col('soup_clean').map_elements(lambda x: remove(x), return_dtype=pl.List(pl.Utf8))
        ])

        return news_data

    @time_it
    def join_soup_lists(self, news_data: pl.DataFrame) -> pl.DataFrame:
        """
        Joins the list of words in the soup column into a single string.

        This method processes the 'soup_clean' column of the news data by:
        1. Joining the list of words into a single string.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be processed.

        Returns
        -------
        pl.DataFrame
            The news data with the soup column joined into a single string.
        """
        news_data = news_data.with_columns([
            pl.col('soup_clean').list.join(' ')
        ])

        return news_data

    def print_soup(self, news_data: pl.DataFrame) -> None:
        """
        Prints the soup and soup_clean columns for debugging purposes.

        This method prints the 'soup' and 'soup_clean' columns of the news data for debugging.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be printed.
        """
        if 'soup_clean' in news_data.columns:
            data = news_data.select(['soup', 'soup_clean'])
            print(data)
        else:
            data = news_data.select(['soup'])
            print(data)