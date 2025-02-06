import os
import re
import string
import nltk
import spacy
import unidecode
from nltk.corpus import stopwords
import polars as pl
from spacy.tokens import Doc

from app.news_recommendation import time_it
from app.news_recommendation.data_repository import DataRepository


class NewsProcessor:

    FORCE_REPROCESS = False

    def __init__(self, force_reprocess):
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
    def execute(self, news_data: pl.DataFrame):
        if self.data_repo.preprocessed_news_parquet_exists() and not self.FORCE_REPROCESS:
            return self.data_repo.load_preprocessed_news_from_parquet()

        news_data = news_data.sort('modified', descending=True)
        # news_data = news_data.select(pl.all().gather(range(1)))


        for step in self.steps:
            news_data = step(news_data)
            self.print_soup(news_data)

        self.data_repo.save_preprocessed_news_to_parquet(news_data)

        return news_data

    @time_it
    def prepare_soup_and_tokenize(self, news_data: pl.DataFrame):
        news_data = news_data.with_columns([
            pl.concat_str([
                pl.col('title'),
                pl.col('caption'),
                # pl.col('body')
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
    def remove_punctuation_and_lowercase(self, news_data: pl.DataFrame):
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
    def remove_stopwords(self, news_data: pl.DataFrame):
        stop_words = set(stopwords.words('portuguese'))

        def remove(text: pl.Series):
            words = text.to_list()
            return [word for word in words if word not in stop_words and len(word)>1]

        return news_data.with_columns([
            pl.col('soup_clean').map_elements(lambda x: remove(x), return_dtype=pl.List(pl.Utf8))
        ])

    @time_it
    def remove_if_contains_numbers(self, news_data: pl.DataFrame):
        def remove(text: pl.Series):
            words = text.to_list()
            return [word for word in words if not any(char.isdigit() for char in word)]

        return news_data.with_columns([
            pl.col('soup_clean').map_elements(lambda x: remove(x), return_dtype=pl.List(pl.Utf8))
        ])

    @time_it
    def remove_non_nouns(self, news_data: pl.DataFrame):
        classes = self.prepare_classes()

        def remove(text):
            words = text.to_list()
            return [word for word in words if word not in classes]

        return news_data.with_columns([
            pl.col('soup_clean').map_elements(lambda x: remove(x), return_dtype=pl.List(pl.Utf8))
        ])

    @time_it
    def prepare_classes(self):
        mac_morpho_tagged = nltk.corpus.mac_morpho.tagged_words()
        return {key.lower(): value for key, value in mac_morpho_tagged if value not in ['N', 'NPROP']}

    @time_it
    def lemmatize(self, news_data: pl.DataFrame):
        nlp = spacy.load('pt_core_news_sm', exclude=[
            "ner",
            "parser",
            "tagger",
            # "tok2vec",
            "attribute_ruler",
            "morphologizer",
            "senter",
            "transformer",
        ])


        soup = [Doc(nlp.vocab, words=x) for x in news_data['soup_clean'].to_list()]

        total_rows = news_data.shape[0]
        counter = 0
        lemma_text_list = []
        # docs = list(nlp.pipe(texts, n_process=2, batch_size=2000)) # default batch_size = 1000
        docs = list(nlp.pipe(soup, n_process=6))

        # 2000  - 1: 31,   2:   22, 4:   17
        # 4000  - 1: 52,   2:   33, 4:   24, 6: 26
        # 8000  - 1: 1:33, 2:   56, 4:   38, 6: 40
        # 16000 - 1: 2:53, 2: 1:42, 4: 1:07, 6: 1:04

        for doc in docs:
            lemma_text_list.append([token.lemma_.lower() for token in doc])
            if counter % 10000 == 0:
                print(f'Rows lemmatized: {counter} / {total_rows}')
            counter += 1

        return news_data.with_columns(pl.Series(name="soup_clean", values=lemma_text_list))

    @time_it
    def remove_accents(self, news_data: pl.DataFrame):
        def remove(text):
            words = text.to_list()
            return [unidecode.unidecode(word) for word in words]

        news_data = news_data.with_columns([
            pl.col('soup_clean').map_elements(lambda x: remove(x), return_dtype=pl.List(pl.Utf8))
        ])

        return news_data

    @time_it
    def join_soup_lists(self, news_data: pl.DataFrame):
        news_data = news_data.with_columns([
            pl.col('soup_clean').list.join(' ')
        ])

        return news_data

    def print_soup(self, news_data: pl.DataFrame):
        if 'soup_clean' in news_data.columns:
            data = news_data.select(['soup', 'soup_clean'])
            print(data)
        else:
            data = news_data.select(['soup'])
            print(data)