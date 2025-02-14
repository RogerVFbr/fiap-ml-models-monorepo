import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity

from app.news_recommendation_1 import time_it
from app.news_recommendation_1.data_repository import DataRepository


class NewsClusterizer:

    FORCE_REPROCESS = False
    NO_OF_CLUSTERS = 10

    def __init__(self, force_reprocess, no_of_clusters):
        self.FORCE_REPROCESS = force_reprocess
        self.NO_OF_CLUSTERS = no_of_clusters

        self.vectorizer = TfidfVectorizer(
            max_features=6000,
            max_df=0.9,
            min_df=5)
        self.data_repo = DataRepository()

    @time_it
    def execute(self, news_data: pl.DataFrame):
        if self.data_repo.classified_news_parquet_exists() and not self.FORCE_REPROCESS:
            news_data, similarity_matrix = self.data_repo.load_classified_news_from_parquet()
            print(news_data.head(5))
            print(similarity_matrix)
            return news_data, similarity_matrix

        news_data = news_data.sort('modified', descending=True)
        centers, labels = self.build_cluster(news_data)
        similarity_matrix = self.build_cluster_similarity_matrix(centers)
        news_data = news_data.with_columns(pl.Series('cluster', labels))

        print()
        print("Clusterized News")
        print(news_data.head(5))

        self.data_repo.save_classified_news_to_parquet(news_data, similarity_matrix)

        return news_data, similarity_matrix

    @time_it
    def build_cluster(self, news_data: pl.DataFrame):
        matrix = self.vectorizer.fit_transform(news_data['soup_clean'])

        clusters = KMeans(
            n_clusters=self.NO_OF_CLUSTERS,
            max_iter=100,
            n_init=5,
            random_state=42
        )

        clusters.fit(matrix)

        return clusters.cluster_centers_, clusters.labels_

    @time_it
    def build_cluster_similarity_matrix(self, centers):
        data = []

        for i in range(centers.shape[0]):
            cluster = []
            for z in range(centers.shape[0]):
                A = centers[i]
                B = centers[z]
                cos_similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
                cos_similarity = cos_similarity.item()
                cluster.append(cos_similarity)
            data.append(cluster)

        schema = [str(x) for x in range(centers.shape[0])]
        df = pl.DataFrame(data, schema=schema)

        print()
        print("Similarity Matrix (Original)")
        print(df)

        ordered = []

        for i in range(len(data)):
            indexed_values = list(enumerate(data[i]))
            sorted_indexed_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)
            ordered.append([index for index, value in sorted_indexed_values])

        ordered = list(map(list, zip(*ordered)))
        schema = [str(x) for x in range(len(ordered))]
        df = pl.DataFrame(ordered, schema=schema)

        print()
        print("Similarity Matrix (Ordered By Index)")
        print(df)

        return df