import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from app.news_recommendation_1 import time_it
from app.news_recommendation_1.data_repository import DataRepository


class NewsClusterizer:
    """
    A class used to cluster news articles based on their content.

    Attributes
    ----------
    FORCE_REPROCESS : bool
        A flag to force reprocessing of the news data even if classified news data already exists.
    NO_OF_CLUSTERS : int
        The number of clusters to form.
    vectorizer : TfidfVectorizer
        A TF-IDF vectorizer to convert the text data into a matrix of TF-IDF features.
    data_repo : DataRepository
        An instance of the DataRepository class to handle data storage and retrieval.

    Methods
    -------
    __init__(self, force_reprocess: bool, no_of_clusters: int)
        Initializes the NewsClusterizer with the given parameters.
    execute(self, news_data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]
        Executes the clustering process on the provided news data.
    build_cluster(self, news_data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]
        Builds clusters from the provided news data.
    build_cluster_similarity_matrix(self, centers: np.ndarray) -> pl.DataFrame
        Builds a similarity matrix for the cluster centers.

    Usage Examples
    --------------
    >>> clusterizer = NewsClusterizer(force_reprocess=True, no_of_clusters=5)
    >>> news_data = pl.DataFrame({
    ...     'soup_clean': ['text data 1', 'text data 2', 'text data 3'],
    ...     'modified': [1627849200, 1627849201, 1627849202]
    ... })
    >>> clustered_news, similarity_matrix = clusterizer.execute(news_data)
    >>> print(clustered_news)
    >>> print(similarity_matrix)
    """
    FORCE_REPROCESS = False
    NO_OF_CLUSTERS = 10

    def __init__(self, force_reprocess, no_of_clusters):
        """
        Initializes the NewsClusterizer with the given parameters.

        Parameters
        ----------
        force_reprocess : bool
            A flag to force reprocessing of the news data even if classified news data already exists.
        no_of_clusters : int
            The number of clusters to form.
        """
        self.FORCE_REPROCESS = force_reprocess
        self.NO_OF_CLUSTERS = no_of_clusters

        self.vectorizer = TfidfVectorizer(
            max_features=6000,
            max_df=0.9,
            min_df=5)
        self.data_repo = DataRepository()

    @time_it
    def execute(self, news_data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Executes the clustering process on the provided news data.

        If classified news data already exists and FORCE_REPROCESS is False, it loads the classified news data.
        Otherwise, it sorts the news data, builds clusters, and saves the classified news data.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be clustered.

        Returns
        -------
        tuple
            A tuple containing the clustered news data and the similarity matrix.

        Usage Examples
        --------------
        >>> clusterizer = NewsClusterizer(force_reprocess=True, no_of_clusters=5)
        >>> news_data = pl.DataFrame({
        ...     'soup_clean': ['text data 1', 'text data 2', 'text data 3'],
        ...     'modified': [1627849200, 1627849201, 1627849202]
        ... })
        >>> clustered_news, similarity_matrix = clusterizer.execute(news_data)
        >>> print(clustered_news)
        >>> print(similarity_matrix)
        """
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
    def build_cluster(self, news_data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds clusters from the provided news data.

        Uses the TF-IDF vectorizer to convert the text data into a matrix of TF-IDF features,
        and then applies KMeans clustering to form clusters.

        Parameters
        ----------
        news_data : pl.DataFrame
            The news data to be clustered.

        Returns
        -------
        tuple
            A tuple containing the cluster centers and the labels for each news article.

        Usage Examples
        --------------
        >>> clusterizer = NewsClusterizer(force_reprocess=True, no_of_clusters=5)
        >>> news_data = pl.DataFrame({
        ...     'soup_clean': ['text data 1', 'text data 2', 'text data 3'],
        ...     'modified': [1627849200, 1627849201, 1627849202]
        ... })
        >>> centers, labels = clusterizer.build_cluster(news_data)
        >>> print(centers)
        >>> print(labels)
        """
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
    def build_cluster_similarity_matrix(self, centers) -> pl.DataFrame:
        """
        Builds a similarity matrix for the cluster centers.

        Computes the cosine similarity between each pair of cluster centers and creates a similarity matrix.
        Also creates an ordered similarity matrix based on the indices of the sorted similarities.

        Parameters
        ----------
        centers : np.ndarray
            The cluster centers.

        Returns
        -------
        pl.DataFrame
            A DataFrame containing the ordered similarity matrix.

        Usage Examples
        --------------
        >>> clusterizer = NewsClusterizer(force_reprocess=True, no_of_clusters=5)
        >>> centers = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> similarity_matrix = clusterizer.build_cluster_similarity_matrix(centers)
        >>> print(similarity_matrix)
        """
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