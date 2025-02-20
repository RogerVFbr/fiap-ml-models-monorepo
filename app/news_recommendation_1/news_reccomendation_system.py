from app.news_recommendation_1 import time_it
from app.news_recommendation_1.data_deliverables_builder import DataDeliverablesBuilder
from app.news_recommendation_1.data_service import DataService
from app.news_recommendation_1.dataframe_configs import DataframeConfigs
from app.news_recommendation_1.predictors.news_cluster_predictor_classic import NewsClusterClassicPredictor
from app.news_recommendation_1.predictors.news_cluster_predictor_nn import NewsClusterNNPredictor
from app.news_recommendation_1.predictors.news_page_predictor import NewsPagePredictor
from app.news_recommendation_1.news_processor import NewsProcessor
from app.news_recommendation_1.news_clusterizer import NewsClusterizer
from app.news_recommendation_1.user_feature_engineering import UserFeatureEngineering


class NewsRecommenderSystem:
    """
    A class to represent the news recommender system.

    Attributes
    ----------
    FORCE_REPROCESS_NEWS_PROCESSOR : bool
        A flag to force reprocessing of the news data.
    FORCE_REPROCESS_NEWS_CLUSTERIZER : bool
        A flag to force reprocessing of the news clusters.
    FORCE_REPROCESS_USER_FEAT_ENGINEERING : bool
        A flag to force reprocessing of user feature engineering.
    FORCE_REPROCESS_CLUSTER_NN_PREDICTOR : bool
        A flag to force reprocessing of the neural network cluster predictor.
    FORCE_REPROCESS_CLUSTER_CLASSIC_PREDICTOR : bool
        A flag to force reprocessing of the classic cluster predictor.
    OUTPUT_PATH : str
        The output path for saving results.
    NO_OF_CLUSTERS : int
        The number of clusters to form.

    Methods
    -------
    __init__(self)
        Initializes the NewsRecommenderSystem with default parameters.
    run(self) -> None
        Executes the entire news recommendation pipeline.

    Usage Examples
    --------------
    >>> recommender = NewsRecommenderSystem()
    >>> recommender.run()
    """
    FORCE_REPROCESS_NEWS_PROCESSOR = False
    FORCE_REPROCESS_NEWS_CLUSTERIZER = False
    FORCE_REPROCESS_USER_FEAT_ENGINEERING = False
    FORCE_REPROCESS_CLUSTER_NN_PREDICTOR = False
    FORCE_REPROCESS_CLUSTER_CLASSIC_PREDICTOR = False

    OUTPUT_PATH = None

    NO_OF_CLUSTERS = 10

    def __init__(self):
        """
        Initializes the NewsRecommenderSystem with default parameters.

        This method sets up the necessary components for the news recommendation system, including data service, news processor, news clusterizer, user feature engineering, and predictors.
        """
        self.df_configs = DataframeConfigs()
        self.data_service = DataService()
        self.news_processor = NewsProcessor(self.FORCE_REPROCESS_NEWS_PROCESSOR)
        self.news_clusterizer = NewsClusterizer(self.FORCE_REPROCESS_NEWS_CLUSTERIZER, self.NO_OF_CLUSTERS)
        self.user_feature_engineering = UserFeatureEngineering(self.FORCE_REPROCESS_USER_FEAT_ENGINEERING, self.NO_OF_CLUSTERS)
        self.news_cluster_nn_predictor = NewsClusterNNPredictor(self.FORCE_REPROCESS_CLUSTER_NN_PREDICTOR, self.OUTPUT_PATH)
        self.news_cluster_classic_predictor = NewsClusterClassicPredictor(self.FORCE_REPROCESS_CLUSTER_CLASSIC_PREDICTOR, self.OUTPUT_PATH)
        # self.news_page_predictor = NewsPagePredictor()
        self.data_delivery = DataDeliverablesBuilder(self.OUTPUT_PATH)

    @time_it
    def run(self):
        """
        Executes the entire news recommendation pipeline.

        This method orchestrates the data onboarding, processing, clustering, feature engineering, prediction, and data delivery steps.

        Returns
        -------
        None

        Example
        -------
        >>> recommender = NewsRecommenderSystem()
        >>> recommender.run()
        """
        user_data_train, user_data_test, news_data = self.data_service.onboard()
        news_data = self.news_processor.execute(news_data)
        news_data, similarity_matrix = self.news_clusterizer.execute(news_data)
        user_data_train, user_data_test = self.user_feature_engineering.execute(user_data_train, user_data_test, news_data)
        user_data_test = self.news_cluster_nn_predictor.execute(user_data_train, user_data_test, similarity_matrix)
        user_data_test = self.news_cluster_classic_predictor.execute(user_data_train, user_data_test, similarity_matrix)
        # self.news_page_predictor.predict(user_data_test, news_data, similarity_matrix)
        self.data_delivery.execute(user_data_test, user_data_train, news_data, similarity_matrix, UserFeatureEngineering.FEATURE_WEIGHTS)
