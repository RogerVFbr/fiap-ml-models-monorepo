from app.news_recommendation_1 import time_it
from app.news_recommendation_1.data_service import DataService
from app.news_recommendation_1.dataframe_configs import DataframeConfigs
from app.news_recommendation_1.predictors.news_cluster_predictor_classic import NewsClusterClassicPredictor
from app.news_recommendation_1.predictors.news_cluster_predictor_nn import NewsClusterNNPredictor
from app.news_recommendation_1.predictors.news_page_predictor import NewsPagePredictor
from app.news_recommendation_1.news_processor import NewsProcessor
from app.news_recommendation_1.news_clusterizer import NewsClusterizer
from app.news_recommendation_1.user_feature_engineering import UserFeatureEngineering


class NewsRecommenderSystem:
    FORCE_REPROCESS_NEWS_PROCESSOR = False
    FORCE_REPROCESS_NEWS_CLUSTERIZER = False
    FORCE_REPROCESS_USER_FEAT_ENGINEERING = False
    FORCE_REPROCESS_CLUSTER_NN_PREDICTOR = False
    FORCE_REPROCESS_CLUSTER_CLASSIC_PREDICTOR = False

    OUTPUT_PATH = None

    NO_OF_CLUSTERS = 10

    def __init__(self):
        self.df_configs = DataframeConfigs()
        self.data_service = DataService()
        self.news_processor = NewsProcessor(self.FORCE_REPROCESS_NEWS_PROCESSOR)
        self.news_clusterizer = NewsClusterizer(self.FORCE_REPROCESS_NEWS_CLUSTERIZER, self.NO_OF_CLUSTERS)
        self.user_feature_engineering = UserFeatureEngineering(self.FORCE_REPROCESS_USER_FEAT_ENGINEERING, self.NO_OF_CLUSTERS)
        self.news_cluster_nn_predictor = NewsClusterNNPredictor(self.FORCE_REPROCESS_CLUSTER_NN_PREDICTOR)
        self.news_cluster_classic_predictor = NewsClusterClassicPredictor(self.FORCE_REPROCESS_CLUSTER_CLASSIC_PREDICTOR)
        self.news_page_predictor = NewsPagePredictor()

    @time_it
    def run(self):
        user_data_train, user_data_test, news_data = self.data_service.onboard()
        news_data = self.news_processor.execute(news_data)
        news_data, similarity_matrix = self.news_clusterizer.execute(news_data)
        user_data_train, user_data_test = self.user_feature_engineering.execute(user_data_train, user_data_test, news_data)
        user_data_test = self.news_cluster_nn_predictor.execute(user_data_train, user_data_test, similarity_matrix)
        user_data_test = self.news_cluster_classic_predictor.execute(user_data_train, user_data_test, similarity_matrix)
        self.news_page_predictor.predict(user_data_test, news_data, similarity_matrix)

        # self.explore(user_data_train, user_data_test, news_data)

    @time_it
    def explore(self, user_data_train, user_data_test, news_data):
        # print()
        # print('+=========================+')
        # print('|          TRAIN          |')
        # print('+=========================+')
        # # print(user_data_train.describe())
        # # print(f'Number of unique user ids: {user_data_test["userId"].n_unique()} (Total rows: {user_data_test.shape[0]})')
        # # print(f'historySize Max/Min : {user_data_train["historySize"].min()} / {user_data_train["historySize"].max()}')
        # print()
        # print("HEAD")
        # print(user_data_train.head(5))
        # # print()
        # # print("TAIL")
        # # print(user_data_train.tail(5))

        print()
        print('+=========================+')
        print('|           TEST          |')
        print('+=========================+')
        print(user_data_test.describe())
        print(f'Number of unique user ids: {user_data_test["userId"].n_unique()} (Total rows: {user_data_test.shape[0]})')
        print()
        print("HEAD")
        print(user_data_test.head(5))
        print()
        print("TAIL")
        print(user_data_test.tail(5))

        # print()
        # print('+=========================+')
        # print('|        NEWS DATA        |')
        # print('+=========================+')
        # # print(news_data.describe())
        # # print()
        # print("HEAD")
        # print(news_data.head(1))
        # print()
        # print("TAIL")
        # print(news_data.tail(1))
        # print()


if __name__ == "__main__":
    train = NewsRecommenderSystem()
    train.run()