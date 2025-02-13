import polars as pl

from app.news_recommendation_2 import time_it
from app.news_recommendation_2.data_service import DataService
from app.news_recommendation_2.dataframe_configs import DataframeConfigs

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

import warnings
from sklearn.exceptions import UndefinedMetricWarning


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

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


    @time_it
    def run(self):
        user_data_train, user_data_test, news_data = self.data_service.onboard()
        cutoff = pl.datetime(2022, 8, 15, 3, 00, 1, 65700)
        window_in_hours = 12

        analytics = (
            user_data_train.lazy()
            .select(
                pl.col("history"),
                pl.col("timestampHistory").alias("timestamp"),
                pl.col("numberOfClicksHistory").alias("engagement"),
                pl.col("timeOnPageHistory").alias("timeOnPage"),
                pl.col("scrollPercentageHistory").alias("scrollPercentage")
            )
            .explode(['history', 'timestamp', 'engagement', 'timeOnPage', 'scrollPercentage'])
            .sort(["history", "timestamp"], descending=False)
            .join((
                news_data.lazy()
                .select([
                    pl.col("page"),
                    pl.col("issued"),
                    pl.col("body").str.len_chars().alias("body_length")
                ])
            ), left_on='history', right_on='page', how='left')
            .filter(
                pl.col("timestamp") >= pl.col("issued").dt.replace_time_zone(None),
                (pl.col("timestamp") >= cutoff - pl.duration(hours=window_in_hours))
            )
            .select(
                pl.col("history").alias("page"),
                pl.col("issued"),
                pl.col("timestamp"),
                pl.col("engagement"),
                (pl.col("timeOnPage") / pl.col("body_length")).alias("timeOnPage_per_char"),
                pl.col("scrollPercentage")
            )
            .group_by('page')
            .agg(
                pl.col("issued").first(),
                pl.len().alias('count'),
                pl.col("timestamp"),
                pl.col("engagement"),
                pl.col("timeOnPage_per_char"),
                pl.col("scrollPercentage"),
            )
            .sort(["issued"], descending=False)
        ).collect()

        print("ANALYTICS")
        print(analytics.head(5))

        weights = {
            'views': 0,
            'engagement': 0,
            'timeOnPage_per_char': 0,
            'scrollPercentage': 1,
        }

        def test(filter_timestamp):
            top_results = (
                analytics.lazy()
                .explode(['timestamp', 'engagement', 'timeOnPage_per_char', 'scrollPercentage'])
                .filter(
                    (pl.col("timestamp") <= filter_timestamp) &
                    (pl.col("timestamp") >= filter_timestamp - pl.duration(hours=window_in_hours))
                )
                .group_by('page')
                .agg(
                    pl.col("issued").first(),
                    pl.len().alias('count'),
                    pl.col("engagement").mean(),
                    pl.col("timeOnPage_per_char").mean(),
                    pl.col("scrollPercentage").mean(),
                )
                .with_columns(
                    ((pl.col("count") - pl.col("count").min()) / (pl.col("count").max() - pl.col("count").min())).alias("count_scaled"),
                    ((pl.col("engagement") - pl.col("engagement").min()) / (pl.col("engagement").max() - pl.col("engagement").min())).alias("engagement_scaled"),
                    ((pl.col("timeOnPage_per_char") - pl.col("timeOnPage_per_char").min()) / (pl.col("timeOnPage_per_char").max() - pl.col("timeOnPage_per_char").min())).alias("timeOnPage_per_char_scaled"),
                    ((pl.col("scrollPercentage") - pl.col("scrollPercentage").min()) / (pl.col("scrollPercentage").max() - pl.col("scrollPercentage").min())).alias("scrollPercentage_scaled"),
                )
                .select(
                    pl.col("page"),
                    pl.col("issued"),
                    (
                            pl.col("count_scaled")*weights['views'] +
                            pl.col("engagement_scaled")*weights['engagement'] +
                            pl.col("timeOnPage_per_char_scaled")*weights['timeOnPage_per_char'] +
                            pl.col("scrollPercentage_scaled")*weights['scrollPercentage']
                     ).alias("selection_index")
                )
                .sort("selection_index", descending=True)
                .head(10)
            ).collect()

            return top_results.get_column("page").to_list()

        print(f"USER DATA TEST (Total rows: {user_data_test.shape[0]})")
        # user_data_test = user_data_test.select(pl.all().gather(range(10)))
        print(user_data_test.head())

        t = (
            user_data_test.lazy()
            .explode(['history', 'timestampHistory'])
            .select(
                pl.col("timestampHistory").alias("timestamp"),
                pl.col("history").alias("target"),
            )
            .join((
                news_data.lazy()
                .select([
                    pl.col("issued"),
                    pl.col("page")
                ])
            ), left_on='target', right_on='page', how='left')
            .with_columns(
                pl.col("timestamp").map_elements(test, return_dtype=pl.List(pl.Utf8)).alias("predictions"),
            )
            .select(
                pl.col("timestamp"),
                pl.col("issued"),
                pl.col("target"),
                pl.col("predictions")
            )
            .sort(["timestamp"], descending=False)
        ).collect()

        print("PREDICTED")
        print(t.head(5))
        # print(t)

        metrics = {
            "Criteria": [],
            "Match Count": [],
            "Accuracy": [],
            "F1 Score": [],
            "Precision": [],
            "Recall": [],
        }

        target = t['target'].to_list()
        preds = t['predictions'].to_list()

        for i in range(10):
            predictions = [target if target in pred[:min(i+1, len(pred))] else '' for pred, target in zip(preds, target)]
            match_count = sum([1 for pred, target in zip(predictions, target) if pred == target])

            metrics['Criteria'].append(f"Hypothesis {i+1}")
            metrics['Match Count'].append(match_count)
            metrics['Accuracy'].append(f"{accuracy_score(target, predictions) * 100:.2f} %")
            metrics['F1 Score'].append(f"{f1_score(target, predictions, average='weighted') * 100:.2f} %")
            metrics['Precision'].append(f"{precision_score(target, predictions, average='weighted') * 100:.2f} %")
            metrics['Recall'].append(f"{recall_score(target, predictions, average='weighted') * 100:.2f} %")

        w = ' '.join([f"{k}: {v}" for k, v in weights.items()])
        print(f"PREDICTION RESULTS (Total samples: {len(t['target'].to_list())}, window_in_hours: {window_in_hours})")
        print(f"Weights -> {w}")
        print(pl.DataFrame(metrics))




if __name__ == "__main__":
    train = NewsRecommenderSystem()
    train.run()