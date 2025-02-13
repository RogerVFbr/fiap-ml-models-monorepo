import polars as pl

from app.news_recommendation_2 import time_it
from app.news_recommendation_2.data_service import DataService
from app.news_recommendation_2.dataframe_configs import DataframeConfigs

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score



class Test:
    OUTPUT_PATH = None

    NO_OF_CLUSTERS = 10

    def __init__(self):
        self.df_configs = DataframeConfigs()
        self.data_service = DataService()


    @time_it
    def run(self):
        user_data_train, user_data_test, news_data = self.data_service.onboard()

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
                pl.col("timestamp") >= pl.col("issued").dt.replace_time_zone(None)
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
        print(analytics.head())

        weights = {
            'count': 1,
            'engagement': 1,
            'timeOnPage_per_char': 1,
            'scrollPercentage': 1,
        }

        def test(filter_timestamp):
            top_results = (
                analytics.lazy()
                .explode(['timestamp', 'engagement', 'timeOnPage_per_char', 'scrollPercentage'])
                .filter(
                    (pl.col("timestamp") <= filter_timestamp) &
                    (pl.col("timestamp") >= filter_timestamp - pl.duration(days=7))
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
                            pl.col("count_scaled")*weights['count'] +
                            pl.col("engagement_scaled")*weights['engagement'] +
                            pl.col("timeOnPage_per_char_scaled")*weights['timeOnPage_per_char'] +
                            pl.col("scrollPercentage_scaled")*weights['scrollPercentage']
                     ).alias("selection_index")
                )
                .sort(["selection_index"], descending=True)
                .head(5)
            )

            return top_results

            # return top_results.collect().get_column("page").to_list()

        # user_data_test = user_data_test.select(pl.all().gather(range(20000)))
        # print(user_data_test)

        filter_timestamp = pl.datetime(2022, 7, 15, 3, 12, 16, 590000)
        a = test(filter_timestamp)
        print(type(a))
        # print(a.collect().get_column("page"))
        # print(a)
        print(a.collect())

        # print(test(filter_timestamp))

        # t = (
        #     user_data_test.lazy()
        #     .explode(['history', 'timestampHistory'])
        #     .select(
        #         pl.col("timestampHistory").alias("timestamp"),
        #         pl.col("history").alias("target"),
        #     )
        #     .with_columns(
        #         pl.col("timestamp").map_elements(test, return_dtype=pl.List(pl.Utf8)).alias("predictions"),
        #     )
        #     .sort(["timestamp"], descending=False)
        # ).collect()
        #
        # print("PREDICTED")
        # # print(t.head(5))
        # print(t)
        #
        # target = t['target'].to_list()
        # predictions = [x[0] for x in t['predictions'].to_list()]
        #
        # print(len(target))
        # print(len(predictions))
        #
        # accuracy = accuracy_score(target, predictions)
        # print(accuracy*100)


if __name__ == "__main__":
    train = Test()
    train.run()