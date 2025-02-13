import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


class NewsPagePredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=6000,
            max_df=0.9,
            min_df=5)

    def predict(self, user_data_test: pl.DataFrame, news_data: pl.DataFrame, similarity_matrix: pl.DataFrame):
        # user_data_test = user_data_test.select(pl.all().gather(range(500)))

        similarity_matrix_list = similarity_matrix.to_numpy().tolist()

        similar_clusters = 10
        selected_news = 10

        def get_similar_clusters(cluster):
            return similarity_matrix_list[cluster][:similar_clusters+1]

        def get_news_by_clusters(row):
            _, clusters_column_name = row
            timestamp = row['target_timestamp']
            search_clusters = row[clusters_column_name]

            cluster_news = news_data.filter(
                (pl.col("cluster").is_in(search_clusters)) &
                (pl.col("issued").dt.replace_time_zone(None) < timestamp) #&
                # (pl.col("issued").dt.replace_time_zone(None) >= (timestamp - timedelta(days=7)))
            ).sort("issued", descending=True)

            return cluster_news.get_column("page")[:selected_news]

        result = (
            user_data_test.lazy()
            .select(
                pl.col("target_page"),
                pl.col("target_timestamp"),
                pl.col("predicted_cluster_nn"),
                pl.col("predicted_cluster_classic"),
            )
            .select(
                pl.col("target_page"),
                pl.col("target_timestamp"),
                pl.col("predicted_cluster_nn").map_elements(get_similar_clusters, return_dtype=pl.List(pl.Int32)).alias("predicted_clusters_nn"),
                pl.col("predicted_cluster_classic").map_elements(get_similar_clusters, return_dtype=pl.List(pl.Int32)).alias("predicted_clusters_classic"),
            )
            .select(
                pl.col("target_page"),
                pl.col("target_timestamp"),
                pl.col("predicted_clusters_nn"),
                pl.struct(pl.col("target_timestamp"), pl.col("predicted_clusters_nn")).map_elements(get_news_by_clusters, return_dtype=pl.List(pl.Utf8)).alias("predictions_nn"),
                pl.col("predicted_clusters_classic"),
                pl.struct(pl.col("target_timestamp"), pl.col("predicted_clusters_classic")).map_elements(get_news_by_clusters, return_dtype=pl.List(pl.Utf8)).alias("predictions_classic"),
            )
            .with_columns(
                pl.when(pl.col("target_page").is_in(pl.col("predictions_nn"))).then(pl.col("target_page")).otherwise(pl.lit('Miss')).alias("hit_nn"),
                pl.when(pl.col("target_page").is_in(pl.col("predictions_classic"))).then(pl.col("target_page")).otherwise(pl.lit('Miss')).alias("hit_classic"),
            )
        ).collect()

        print(result)

        metrics = {
            "Criteria": [],
            "Match Count NN": [],
            "Match Count CLS": [],
            "Accuracy NN": [],
            "Accuracy CLS": [],
            "F1 Score NN": [],
            "F1 Score CLS": [],
            "Precision NN": [],
            "Precision CLS": [],
            "Recall NN": [],
            "Recall CLS": [],
        }

        target = result['target_page'].to_list()
        preds_nn = result['predictions_nn'].to_list()
        preds_classic = result['predictions_classic'].to_list()

        for i in range(10):
            predictions_nn = [target if target in pred[:min(i+1, len(pred))] else '' for pred, target in zip(preds_nn, target)]
            match_count_nn = sum([1 for pred, target in zip(predictions_nn, target) if pred == target])
            predictions_classic = [target if target in pred[:min(i+1, len(pred))] else '' for pred, target in zip(preds_classic, target)]
            match_count_classic = sum([1 for pred, target in zip(predictions_classic, target) if pred == target])

            metrics['Criteria'].append(f"Hypothesis {i+1}")
            metrics['Match Count NN'].append(match_count_nn)
            metrics['Accuracy NN'].append(f"{accuracy_score(target, predictions_nn) * 100:.2f} %")
            metrics['F1 Score NN'].append(f"{f1_score(target, predictions_nn, average='weighted') * 100:.2f} %")
            metrics['Precision NN'].append(f"{precision_score(target, predictions_nn, average='weighted') * 100:.2f} %")
            metrics['Recall NN'].append(f"{recall_score(target, predictions_nn, average='weighted') * 100:.2f} %")

            metrics['Match Count CLS'].append(match_count_classic)
            metrics['Accuracy CLS'].append(f"{accuracy_score(target, predictions_classic) * 100:.2f} %")
            metrics['F1 Score CLS'].append(f"{f1_score(target, predictions_classic, average='weighted') * 100:.2f} %")
            metrics['Precision CLS'].append(f"{precision_score(target, predictions_classic, average='weighted') * 100:.2f} %")
            metrics['Recall CLS'].append(f"{recall_score(target, predictions_classic, average='weighted') * 100:.2f} %")

        print(f"PREDICTION RESULTS (Total samples: {user_data_test.shape[0]}, similar_clusters: {similar_clusters}, selected_news: {selected_news})")
        print(pl.DataFrame(metrics))

