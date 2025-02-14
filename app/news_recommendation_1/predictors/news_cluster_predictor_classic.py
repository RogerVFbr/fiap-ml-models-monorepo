import polars as pl
import polars.selectors as cs
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight

from app.news_recommendation_1 import time_it
from app.news_recommendation_1.data_repository import DataRepository
from app.news_recommendation_1.predictors.news_cluster_predictor_classic_presets import NewsClusterClassicPredictorPresets


class NewsClusterClassicPredictor:

    MODEL_NAME = "BetelgeuseClassic"

    SELECTED_MODEL = None
    SELECTION_INDEX = 0
    PREDICTIONS = None

    FORCE_REPROCESS = False

    def __init__(self, force_reprocess: bool, output_path: str):
        self.FORCE_REPROCESS = force_reprocess
        self.presets = NewsClusterClassicPredictorPresets()
        self.data_repo = DataRepository(output_path)

    @time_it
    def execute(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, similarity_matrix: pl.DataFrame):
        if self.data_repo.predicted_classic_user_data_test_parquet_exists() and not self.FORCE_REPROCESS:
            data = self.data_repo.load_predicted_classic_user_data_test_from_parquet()
            print(data.head(5))
            return data

        train_and_test = self.setup_train_and_test(user_data_train, user_data_test)
        self.train_and_evaluate_models(train_and_test, similarity_matrix)
        user_data_test = self.set_predictions_on_dataset(user_data_test)

        self.data_repo.save_predicted_classic_user_data_test_to_parquet(user_data_test)
        self.data_repo.save_sklearn_model(self.MODEL_NAME, self.SELECTED_MODEL.best_estimator_)
        return user_data_test

    @time_it
    def setup_train_and_test(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame):
        data_train = user_data_train.select([cs.starts_with("cluster_"), "target_cluster"]).to_pandas()
        X_train = data_train.iloc[:, :-1]
        y_train = data_train.iloc[:, -1]

        # data_test = user_data_test.select([cs.starts_with("cluster_"), "target_cluster"]).to_pandas().dropna()
        data_test = user_data_test.select([cs.starts_with("cluster_"), "target_cluster"]).to_pandas()

        # # print rows of data_test with NaN values
        # print("NUUUUUL")
        # print(data_test[data_test.isnull().any(axis=1)])

        X_test = data_test.iloc[:, :-1]
        y_test = data_test.iloc[:, -1] # Contains NaN

        classes_weights = list(class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train
        ))

        return X_train, y_train, X_test, y_test, classes_weights

    @time_it
    def train_and_evaluate_models(self, train_and_test, similarity_matrix):
        hypothesis_1 = None
        hypothesis_2 = None
        hypothesis_3 = None

        for estimator_name, estimator in self.presets.get_presets().items():
            model = self.train_model(estimator_name, estimator['estimator'], estimator['params'], train_and_test)
            result = self.evaluate_model(estimator_name, model, train_and_test, similarity_matrix)
            metrics_hypothesis_1, metrics_hypothesis_2, metrics_hypothesis_3 = result

            if hypothesis_1 is None:
                hypothesis_1 = metrics_hypothesis_1
            else:
                for key, value in metrics_hypothesis_1.items():
                    hypothesis_1[key].append(value[0])

            if hypothesis_2 is None:
                hypothesis_2 = metrics_hypothesis_2
            else:
                for key, value in metrics_hypothesis_2.items():
                    hypothesis_2[key].append(value[0])

            if hypothesis_3 is None:
                hypothesis_3 = metrics_hypothesis_3
            else:
                for key, value in metrics_hypothesis_3.items():
                    hypothesis_3[key].append(value[0])

        print("+=============+")
        print("|   SUMMARY   |")
        print("+=============+")
        print()

        hypothesis_1_df = pl.DataFrame(hypothesis_1)
        hypothesis_2_df = pl.DataFrame(hypothesis_2)
        hypothesis_3_df = pl.DataFrame(hypothesis_3)

        print("Hypothesis 1: Prediction matches target")
        print(hypothesis_1_df)

        print("Hypothesis 2: Prediction matches target or top 2 similar clusters")
        print(hypothesis_2_df)

        print("Hypothesis 3: Prediction matches target or top 3 similar clusters")
        print(hypothesis_3_df)

        print()
        print("Selected Model:")
        print(self.SELECTED_MODEL.best_estimator_)

        return hypothesis_1_df, hypothesis_2_df, hypothesis_3_df

    @time_it
    def train_model(self, estimator_name, estimator, params, train_and_test,):
        X_train, y_train, _, _, classes_weights = train_and_test

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=params,
            scoring='accuracy',
            n_jobs=-1,
            cv=3,
            verbose=3,
        )

        if estimator_name == "XGBClassifier":
            model = grid_search.fit(X_train, y_train, sample_weight=classes_weights)
        else:
            model = grid_search.fit(X_train, y_train)

        return model

    @time_it
    def evaluate_model(self, estimator_name, model, train_and_test, similarity_matrix):
        _, _, X_test, y_test, _ = train_and_test

        prediction = model.predict(X_test)

        metrics_hypothesis_1 = self.evaluate_hipothesis_1(prediction, y_test, estimator_name)
        metrics_hypothesis_2 = self.evaluate_hipothesis_2(prediction, y_test, estimator_name, similarity_matrix)
        metrics_hypothesis_3 = self.evaluate_hipothesis_3(prediction, y_test, estimator_name, similarity_matrix)

        hypothesis = metrics_hypothesis_1
        index = (hypothesis["Accuracy"][0] + hypothesis["F1 Score"][0] + hypothesis["Precision"][0] + hypothesis["Recall"][0])/ 4

        if index > self.SELECTION_INDEX:
            self.SELECTED_MODEL = model
            self.SELECTION_INDEX = index
            self.PREDICTIONS = prediction

        return metrics_hypothesis_1, metrics_hypothesis_2, metrics_hypothesis_3

    def evaluate_hipothesis_1(self, predicted, y_test, model_name):
        accuracy = accuracy_score(y_test, predicted)
        recall = recall_score(y_test, predicted, average='weighted')
        f1 = f1_score(y_test, predicted, average='weighted')
        precision = precision_score(y_test, predicted, average='weighted')

        metrics = {
            "Architecture": [model_name],
            "Accuracy": [accuracy * 100],
            "F1 Score": [f1 * 100],
            "Precision": [precision * 100],
            "Recall": [recall * 100]
        }

        # print("Hypothesis 1: Prediction matches target")
        # print(pl.DataFrame(metrics))

        return metrics

    def evaluate_hipothesis_2(self, predicted, y_test, model_name, similarity_matrix: pl.DataFrame):
        similarity_matrix_list = similarity_matrix.to_numpy().tolist()
        modified_predicted = predicted.copy()
        y_test_list = y_test.tolist()

        for i in range(len(y_test_list)):
            prediction = predicted[i]
            test = y_test_list[i]
            top3_predicted_similars = similarity_matrix_list[prediction][:3]
            if (test in top3_predicted_similars):
                modified_predicted[i] = test

        # print(f"Number of differences between predicted and modified_predicted: {(predicted != modified_predicted).sum().item()} ({((predicted != modified_predicted).sum().item() / len(y_test) * 100):.2f} %)")

        accuracy = accuracy_score(y_test, modified_predicted)
        recall = recall_score(y_test, modified_predicted, average='weighted')
        f1 = f1_score(y_test, modified_predicted, average='weighted')
        precision = precision_score(y_test, modified_predicted, average='weighted')

        metrics = {
            "Architecture": [model_name],
            "Accuracy": [accuracy * 100],
            "F1 Score": [f1 * 100],
            "Precision": [precision * 100],
            "Recall": [recall * 100]
        }

        # print("Hypothesis 2: Prediction matches target or top 2 similar clusters")
        # print(pl.DataFrame(metrics))

        return metrics

    def evaluate_hipothesis_3(self, predicted, y_test, model_name, similarity_matrix: pl.DataFrame):
        similarity_matrix_list = similarity_matrix.to_numpy().tolist()
        modified_predicted = predicted.copy()
        y_test_list = y_test.tolist()

        for i in range(len(y_test_list)):
            prediction = predicted[i]
            test = y_test_list[i]
            top3_predicted_similars = similarity_matrix_list[prediction][:4]
            if (test in top3_predicted_similars):
                modified_predicted[i] = test

        # print(f"Number of differences between predicted and modified_predicted: {(predicted != modified_predicted).sum().item()} ({((predicted != modified_predicted).sum().item() / len(y_test) * 100):.2f} %)")

        accuracy = accuracy_score(y_test, modified_predicted)
        recall = recall_score(y_test, modified_predicted, average='weighted')
        f1 = f1_score(y_test, modified_predicted, average='weighted')
        precision = precision_score(y_test, modified_predicted, average='weighted')

        metrics = {
            "Architecture": [model_name],
            "Accuracy": [accuracy * 100],
            "F1 Score": [f1 * 100],
            "Precision": [precision * 100],
            "Recall": [recall * 100]
        }

        # print("Hypothesis 2: Prediction matches target or top 3 similar clusters")
        # print(pl.DataFrame(metrics))

        return metrics

    def set_predictions_on_dataset(self, user_data_test: pl.DataFrame):
        user_data_test = user_data_test.with_columns(
            pl.Series(self.PREDICTIONS).alias("predicted_cluster_classic")
        )

        return user_data_test
