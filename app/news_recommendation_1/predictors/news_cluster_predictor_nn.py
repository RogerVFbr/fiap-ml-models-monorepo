import time

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from tqdm import tqdm
import inspect
import polars as pl
import polars.selectors as cs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from app.news_recommendation_1 import time_it

import app.news_recommendation_1.predictors.news_cluster_predictor_nn_model as model_module
from app.news_recommendation_1.data_repository import DataRepository


class NewsClusterNNPredictor:

    MODEL_NAME = "BetelgeuseNN"

    TRAIN_EPOCHS = 10
    LEARNING_RATE = 0.01
    BATCH_SIZE = 1024

    SELECTED_MODEL = None
    SELECTION_INDEX = 0
    PREDICTIONS = None

    FORCE_REPROCESS = False

    def __init__(self, force_reprocess: bool, output_path: str):
        self.FORCE_REPROCESS = force_reprocess

        self.set_seed(42)
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.device = "cpu"

        self.data_repo = DataRepository(output_path)

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    @time_it
    def execute(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame, similarity_matrix: pl.DataFrame):
        if self.data_repo.predicted_nn_user_data_test_parquet_exists() and not self.FORCE_REPROCESS:
            data = self.data_repo.load_predicted_nn_user_data_test_from_parquet()
            print(data.head(5))
            return data

        # user_data_train = user_data_train.select(pl.all().gather(range(10000)))
        # self.setup_apple_silicon_mps()

        feature_count, X_train, X_test, y_train, y_test = self.setup_train_and_test(user_data_train, user_data_test)
        models = self.initialize_models(feature_count)
        train_loader = self.setup_pytorch_loader(X_train, y_train)
        benchmarks = self.calculate_benchmarks(user_data_train)

        self.train_and_evaluate_models(models, y_train, train_loader, X_test, y_test, benchmarks, similarity_matrix)

        user_data_test = self.set_predictions_on_dataset(user_data_test)
        self.data_repo.save_predicted_nn_user_data_test_to_parquet(user_data_test)
        self.data_repo.save_pytorch_model(self.MODEL_NAME, self.SELECTED_MODEL, X_test.cpu().numpy())
        return user_data_test

    @time_it
    def setup_train_and_test(self, user_data_train: pl.DataFrame, user_data_test: pl.DataFrame):
        data_train = user_data_train.select([cs.starts_with("cluster_"), "target_cluster"]).to_torch()
        X_train = data_train[:, :-1].to(dtype=torch.float32, device=self.device)
        y_train = data_train[:, -1].long().to(self.device)

        data_test = user_data_test.select([cs.starts_with("cluster_"), "target_cluster"]).to_torch()
        X_test = data_test[:, :-1].to(dtype=torch.float32, device=self.device)
        y_test = data_test[:, -1].long().to(self.device)

        print("Shape of X_train .............:", X_train.shape)
        print("Shape of y_train .............:", y_train.shape)
        print("Data types of X_train ........:", X_train.dtype)
        print("Data types of y_train ........:", y_train.dtype)
        print("Shape of X_test ..............:", X_test.shape)
        print("Shape of y_test ..............:", y_test.shape)
        print("Data types of X_test .........:", X_test.dtype)
        print("Data types of y_test .........:", y_test.dtype)

        return X_train.shape[1], X_train, X_test, y_train, y_test

    @time_it
    def setup_pytorch_loader(self, X_train, y_train):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        return train_loader

    @time_it
    def train_and_evaluate_models(self, models, y_train, train_loader, X_test, y_test, benchmarks, similarity_matrix):
        hypothesis_1 = None
        hypothesis_2 = None
        hypothesis_3 = None

        for model in models:
            self.setup_model(y_train, model)
            model = self.train_model(train_loader, model)
            result = self.evaluate_model(X_test, y_test, benchmarks, model, similarity_matrix)
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
        print(self.SELECTED_MODEL)

        return hypothesis_1_df, hypothesis_2_df, hypothesis_3_df

    @time_it
    def initialize_models(self, feature_count):
        class_names = [(name, obj) for name, obj in inspect.getmembers(model_module, inspect.isclass)]
        models = [cls(feature_count, feature_count).to(self.device) for name, cls in class_names]
        return models

    def setup_model(self, y_train, model):
        y_train_cpu = y_train.cpu()
        class_sample_count = np.array(
            [len(np.where(y_train_cpu.numpy() == t)[0]) for t in np.unique(y_train_cpu.numpy())])
        weight = 1. / class_sample_count
        class_weights = torch.FloatTensor(weight).to(y_train.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)

    @time_it
    def train_model(self, train_loader, model):
        num_epochs = self.TRAIN_EPOCHS
        for epoch in range(num_epochs):
            epoch_loss = 0
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                start_time = time.time()
                for batch_X, batch_y in pbar:
                    self.optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    elapsed_time = time.time() - start_time
                    pbar.set_postfix(loss=epoch_loss / len(train_loader), elapsed_time=f"{elapsed_time:.2f}s")
        return model

    def evaluate_model(self, X_test, y_test, benchmarks, model, similarity_matrix):
        X_test = X_test.to(dtype=torch.float32)

        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)

            # self.evaluate_model_accuracy(predicted, y_test, benchmarks, model)
            metrics_hypothesis_1 = self.evaluate_hipothesis_1(predicted, y_test, model.__class__.__name__)
            metrics_hypothesis_2 = self.evaluate_hipothesis_2(predicted, y_test, model.__class__.__name__, similarity_matrix)
            metrics_hypothesis_3 = self.evaluate_hipothesis_3(predicted, y_test, model.__class__.__name__, similarity_matrix)

        hypothesis = metrics_hypothesis_1
        index = (hypothesis["Accuracy"][0] + hypothesis["F1 Score"][0] + hypothesis["Precision"][0] + hypothesis["Recall"][0])/ 4

        if index > self.SELECTION_INDEX:
            self.SELECTED_MODEL = model
            self.SELECTION_INDEX = index
            self.PREDICTIONS = predicted

        return metrics_hypothesis_1, metrics_hypothesis_2, metrics_hypothesis_3

    def evaluate_model_accuracy(self, predicted, y_test, benchmarks, model):
        latest_percent, latest_percent_per_class, most_common_percent, most_common_percent_per_class = benchmarks
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print()
        print( "+==============================+")
        print(f'|    Model accuracy: {accuracy*100:.2f}%    |')
        print( "+==============================+")
        print()
        print(f'Benchmarks:')
        print(f'  * Latest cluster matches target ........: {latest_percent:.2f}%')
        print(f'  * Most common cluster matches target ...: {most_common_percent:.2f}% ')
        print()
        print(model)

    def evaluate_hipothesis_1(self, predicted, y_test, model_name):
        accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
        recall = recall_score(y_test.cpu(), predicted.cpu(), average='weighted')
        f1 = f1_score(y_test.cpu(), predicted.cpu(), average='weighted')
        precision = precision_score(y_test.cpu(), predicted.cpu(), average='weighted')

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
        modified_predicted = predicted.clone().cpu()

        for i in range(len(y_test)):
            prediction = predicted[i]
            test = y_test[i]
            top3_predicted_similars = similarity_matrix_list[prediction][:3]
            if (test in top3_predicted_similars):
                modified_predicted[i] = test

        # print(f"Number of differences between predicted and modified_predicted: {(predicted != modified_predicted).sum().item()} ({((predicted != modified_predicted).sum().item() / len(y_test) * 100):.2f} %)")

        accuracy = accuracy_score(y_test.cpu(), modified_predicted.cpu())
        recall = recall_score(y_test.cpu(), modified_predicted.cpu(), average='weighted')
        f1 = f1_score(y_test.cpu(), modified_predicted.cpu(), average='weighted')
        precision = precision_score(y_test.cpu(), modified_predicted.cpu(), average='weighted')

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
        modified_predicted = predicted.clone().cpu()

        for i in range(len(y_test)):
            prediction = predicted[i]
            test = y_test[i]
            top3_predicted_similars = similarity_matrix_list[prediction][:4]
            if (test in top3_predicted_similars):
                modified_predicted[i] = test

        # print(f"Number of differences between predicted and modified_predicted: {(predicted != modified_predicted).sum().item()} ({((predicted != modified_predicted).sum().item() / len(y_test) * 100):.2f} %)")

        accuracy = accuracy_score(y_test.cpu(), modified_predicted.cpu())
        recall = recall_score(y_test.cpu(), modified_predicted.cpu(), average='weighted')
        f1 = f1_score(y_test.cpu(), modified_predicted.cpu(), average='weighted')
        precision = precision_score(y_test.cpu(), modified_predicted.cpu(), average='weighted')

        metrics = {
            "Architecture": [model_name],
            "Accuracy": [accuracy * 100],
            "F1 Score": [f1 * 100],
            "Precision": [precision * 100],
            "Recall": [recall * 100]
        }

        # print("Hypothesis 3: Prediction matches target or top 3 similar clusters")
        # print(pl.DataFrame(metrics))

        return metrics

    def evaluate_model_accuracy_per_class(self, predicted, y_test, benchmarks):
        latest_percent, latest_percent_per_class, most_common_percent, most_common_percent_per_class = benchmarks

        # Calculate accuracy per class
        class_accuracies = []
        for class_label in y_test.unique():
            class_mask = (y_test == class_label)
            class_correct = (predicted[class_mask] == y_test[class_mask]).sum().item()
            class_total = class_mask.sum().item()
            class_accuracy = class_correct / class_total
            class_accuracies.append({"class_label": class_label.item(), "model": class_accuracy * 100})

        class_accuracies_df = pl.DataFrame(class_accuracies)
        class_accuracies_df.insert_column(1, most_common_percent_per_class["common_matches_percentage"])
        class_accuracies_df.insert_column(1, latest_percent_per_class["matches_percentage"])
        class_accuracies_df = class_accuracies_df.with_columns([
            (pl.col("model") - pl.col("matches_percentage")).alias("model vs latest"),
            (pl.col("model") - pl.col("common_matches_percentage")).alias("model vs common")
        ])

        class_accuracies_df = class_accuracies_df.rename({
            "matches_percentage": "latest matches target",
            "common_matches_percentage": "most common matches target",
            "model": "model matches target",
        })

        print()
        print("Accuracy per class:")
        print(class_accuracies_df)

    def evaluate_model_class_distribution(self, predicted, y_test):
        # Create a Polars DataFrame with expected and actual results
        results_df = pl.DataFrame({
            "Expected": y_test.tolist(),
            "Predicted": predicted.tolist()
        })

        expected_distribution = results_df['Expected'].value_counts().sort("Expected").rename({
            "Expected": "Class",
            "count": "Expected"
        })

        predicted_distribution = results_df['Predicted'].value_counts().sort("Predicted").rename({
            "Predicted": "Class",
            "count": "Predicted"
        })

        predicted_distribution = expected_distribution.join(predicted_distribution, on="Class",
                                                            how="left").fill_null(0)
        print()
        print("Distribution:\n", predicted_distribution)
        print()

    @time_it
    def calculate_benchmarks(self, user_data_train: pl.DataFrame):
        latest_percent, latest_percent_per_class = self.calculate_benchmarks_latest(user_data_train)
        most_common_percent, most_common_percent_per_class = self.calculate_benchmarks_most_common(user_data_train)
        return latest_percent, latest_percent_per_class, most_common_percent, most_common_percent_per_class

    @time_it
    def calculate_benchmarks_latest(self, user_data_train: pl.DataFrame):
        # Extract the penultimate element from the "clusterHistory" column
        penultimate_clusters = user_data_train.select(
            pl.col("clusterHistory").map_elements(lambda x: x[-2] if len(x) > 1 else None, return_dtype=pl.Int64).alias("penultimate_cluster")
        )

        # Calculate the percentage of rows where the penultimate element matches the "target_cluster"
        matches = penultimate_clusters.with_columns(
            (pl.col("penultimate_cluster") == user_data_train["target_cluster"]).alias("matches")
        )["matches"].sum()

        total_rows = user_data_train.height
        latest_percent = (matches / total_rows) * 100

        # Calculate and print matches per class
        latest_percent_per_class = penultimate_clusters.with_columns(
            (pl.col("penultimate_cluster") == user_data_train["target_cluster"]).alias("matches")
        ).group_by("penultimate_cluster").agg(
            (pl.sum("matches") * 100 / pl.count()).alias("matches_percentage")
        ).sort("penultimate_cluster")

        return latest_percent, latest_percent_per_class

    @time_it
    def calculate_benchmarks_most_common(self, user_data_train: pl.DataFrame):
        most_common_clusters = user_data_train.select(
            pl.col("clusterHistory").map_elements(
                lambda x: max(set(x[:-1]), key=lambda y: x[:-1].count()) if len(x) > 1 else None,
                return_dtype=pl.Int64
            ).alias("most_common_cluster")
        )

        # Calculate the percentage of rows where the most common class matches the "target_cluster"
        common_matches = most_common_clusters.with_columns(
            (pl.col("most_common_cluster") == user_data_train["target_cluster"]).alias("common_matches")
        )["common_matches"].sum()

        total_rows = user_data_train.height

        most_common_percent = (common_matches / total_rows) * 100

        # Calculate and print most common matches per class
        most_common_percent_per_class = most_common_clusters.with_columns(
            (pl.col("most_common_cluster") == user_data_train["target_cluster"]).alias("common_matches")
        ).group_by("most_common_cluster").agg(
            (pl.sum("common_matches") * 100 / pl.count()).alias("common_matches_percentage")
        ).sort("most_common_cluster")

        return most_common_percent, most_common_percent_per_class

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_apple_silicon_mps(self):
        print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
        print(f"Is MPS available? {torch.backends.mps.is_available()}")

        # Set the device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def set_predictions_on_dataset(self, user_data_test: pl.DataFrame):
        user_data_test = user_data_test.with_columns(
            pl.Series(self.PREDICTIONS.cpu().numpy()).alias("predicted_cluster_nn")
        )

        return user_data_test