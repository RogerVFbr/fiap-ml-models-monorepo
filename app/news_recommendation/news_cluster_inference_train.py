from random import random

import polars as pl
import polars.selectors as cs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

from app.news_recommendation import time_it
from app.news_recommendation.news_cluster_inference_model import NewsClusterInferenceModel


class NewsClusterInferenceTrain:

    TRAIN_EPOCHS = 10

    def __init__(self):
        self.model = None
        self.criterion = None
        self.optimizer = None

    @time_it
    def execute(self, user_data_train: pl.DataFrame):
        # user_data_train = user_data_train.select(pl.all().gather(range(10000)))
        self.set_seed(42)
        feature_count, X_train, X_test, y_train, y_test = self.setup_train_and_test(user_data_train)
        train_loader = self.setup_pytorch_set_and_loader(X_train, y_train)
        self.setup_model(feature_count, y_train)
        self.train_model(train_loader)
        benchmarks = self.calculate_benchmarks(user_data_train)
        self.evaluate_model(X_test, y_test, benchmarks)


    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @time_it
    def setup_train_and_test(self, user_data_train: pl.DataFrame):
        data = user_data_train.select([cs.starts_with("cluster_"), "target_cluster"]).to_torch()
        X = data[:, :-1]
        y = data[:, -1].long()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X.shape[1], X_train, X_test, y_train, y_test

    @time_it
    def setup_pytorch_set_and_loader(self, X_train, y_train):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
        return train_loader

    @time_it
    def setup_model(self, feature_count, y_train):
        # Calculate class weights
        class_sample_count = np.array([len(np.where(y_train.numpy() == t)[0]) for t in np.unique(y_train.numpy())])
        weight = 1. / class_sample_count
        class_weights = torch.FloatTensor(weight).to(y_train.device)

        self.model = NewsClusterInferenceModel(feature_count, feature_count)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    @time_it
    def train_model(self, train_loader):
        num_epochs = self.TRAIN_EPOCHS
        for epoch in range(num_epochs):
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(dtype=torch.float32)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    @time_it
    def evaluate_model(self, X_test, y_test, benchmarks):
        latest_percent, latest_percent_per_class, most_common_percent, most_common_percent_per_class = benchmarks

        X_test = X_test.to(dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            print(f'Model accuracy: {accuracy*100:.2f}%')
            print(f'Benchmarks: {latest_percent:.2f}% (latest), {most_common_percent:.2f}% (most common)')

            # Calculate accuracy per class
            class_accuracies = []
            for class_label in y_test.unique():
                class_mask = (y_test == class_label)
                class_correct = (predicted[class_mask] == y_test[class_mask]).sum().item()
                class_total = class_mask.sum().item()
                class_accuracy = class_correct / class_total
                class_accuracies.append({"class_label": class_label.item(), "model": class_accuracy*100})

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

            print("Accuracy per class:")
            print(class_accuracies_df)

            # Create a Polars DataFrame with expected and actual results
            results_df = pl.DataFrame({
                "Expected": y_test.tolist(),
                "Predicted": predicted.tolist()
            })

            distribution = results_df['Expected'].value_counts().sort("Expected").rename({
                "Expected": "Class",
                "count": "Expected"
            })

            actual_distribution = results_df['Predicted'].value_counts().sort("Predicted").rename({
                "Predicted": "Class",
                "count": "Predicted"
            })

            joined_distribution = distribution.join(actual_distribution, on="Class", suffix="_predicted")

            print("Distribution:\n", joined_distribution)

    @time_it
    def calculate_benchmarks(self, user_data_train: pl.DataFrame):
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

            # print(f'Percentage of rows were penultimate cluster matches target: {latest_percent:.2f} %')

            # Calculate and print matches per class
            latest_percent_per_class = penultimate_clusters.with_columns(
                (pl.col("penultimate_cluster") == user_data_train["target_cluster"]).alias("matches")
            ).group_by("penultimate_cluster").agg(
                (pl.sum("matches") * 100 / pl.count()).alias("matches_percentage")
            ).sort("penultimate_cluster")

            # print(latest_percent_per_class)

            #
            #
            #

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

            most_common_percent = (common_matches / total_rows) * 100

            # print(f'Percentage of rows where most common cluster matches target: {most_common_percent:.2f} %')

            # Calculate and print most common matches per class
            most_common_percent_per_class = most_common_clusters.with_columns(
                (pl.col("most_common_cluster") == user_data_train["target_cluster"]).alias("common_matches")
            ).group_by("most_common_cluster").agg(
                (pl.sum("common_matches") * 100 / pl.count()).alias("common_matches_percentage")
            ).sort("most_common_cluster")

            # print(most_common_percent_per_class)

            return latest_percent, latest_percent_per_class, most_common_percent, most_common_percent_per_class