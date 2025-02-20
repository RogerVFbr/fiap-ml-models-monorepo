import logging
import numpy
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt

from app.stock_price_prediction.model import StockPricePredictionModel

class StockPricePrediction:

    MODEL_NAME = "Armadillo"
    SEQUENCE_LENGTH = 15
    MODEL_INPUT_SIZE = 1
    MODEL_HIDDEN_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.001

    SCALER = MinMaxScaler()
    MODEL = StockPricePredictionModel(MODEL_INPUT_SIZE, MODEL_HIDDEN_SIZE)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
    LOSS_FN = nn.MSELoss()

    LOGGER = logging.getLogger(__name__)
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
    OUTPUT_PATH = None
    FILE_BASE_NAME = None

    def __init__(self):
        self.LOGGER.info(f"Initializing {StockPricePrediction.__name__}")

        self.df = None
        self.df_scaled = None
        self.X = numpy.array([])
        self.y = numpy.array([])
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.loss_history = []

        self.FILE_BASE_NAME = self.build_file_base_name()

    def run(self):
        self.LOGGER.info(f"Running {StockPricePrediction.__name__}")

        self.load_data()
        self.scale_data()
        self.build_features_and_targets()
        self.split_train_and_test()
        output = self.train()
        self.report(output)
        self.save_model()

        self.LOGGER.info(f"{StockPricePrediction.__name__} terminated")

    def load_data(self):
        self.LOGGER.info(f"Loading data")

        self.df = pd.read_csv(f"{self.SCRIPT_PATH}/netflix.csv")
        self.df = self.df["Close"]

    def scale_data(self):
        self.LOGGER.info(f"Scaling data")

        self.df_scaled = self.SCALER.fit_transform(np.array(self.df)[..., None]).squeeze()

    def build_features_and_targets(self):
        self.LOGGER.info(f"Building features and targets")

        X, y = [], []

        for i in range(len(self.df_scaled) - self.SEQUENCE_LENGTH):
            X.append(self.df_scaled[i: i + self.SEQUENCE_LENGTH])
            y.append(self.df_scaled[i + self.SEQUENCE_LENGTH])

        self.X = np.array(X)[..., None]
        self.y = np.array(y)[..., None]

    def split_train_and_test(self):
        self.LOGGER.info(f"Splitting train and test")

        self.train_x = torch.from_numpy(self.X[:int(0.8 * self.X.shape[0])]).float()
        self.train_y = torch.from_numpy(self.y[:int(0.8 * self.X.shape[0])]).float()
        self.test_x = torch.from_numpy(self.X[int(0.8 * self.X.shape[0]):]).float()
        self.test_y = torch.from_numpy(self.y[int(0.8 * self.X.shape[0]):]).float()

    def train(self):
        self.LOGGER.info(f"Training")

        for epoch in range(self.EPOCHS):
            output = self.MODEL(self.train_x)
            loss = self.LOSS_FN(output, self.train_y)

            self.OPTIMIZER.zero_grad()
            loss.backward()
            self.OPTIMIZER.step()

            loss_detached = loss.detach().numpy()
            self.loss_history.append(loss_detached)

            if epoch % 10 == 0 and epoch != 0:
                self.LOGGER.info(f"<{epoch}> Loss: {loss_detached}")

        self.MODEL.eval()
        with torch.no_grad():
            return self.MODEL(self.test_x)

    def report(self, output):
        self.LOGGER.info(f"Saving report to path '{self.OUTPUT_PATH}'")
        pred = self.SCALER.inverse_transform(output.numpy())
        real = self.SCALER.inverse_transform(self.test_y.numpy())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(pred.squeeze(), color="red", label="predicted")
        ax1.plot(real.squeeze(), color="green", label="real")
        ax1.set_title('Predicted vs Real')
        ax1.legend()

        ax2.plot(self.loss_history, color="blue", label="loss")
        ax2.set_title(f'Loss per Epoch (Final Loss: {self.loss_history[-1]:.8f})')
        ax2.legend()

        plt.tight_layout()
        # plt.show()

        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)

        file_name = self.FILE_BASE_NAME + ".png"
        fig.savefig(file_name)

        self.LOGGER.info(f"Report saved -> {file_name}")

    def save_model(self):
        self.LOGGER.info(f"Saving model to path '{self.OUTPUT_PATH}'")

        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)

        file_name = self.FILE_BASE_NAME + ".pth"
        tensor = torch.FloatTensor(self.test_x)
        traced_cell = torch.jit.trace(self.MODEL, tensor)
        torch.jit.save(traced_cell, file_name)

        self.LOGGER.info(f"Model saved -> {file_name}")

    def build_file_base_name(self):
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename_with_datetime = f"{self.MODEL_NAME}-{current_datetime}"
        return os.path.join(self.OUTPUT_PATH, filename_with_datetime)