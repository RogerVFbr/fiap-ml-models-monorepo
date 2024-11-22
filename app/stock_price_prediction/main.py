from app import ARGS
from app.stock_price_prediction.training import StockPricePrediction

if __name__ == "__main__":
    StockPricePrediction.EPOCHS = ARGS.epochs
    StockPricePrediction.OUTPUT_PATH = ARGS.output
    prediction = StockPricePrediction()
    prediction.run()