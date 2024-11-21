from app import ARGS
from app.stock_price_prediction.training import StockPricePrediction

if __name__ == "__main__":
    prediction = StockPricePrediction()
    prediction.EPOCHS = ARGS.epochs
    prediction.OUTPUT_PATH = ARGS.output
    prediction.run()

    # ARGS
    # No of Epochs
    # output path