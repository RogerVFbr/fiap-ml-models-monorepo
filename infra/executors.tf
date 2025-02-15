module "stock_price_prediction" {
  source = "./executor"
  script_dir = "stock_price_prediction"
  entry_point = "main"
  epochs = 200
  output_dir = var.output_dir
}

module "news_recommendation_1" {
  source = "./executor"
  script_dir = "news_recommendation_1"
  entry_point = "main"
  epochs = 10
  output_dir = var.output_dir
  depends_on = [
    module.stock_price_prediction
  ]
}