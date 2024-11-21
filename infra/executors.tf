module "stock_price_prediction" {
  source = "./executor"
  script_dir = "stock_price_prediction"
  entry_point = "stock_price_prediction.main"
  epochs = 100
  output_dir = var.output_dir
}