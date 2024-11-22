module "stock_price_prediction" {
  source = "./executor"
  script_dir = "stock_price_prediction"
  entry_point = "main"
  epochs = 200
  output_dir = var.output_dir
}