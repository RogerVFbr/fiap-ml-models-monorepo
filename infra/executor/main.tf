data "archive_file" "this" {
  type        = "zip"
  source_dir  = "../${var.app_dir}/${var.script_dir}"
  output_path = "${var.script_dir}.zip"
}

resource "null_resource" "script_execution" {
  triggers = {
    src_hash = data.archive_file.this.output_sha
  }

  provisioner "local-exec" {
    command = <<EOF
      cd ../${var.app_dir}
      python -m ${var.app_dir}.${var.entry_point} --epochs ${var.epochs} --output "./${var.app_dir}/${var.output_dir}"
      EOF
  }
}

# python -m app.stock_price_prediction.main --epochs 20 --output "./app/output"

