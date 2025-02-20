variable "app_dir" {
  description = "Path to the local directory"
  type        = string
  default     = "app"
}

variable "script_dir" {
  description = "Script directory"
  type        = string
}

variable "entry_point" {
  description = "Module entrypoint"
  type        = string
}

variable "epochs" {
  description = "Epochs"
  type        = number
}

variable "output_dir" {
  description = "Output directory"
  type        = string
  default     = "output"
}