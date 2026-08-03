variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "images_to_keep" {
  description = "Number of container image versions to keep"
  type        = number
  default     = 10
}

variable "labels" {
  description = "Resource labels"
  type        = map(string)
  default     = {}
}
