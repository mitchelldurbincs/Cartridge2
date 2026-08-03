variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the cluster"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "vpc_id" {
  description = "VPC network ID"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for the cluster"
  type        = string
}

variable "pods_range_name" {
  description = "Name of the secondary range for pods"
  type        = string
}

variable "services_range_name" {
  description = "Name of the secondary range for services"
  type        = string
}

variable "enable_private_cluster" {
  description = "Enable private cluster (nodes have no public IPs)"
  type        = bool
  default     = false
}

variable "master_cidr" {
  description = "CIDR for the GKE master (only used if private cluster enabled)"
  type        = string
  default     = "172.16.0.0/28"
}

variable "release_channel" {
  description = "GKE release channel (RAPID, REGULAR, STABLE)"
  type        = string
  default     = "REGULAR"
}

variable "deletion_protection" {
  description = "Enable deletion protection for the cluster"
  type        = bool
  default     = true
}

variable "labels" {
  description = "Resource labels to apply to the cluster"
  type        = map(string)
  default     = {}
}
