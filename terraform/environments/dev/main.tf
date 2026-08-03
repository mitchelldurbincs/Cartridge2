# Development Environment - Cartridge2
# Provisions GCP infrastructure for development workloads

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Uncomment for remote state (recommended for teams)
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "cartridge/dev"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Local values
locals {
  # Fixed because k8s/overlays/dev binds Filestore to cartridge-dev-vpc.
  name_prefix = "cartridge-dev"
  labels = {
    environment = "dev"
    project     = "cartridge"
    managed-by  = "terraform"
  }
}

# Networking Module
module "networking" {
  source = "../../modules/networking"

  project_id  = var.project_id
  region      = var.region
  name_prefix = local.name_prefix
}

# GKE Autopilot Cluster
module "gke" {
  source = "../../modules/gke"

  project_id          = var.project_id
  region              = var.region
  name_prefix         = local.name_prefix
  vpc_id              = module.networking.vpc_id
  subnet_id           = module.networking.subnet_id
  pods_range_name     = module.networking.pods_range_name
  services_range_name = module.networking.services_range_name

  # Dev settings
  enable_private_cluster = false
  release_channel        = "REGULAR"
  deletion_protection    = false
  labels                 = local.labels
}

# Artifact Registry
module "artifact_registry" {
  source = "../../modules/artifact-registry"

  project_id  = var.project_id
  region      = var.region
  name_prefix = local.name_prefix

  images_to_keep = 5
  labels         = local.labels
}
