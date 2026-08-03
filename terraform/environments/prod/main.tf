# Production Environment - Cartridge2
# Provisions GCP infrastructure for production workloads

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state is required for production
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "cartridge/prod"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Local values
locals {
  # Fixed because k8s/overlays/prod binds Filestore to cartridge-prod-vpc.
  name_prefix = "cartridge-prod"
  labels = {
    environment = "prod"
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

  # Production CIDR ranges (larger for more pods)
  subnet_cidr   = "10.0.0.0/18"
  pods_cidr     = "10.64.0.0/14"
  services_cidr = "10.68.0.0/18"
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

  # Production settings
  enable_private_cluster = true
  master_cidr            = "172.16.0.0/28"
  release_channel        = "STABLE"
  deletion_protection    = true
  labels                 = local.labels
}

# Artifact Registry
module "artifact_registry" {
  source = "../../modules/artifact-registry"

  project_id  = var.project_id
  region      = var.region
  name_prefix = local.name_prefix

  images_to_keep = 20
  labels         = local.labels
}
