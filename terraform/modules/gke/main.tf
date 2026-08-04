# GKE Module - Autopilot Cluster
# Creates a GKE Autopilot cluster for Cartridge2 workloads

resource "google_container_cluster" "cluster" {
  name     = "${var.name_prefix}-cluster"
  project  = var.project_id
  location = var.region

  # Enable Autopilot mode
  enable_autopilot = true

  # Network configuration
  network    = var.vpc_id
  subnetwork = var.subnet_id

  # Use VPC-native (alias IP) networking
  ip_allocation_policy {
    cluster_secondary_range_name  = var.pods_range_name
    services_secondary_range_name = var.services_range_name
  }

  # Private cluster configuration (optional)
  dynamic "private_cluster_config" {
    for_each = var.enable_private_cluster ? [1] : []
    content {
      enable_private_nodes    = true
      enable_private_endpoint = false
      master_ipv4_cidr_block  = var.master_cidr
    }
  }

  # Maintenance window
  maintenance_policy {
    recurring_window {
      start_time = "2024-01-01T09:00:00Z"
      end_time   = "2024-01-01T17:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA,SU"
    }
  }

  # Release channel
  release_channel {
    channel = var.release_channel
  }

  # Deletion protection (disable for dev)
  deletion_protection = var.deletion_protection

  # Resource labels
  resource_labels = var.labels
}
