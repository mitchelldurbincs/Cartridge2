output "vpc_id" {
  description = "VPC network ID"
  value       = google_compute_network.vpc.id
}

output "subnet_id" {
  description = "GKE subnet ID"
  value       = google_compute_subnetwork.gke.id
}

output "pods_range_name" {
  description = "Name of the secondary range for pods"
  value       = "pods"
}

output "services_range_name" {
  description = "Name of the secondary range for services"
  value       = "services"
}
