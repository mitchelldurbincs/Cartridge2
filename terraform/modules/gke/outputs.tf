output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.cluster.name
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.cluster.location
}
