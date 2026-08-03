# Artifact Registry Module
# Creates the Docker repository used by Cartridge2 workloads.

resource "google_artifact_registry_repository" "images" {
  repository_id = "${var.name_prefix}-images"
  project       = var.project_id
  location      = var.region
  format        = "DOCKER"
  description   = "Container images for Cartridge2"

  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = var.images_to_keep
    }
  }

  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s" # 7 days
    }
  }

  labels = var.labels
}
