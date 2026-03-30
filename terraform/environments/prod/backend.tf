terraform {
  backend "s3" {
    bucket = "agdg-terraform-state-prod-v2"
    key    = "prod/terraform.tfstate"
    region = "ca-central-1"

    use_lockfile = true
    encrypt      = true
  }
}
