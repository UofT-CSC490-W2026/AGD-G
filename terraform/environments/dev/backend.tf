terraform {
  backend "s3" {
    bucket = "agdg-terraform-state-dev-v2"
    key    = "dev/terraform.tfstate"
    region = "ca-central-1"

    use_lockfile = true
    encrypt      = true
  }
}
