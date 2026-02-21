terraform {
  backend "s3" {
    bucket = "agd-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "ca-central-1"
  }
}
