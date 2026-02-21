terraform {
  backend "s3" {
    bucket = "agd-terraform-state"
    key    = "dev/terraform.tfstate"
    region = "ca-central-1"
  }
}
