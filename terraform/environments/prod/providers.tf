provider "aws" {
  region = var.aws_region
}

provider "postgresql" {
  host            = rds.endpoint
  username        = "postgres"
  password        = var.db_password
  sslmode         = "require"
}

