variable "aws_region" {
  type    = string
  default = "ca-central-1"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "db_username" {
  type    = string
  default = "postgres"
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "bucket_name" {
  type = string
}

variable "allowed_db_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed to access Postgres."
}
