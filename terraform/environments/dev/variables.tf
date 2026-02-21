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

# ── Protection settings (off for dev, on for prod via tfvars) ────

variable "db_deletion_protection" {
  type        = bool
  description = "Prevent accidental RDS deletion."
  default     = false
}

variable "db_backup_retention_days" {
  type        = number
  description = "Days to retain automated RDS backups."
  default     = 0
}

variable "s3_versioning" {
  type        = bool
  description = "Enable S3 object versioning for recovery."
  default     = false
}
