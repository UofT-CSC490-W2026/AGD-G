variable "aws_region" {
  type = string
}

variable "environment" {
  type = string
}

variable "db_username" {
  type = string
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "security_group_id" {
  type = string
}

variable "deletion_protection" {
  type        = bool
  description = "Prevent accidental RDS deletion."
  default     = false
}

variable "backup_retention_days" {
  type        = number
  description = "Days to retain automated RDS backups."
  default     = 0
}

