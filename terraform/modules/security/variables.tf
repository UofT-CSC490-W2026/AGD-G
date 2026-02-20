variable "environment" {
  type = string
}

variable "allowed_db_cidr_blocks" {
  type        = list(string)
  description = "Private CIDR blocks that are allowed to access PostgreSQL on 5432."
  default     = []

  #validation {
  #  condition     = !contains(var.allowed_db_cidr_blocks, "0.0.0.0/0")
  #  error_message = "Do not allow 0.0.0.0/0 for database ingress."
  #}
}
