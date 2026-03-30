variable "bucket_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "versioning" {
  type        = bool
  description = "Enable S3 object versioning for recovery."
  default     = false
}

variable "deletion_protection" {
  type        = bool
  description = "Prevent S3 from being deleted if non-empty."
}
