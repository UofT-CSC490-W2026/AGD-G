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
