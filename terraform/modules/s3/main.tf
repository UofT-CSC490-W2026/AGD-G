resource "aws_s3_bucket" "data_lake" {
  bucket = var.bucket_name
  force_destroy = true
  tags = {
    Project     = "agd"
    Environment = var.environment
  }
}
