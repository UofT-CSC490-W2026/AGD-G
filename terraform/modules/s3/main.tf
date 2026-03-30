resource "aws_s3_bucket" "data_lake" {
  bucket        = var.bucket_name
  force_destroy = !var.deletion_protection
  tags = {
    Project     = "agd"
    Environment = var.environment
  }
}

resource "aws_iam_policy" "modal_policy" {
  name = "modal-s3-policy-${var.environment}"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = var.versioning ? "Enabled" : "Suspended"
  }
}
