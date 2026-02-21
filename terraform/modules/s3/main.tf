resource "aws_s3_bucket" "data_lake" {
  bucket = var.bucket_name
  force_destroy = true
  tags = {
    Project     = "agd"
    Environment = var.environment
  }
}

resource "aws_iam_policy" "modal_policy" {
  name = "modal-s3-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:*"
        ]
        Resource = "*"
      }
    ]
  })
}
