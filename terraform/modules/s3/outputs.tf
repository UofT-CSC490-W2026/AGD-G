output "bucket_name" {
  value = aws_s3_bucket.data_lake.bucket
}

output "modal_policy_arn" {
  value = aws_iam_policy.modal_policy.arn
  description = "ARN of the S3 IAM connect policy for modal"
}
