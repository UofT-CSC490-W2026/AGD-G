output "endpoint" {
  value = aws_db_instance.postgres.address
}

output "modal_policy_arn" {
  value = aws_iam_policy.modal_policy.arn
  description = "ARN of the RDS IAM connect policy for modal"
}
