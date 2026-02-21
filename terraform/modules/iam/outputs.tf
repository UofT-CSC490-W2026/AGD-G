output "modal_key_id" {
  value = aws_iam_access_key.modal_key.id
}

output "modal_key_secret" {
  value     = aws_iam_access_key.modal_key.secret
  sensitive = true
}
