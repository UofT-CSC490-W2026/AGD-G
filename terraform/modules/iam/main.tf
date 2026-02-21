resource "aws_iam_user" "modal_user" {
  name = "modal-service-user"
}

resource "aws_iam_access_key" "modal_key" {
  user = aws_iam_user.modal_user.name
}

resource "aws_iam_user_policy_attachment" "modal_rds_attach" {
  user       = aws_iam_user.modal_user.name
  policy_arn = var.modal_rds_policy_arn
}

resource "aws_iam_user_policy_attachment" "modal_s3_attach" {
  user       = aws_iam_user.modal_user.name
  policy_arn = var.modal_s3_policy_arn
}
