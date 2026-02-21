resource "aws_db_instance" "postgres" {
  identifier = "agd-${var.environment}-postgres"

  engine         = "postgres"
  engine_version = "15"
  instance_class = "db.t3.micro"

  allocated_storage = 20

  username = var.db_username
  password = var.db_password
  iam_database_authentication_enabled = true

  publicly_accessible    = true
  vpc_security_group_ids = [var.security_group_id]

  skip_final_snapshot = true
  deletion_protection = false
}

##
# Allow modal user to log in to PostgreSQL
##
resource "postgresql_role" "modal_user" {
  depends_on = [aws_db_instance.postgres]

  name  = "modal_user"
  login = true
  inherit = true
  roles = ["rds_iam"]
}

resource "postgresql_schema" "public" {
  name  = "public"
  owner = postgresql_role.modal_user.name

  depends_on = [
    postgresql_role.modal_user
  ]
}

data "aws_caller_identity" "current" {}

resource "aws_iam_policy" "modal_policy" {
  name = "modal-rds-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "rds-db:connect"
        ]
        Resource = "arn:aws:rds-db:${var.aws_region}:${data.aws_caller_identity.current.account_id}:dbuser:${aws_db_instance.postgres.resource_id}/modal_user"
      }
    ]
  })
}

