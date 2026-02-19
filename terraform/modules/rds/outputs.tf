resource "aws_db_instance" "postgres" {
  identifier = "agd-${var.environment}-postgres"

  engine         = "postgres"
  engine_version = "15"
  instance_class = "db.t3.micro"

  allocated_storage = 20

  username = var.db_username
  password = var.db_password

  publicly_accessible    = false
  vpc_security_group_ids = [var.security_group_id]

  skip_final_snapshot = true
  deletion_protection = false
}
