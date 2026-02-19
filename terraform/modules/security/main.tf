resource "aws_security_group" "db_sg" {
  name = "agd-${var.environment}-db-sg"

  dynamic "ingress" {
    for_each = length(var.allowed_db_cidr_blocks) > 0 ? [1] : []
    content {
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      cidr_blocks = var.allowed_db_cidr_blocks
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
