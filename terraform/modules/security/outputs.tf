output "db_sg_id" {
  value = aws_security_group.db_sg.id
  description = "Security group ID for the RDS database"
}
