module "security" {
  source = "../../modules/security"

  environment             = var.environment
  allowed_db_cidr_blocks  = var.allowed_db_cidr_blocks
}

module "rds" {
  source = "../../modules/rds"

  environment       = var.environment
  db_username       = var.db_username
  db_password       = var.db_password
  security_group_id = module.security.db_sg_id
}

module "s3" {
  source = "../../modules/s3"

  environment = var.environment
  bucket_name = var.bucket_name
}
