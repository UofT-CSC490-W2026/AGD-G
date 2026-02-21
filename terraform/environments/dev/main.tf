module "security" {
  source = "../../modules/security"

  environment            = var.environment
  allowed_db_cidr_blocks = var.allowed_db_cidr_blocks
}

module "iam" {
  source = "../../modules/iam"

  modal_rds_policy_arn = module.rds.modal_policy_arn
  modal_s3_policy_arn = module.s3.modal_policy_arn
}

module "rds" {
  source = "../../modules/rds"

  aws_region            = var.aws_region
  environment           = var.environment
  db_username           = var.db_username
  db_password           = var.db_password
  security_group_id     = module.security.db_sg_id
  deletion_protection   = var.db_deletion_protection
  backup_retention_days = var.db_backup_retention_days
}

module "s3" {
  source = "../../modules/s3"

  environment = var.environment
  bucket_name = var.bucket_name
  versioning  = var.s3_versioning
}

module "modal" {
  source = "../../modules/modal"

  key_id     = module.iam.modal_key_id
  key_secret = module.iam.modal_key_secret
}
