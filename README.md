# AGD-G
Group-1 Project

## Terraform usage

1. Install Terraform
2. Install AWS CLI
3. `aws configure`
    - Enter AWS credentials
    - Enter `ca-central-1` as region
4. `cd terraform/environments/dev`
5. `terraform init`

Then to update infrastructure:
```
terraform apply \
    -var=db_password='<a secure password>' \
    -var=bucket_name=agd-dev-tyson \
    -var=allowed_db_cidr_blocks='["138.51.0.0/16"]'
```

Production environment (`terraform/environments/prod`) is current unused
