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

## Modal setup

1. Create a Modal account, create a workspace, or join someone else's
2. Install Modal CLI
3. `modal setup`
4. Create secrets to grant Modal access to AWS and HuggingFace:
```
modal secret create aws \
  AWS_ACCESS_KEY_ID=... \
  AWS_SECRET_ACCESS_KEY=... \
  AWS_DEFAULT_REGION=ca-central-1

modal secret create aws-rds \
  DB_PASSWORD=...

modal secret create huggingface \
  HF_TOKEN=...
```

## Modal usage

**In Python**

Import Modal and any AWS utility functions you need:
```
import modal
from aws import get_db_connection, get_image, put_image
```

Define an image by chaining methods, using

- `.uv_pip_install(["package1", "package2", ...])` to make packages available
  (don't forget to also import them at the top of the file)
- `.add_local_file("local_path", "remote_path")`
  or `.add_local_dir("local_path", "remote_path")`
  to upload needed files, including Python modules, to the image

```
aws_image = (
        modal.Image.debian_slim()
        .uv_pip_install(["boto3", "psycopg2-binary"])
        .add_local_file("aws.py", "/root/aws.py")
    )
```

Define an app, using

- `image=` to the image you created above
- `secrets=[modal.Secret.from_name("secret-name")]` to make secrets available
  as environment variables (note that you need to include `"aws"` for S3 access
  and both `"aws"` and `"aws-rds"` for RDS access)

```
app = modal.App(
        "app-name",
        image=aws_image,
        secrets=[
            modal.Secret.from_name("aws"),
            modal.Secret.from_name("aws-rds"),
        ],
    )
```

Decorate any top-level function that needs to run on Modal servers with `@app.function()`.
You do *not* need to do this for functions that that function calls.
Call such a function with `name.remote`.

```
def bar(y):
    return 2 * y

@app.function()
def foo(x):
    for i in range(10000):
        print(bar(x + i))

def main():
    print(bar(80))
    foo.remote()
```

Define the entrypoint by prefixing it with `@app.local_entrypoint()`:

```
@app.local_entrypoint()
def main():
    foo.remote()
```

**In the shell**

Now, just run your app with `modal run do_stuff.py`. Execution will start at the entrypoint.
All `@app.function()` functions will be run on Modal servers and all CLI output will appear
in your terminal. If you cancel the process with `Control-C`, it will stop on the Modal server too.

## Command-line access to RDS

Install `postgresql`, replace `...` with the database password, and access the database locally:
```
PGPASSWORD='...' psql -h agd-dev-postgres.cdsyi46ammw7.ca-central-1.rds.amazonaws.com -U postgres -d postgres -p 5432
```
