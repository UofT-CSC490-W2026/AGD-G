<img width="360" height="93" alt="AGD-G logo rendered in white monospace text on black background, depicting the letters AGD dot G in an artistic gradient in the foreground, with a subtle line chart in the background" src="https://github.com/user-attachments/assets/1d8ea8e5-0f1e-4bdb-b042-493b8bc8f6de" />

# AGD-G

<!-- coverage:project:start -->
![PROJECT coverage](https://img.shields.io/badge/coverage-100.00%25-brightgreen)

Overall automated line coverage: `100.00%`
<!-- coverage:project:end -->

AGD-G is an adversarial attack pipeline targeting chart-based visual question answering (chart QA).
It ingests chart datasets (ChartBench, ChartX, ChartQA-X), preprocesses images, generates
clean VLM captions, produces target captions via a pluggable targeting strategy, runs adversarial
attacks (AttackVLM variants with CLIP and LLaVA surrogates), and evaluates attack success via
sentence-similarity scoring. All GPU-intensive stages run on [Modal](https://modal.com) with
data persisted in AWS S3 (images) and RDS PostgreSQL (metadata and captions).

## Pipeline stages

Each stage is idempotent -- it skips rows that already have results, so re-running is safe.

1. **Ingest** -- Import chart datasets from HuggingFace, upload raw images to S3, and insert
   `samples` rows.
   `modal run modal_run/ingest.py`

2. **Preprocess** -- Validate, crop whitespace, letterbox-resize to 512x512, and upload cleaned
   PNGs.
   `modal run modal_run/ingest.py -- --preprocess-only`

3. **Clean caption** -- Run a VLM (default: LLaVA 1.5-7B) on each preprocessed chart to generate
   a baseline caption stored in `clean_answers`.
   `modal run modal_run/evaluate.py -- --mode clean`

4. **Target caption** -- A targeting strategy (currently Qwen2.5-VL-7B) takes the chart image and
   clean caption, then produces a new caption that preserves chart type but changes the subject.
   `modal run modal_run/target.py -- --strategy qwen`

5. **Attack** -- Apply an adversarial perturbation (AttackVLM text/image/OCR variants) so the
   chart image looks unchanged to humans but a VLM reads it as the target caption.
   `modal run modal_run/attack.py`

6. **Evaluate** -- Query the VLM on each adversarial chart and compare its answer to the clean
   and target captions using sentence-transformer cosine similarity. An attack succeeds when the
   answer is closer to the target.
   `modal run modal_run/evaluate.py -- --mode evaluate`

## Project structure

```
src/agdg/
  data_pipeline/       Orchestration for each pipeline stage (ingest, preprocess,
                       clean caption, target caption, attack, evaluate)
    aws/               RDS helpers, S3 helpers, schema.sql
  attack/              Adversarial attack methods and surrogate models
    methods/           AttackVLM variants (text, image, OCR, untargeted)
    surrogates/        CLIP and LLaVA embedding models used as surrogates
  targeting/           Pluggable targeting strategies (Qwen, extensible)
  scoring/             Sentence-similarity evaluation helpers

modal_run/             Modal app entrypoints for each pipeline stage
tests/                 Pytest suite mirroring src/agdg/ and modal_run/
terraform/             AWS infrastructure (S3, RDS, IAM, security groups)
docs/                  Background on adversarial-guided diffusion (AGD)
```

## Setup

### Terraform (AWS infrastructure)

1. Install Terraform and the AWS CLI
2. `aws configure` -- enter credentials and set region to `ca-central-1`
3. `cd terraform/environments/dev && terraform init`

To apply changes:
```
terraform apply \
    -var=db_password='<a secure password>' \
    -var=bucket_name=agd-dev-tyson \
    -var=allowed_db_cidr_blocks='["138.51.0.0/16"]'
```

The production environment (`terraform/environments/prod`) is currently unused.

### Modal (GPU compute)

1. Create or join a [Modal](https://modal.com) workspace
2. Install the Modal CLI and run `modal setup`
3. Create the required secrets:
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

Run any pipeline stage with `modal run <entrypoint>`. All `@app.function()` work executes on
Modal GPUs; CLI output streams to your terminal. `Ctrl-C` stops the remote job.

### Command-line access to RDS

Install `postgresql` and connect directly:
```
PGPASSWORD='...' psql -h agd-dev-postgres.cdsyi46ammw7.ca-central-1.rds.amazonaws.com -U postgres -d postgres -p 5432
```
