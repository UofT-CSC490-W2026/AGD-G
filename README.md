<img width="360" height="93" alt="AGD-G logo rendered in white monospace text on black background, depicting the letters AGD dot G in an artistic gradient in the foreground, with a subtle line chart in the background" src="https://github.com/user-attachments/assets/1d8ea8e5-0f1e-4bdb-b042-493b8bc8f6de" />

# AGD-G

<!-- coverage:project:start -->
![PROJECT coverage](https://img.shields.io/badge/coverage-100.00%25-brightgreen)

Overall automated line coverage: `100.00%`
<!-- coverage:project:end -->

AGD-G is a research pipeline for studying adversarial attacks on chart-based visual question
answering (chart VQA). The pipeline ingests chart datasets, generates baseline and adversarial
captions, applies gradient-based perturbations, and measures whether a vision-language model (VLM)
can be fooled into describing a chart as something it is not -- all while the image looks unchanged
to a human.

**This project is intended solely for academic research into the robustness of multimodal models.
It is not a tool for generating misinformation.**

## Table of contents

- [Getting started](#getting-started)
- [Pipeline stages](#pipeline-stages)
- [Datasets](#datasets)
- [Models and references](#models-and-references)
- [Environment variables](#environment-variables)
- [Project structure](#project-structure)

## Getting started

There are two ways to run AGD-G. Both execute the same underlying code.

| | **Local CLI (Docker)** | **Modal (cloud)** |
|---|---|---|
| GPU | Your own GPU via `--gpus all` | Modal-managed (H100, A10G) |
| Storage | Local Postgres + MinIO (Docker Compose) | AWS RDS + S3 |
| Accounts needed | None | Modal + AWS |
| Invoke | `agdg <stage>` | `modal run modal_run/<stage>.py` |

### Option A: Local CLI (Docker Compose)

No cloud accounts needed. Runs on your own machine or HPC cluster.

1. Install [Docker](https://docs.docker.com/get-docker/)
2. Start the local infrastructure and install the package:

```bash
docker compose up -d
cp .env.local.example .env.local
pip install .
```

3. Run any pipeline stage (see [Pipeline stages](#pipeline-stages)):

```bash
agdg ingest --limit 10
agdg target --strategy qwen --preview
agdg attack --limit 5
agdg evaluate --limit 5
```

You can also skip the local Python install and run everything inside Docker:

```bash
docker compose run --rm --profile app agdg <stage> [flags]

# For GPU stages (requires nvidia-container-toolkit)
docker compose run --rm --profile app --gpus all agdg attack --limit 5
```

For HPC clusters, the Dockerfile converts to Singularity:
`singularity build agdg.sif docker-daemon://agdg:latest`

### Option B: Modal (cloud GPU)

1. Create or join a [Modal](https://modal.com) workspace
2. Install the Modal CLI and run `modal setup`
3. Provision AWS infrastructure with Terraform (see `terraform/environments/dev`):

```bash
cd terraform/environments/dev && terraform init
terraform apply \
    -var=db_password='<a secure password>' \
    -var=bucket_name=agd-dev-tyson \
    -var=allowed_db_cidr_blocks='["138.51.0.0/16"]'
```

4. Create the required Modal secrets:

```bash
modal secret create aws \
  AWS_ACCESS_KEY_ID=... \
  AWS_SECRET_ACCESS_KEY=... \
  AWS_DEFAULT_REGION=ca-central-1

modal secret create aws-rds \
  DB_PASSWORD=...

modal secret create huggingface \
  HF_TOKEN=...
```

5. Run any pipeline stage (see [Pipeline stages](#pipeline-stages)):

```bash
modal run modal_run/ingest.py
modal run modal_run/target.py -- --strategy qwen
modal run modal_run/attack.py
modal run modal_run/evaluate.py -- --mode evaluate
```

Every `@app.function()` executes on Modal GPUs (H100 for target generation, A10G for attacks and
evaluation). The CLI streams output to your terminal; `Ctrl-C` stops the remote job.

## Pipeline stages

The pipeline has six stages. Each stage is **idempotent** -- it skips rows that already have
results, so re-running is safe.

### 1. Ingest

Import chart images and QA pairs from three HuggingFace datasets, upload raw PNGs to S3, and
insert metadata rows into PostgreSQL.

```bash
# Modal
modal run modal_run/ingest.py

# Local
agdg ingest
```

| Flag | Effect |
|------|--------|
| `--clean` | Wipe the database and S3 bucket before importing |
| `--skip-import` | Skip dataset import (useful to re-run preprocessing only) |
| `--skip-preprocess` | Skip the preprocessing step |
| `--limit N` | Cap the number of rows to process |

### 2. Preprocess

Validate every raw chart image, crop surrounding whitespace, letterbox-resize to 512x512, and
upload the cleaned PNG. This step is included in the ingest job by default, but can be run on its
own:

```bash
# Modal
modal run modal_run/ingest.py -- --skip-import

# Local
agdg ingest --skip-import
```

### 3. Clean caption

Run a VLM on each preprocessed chart to produce a baseline ("clean") caption that describes what
the chart actually shows. The default model is [LLaVA][llava] 1.5-7B.

```bash
# Modal
modal run modal_run/evaluate.py -- --mode clean

# Local
agdg clean-responses
```

| Flag | Effect |
|------|--------|
| `--model <hf-id>` | Use a different HuggingFace VLM (default `llava-hf/llava-1.5-7b-hf`) |
| `--limit N` | Cap rows |

### 4. Target caption

A targeting strategy takes each chart image and its clean caption, then generates a plausible but
**incorrect** caption -- same chart type, different subject. The default strategy uses
[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) with chain-of-thought
prompting.

```bash
# Modal
modal run modal_run/target.py -- --strategy qwen

# Local
agdg target --strategy qwen
```

| Flag | Effect |
|------|--------|
| `--strategy <name>` | Targeting strategy (default `qwen`) |
| `--source <dataset>` | Restrict to a single dataset source |
| `--batch-size N` | Rows per batch (default 100) |
| `--preview` | Sample a few rows per source and print traces without writing to the DB |
| `--per-source N` | Rows per source in preview mode (default 10) |
| `--max-rows N` | Cap rows |

### 5. Attack

Apply an adversarial perturbation so the chart image looks unchanged to humans but a VLM reads it
as the target caption. The attack is PGD-style: iterative sign-gradient updates on a pixel-space
perturbation, optimized against a surrogate model.

```bash
# Modal
modal run modal_run/attack.py

# Local
agdg attack
```

| Flag | Effect |
|------|--------|
| `--method <name>` | Attack variant: `targeted_text` (default), `targeted_image`, `targeted_text_ocr`, `untargeted` |
| `--surrogate <name>` | Surrogate model: `llava` (default), `clip_text`, `clip_image`, `clip_text_patch` |
| `--steps N` | PGD iterations (default 300) |
| `--strategy <name>` | Restrict to adversarial images produced with this targeting strategy |
| `--limit N` | Cap rows |

**Attack variants:**

| Method | Surrogate | Description |
|--------|-----------|-------------|
| `targeted_text` | [LLaVA][llava] or [CLIP][clip] | Maximize image-to-target-text similarity; optionally repel from clean caption |
| `targeted_image` | [CLIP][clip] | Maximize image-to-target-image similarity |
| `targeted_text_ocr` | [LLaVA][llava] or [CLIP][clip] | Same as `targeted_text` but confine perturbation to text-like regions via edge/std masking |
| `untargeted` | [CLIP][clip] | Minimize similarity to the clean image (no target) |

### 6. Evaluate

Query the VLM on each adversarial chart and compare its answer to both the clean and target
captions using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
sentence embeddings and cosine similarity. An attack **succeeds** when the VLM's answer is
semantically closer to the target caption than to the clean caption.

```bash
# Modal
modal run modal_run/evaluate.py -- --mode evaluate

# Local
agdg evaluate
```

| Flag | Effect |
|------|--------|
| `--strategy <name>` | Restrict evaluation to a targeting strategy |
| `--limit N` | Cap rows |

## Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| [ChartBench][chartbench] | Large-scale benchmark for chart understanding | [GitHub](https://github.com/IDEA-FinAI/ChartBench) |
| [ChartX][chartx] | Multi-type chart dataset with rich annotations | [HuggingFace](https://huggingface.co/datasets/princeton-nlp/ChartX) |
| [ChartQA-X][chartqax] | Explainable chart QA with rationale annotations | [GitHub](https://github.com/vis-nlp/ChartQA-X) |

## Models and references

| Model / Method | Role in pipeline | Reference |
|----------------|-----------------|-----------|
| [CLIP][clip] (ViT-L/14-336) | Attack surrogate -- image and text embeddings for similarity losses | Radford et al. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), 2021 |
| [LLaVA][llava] 1.5-7B | Clean captioning VLM, attack surrogate (token-likelihood loss), evaluation VLM | Liu et al. [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), 2023 |
| Qwen2.5-VL-7B-Instruct | Target caption generation | [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923), 2025 |
| [AttackVLM][attackvlm] | PGD-based adversarial attack framework | Zhao et al. [On Evaluating Adversarial Robustness of Large Vision-Language Models](https://arxiv.org/abs/2305.16934), 2023 |
| all-MiniLM-L6-v2 | Sentence-similarity scoring for evaluation | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

[chartbench]: https://github.com/IDEA-FinAI/ChartBench
[chartx]: https://huggingface.co/datasets/princeton-nlp/ChartX
[chartqax]: https://github.com/vis-nlp/ChartQA-X
[attackvlm]: https://arxiv.org/abs/2305.16934
[clip]: https://arxiv.org/abs/2103.00020
[llava]: https://arxiv.org/abs/2304.08485

## Environment variables

All storage configuration for the local CLI is driven by environment variables (see
`.env.local.example`). When none are set, the code falls back to the production AWS defaults, so
existing Modal deployments are unaffected.

| Variable | Default | Description |
|----------|---------|-------------|
| `AGDG_DB_HOST` | *(AWS RDS host)* | PostgreSQL host |
| `AGDG_DB_PORT` | `5432` | PostgreSQL port |
| `AGDG_DB_USER` | `modal_user` | PostgreSQL user |
| `AGDG_DB_PASSWORD` | *(unset -- uses IAM token)* | When set, uses plain password auth |
| `AGDG_DB_NAME` | `postgres` | Database name |
| `AGDG_DB_SSLMODE` | `require` | SSL mode (`disable` for local) |
| `AGDG_S3_ENDPOINT` | *(unset -- uses AWS)* | S3-compatible endpoint (e.g. `http://localhost:9000`) |
| `AGDG_S3_BUCKET` | `agd-dev-tyson` | S3 bucket name |
| `AGDG_S3_REGION` | `ca-central-1` | S3 region |

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

### Direct access to RDS

```bash
PGPASSWORD='...' psql -h agd-dev-postgres.cdsyi46ammw7.ca-central-1.rds.amazonaws.com -U postgres -d postgres -p 5432
```
