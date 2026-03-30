"""Modal container image builders for GPU and data-pipeline jobs."""

import modal


def build_image():
    """Image for the per-image AttackVLM experiment (GPU, torch, etc.)."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_pyproject("pyproject.toml")
        .add_local_file("tests/data/images/xray-fish-profile.png", "/root/clean_image.png", copy=True)
        .add_local_file("tests/data/images/data_viz.png", "/root/data_viz.png", copy=True)
        .add_local_file("tests/data/images/target.jpg", "/root/target.jpg", copy=True)
        .add_local_python_source("agdg")
        .add_local_python_source("modal_run")
    )


def build_data_pipeline_image():
    """Image for data-pipeline jobs (ingest, attack-pipeline, evaluate)."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_pyproject("pyproject.toml")
        .add_local_file("src/agdg/data_pipeline/aws/schema.sql", "/root/agdg/data_pipeline/aws/schema.sql", copy=True)
        .add_local_python_source("agdg")
        .add_local_python_source("modal_run")
    )
