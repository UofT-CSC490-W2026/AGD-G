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


def build_web_image():
    """Image for the browser-facing Modal API."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_pyproject("pyproject.toml")
        .add_local_dir("src/agdg", "/root/agdg", copy=True)
        .add_local_dir("modal_run", "/root/modal_run", copy=True)
        .env({"PYTHONPATH": "/root"})
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
