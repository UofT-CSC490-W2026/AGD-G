import modal

def build_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_pyproject("pyproject.toml")
        .add_local_file("tests/data/images/xray-fish-profile.png", "/root/clean_image.png", copy=True)
        .add_local_file("tests/data/images/data_viz.png", "/root/data_viz.png", copy=True)
        .add_local_file("tests/data/images/target.jpg", "/root/target.jpg", copy=True)
        .add_local_python_source("agdg")
        .add_local_python_source("modal_run")
    )
