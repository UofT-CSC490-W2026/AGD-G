from .helpers import import_fresh, install_fake_modal


def test_build_image_returns_configured_modal_image():
    install_fake_modal()
    module = import_fresh("modal_run.image")

    image = module.build_image()

    assert image is not None


def test_build_data_pipeline_image_returns_configured_modal_image():
    install_fake_modal()
    module = import_fresh("modal_run.image")

    image = module.build_data_pipeline_image()

    assert image is not None


def test_build_web_image_returns_configured_modal_image():
    install_fake_modal()
    module = import_fresh("modal_run.image")

    image = module.build_web_image()

    assert image is not None
