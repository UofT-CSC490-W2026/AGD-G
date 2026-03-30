from .helpers import import_fresh, install_fake_modal


def test_get_allowed_origins_splits_csv():
    install_fake_modal()
    module = import_fresh("modal_run.web_api")

    origins = module.get_allowed_origins("https://a.example, https://b.example ,http://localhost:5500")

    assert origins == [
        "https://a.example",
        "https://b.example",
        "http://localhost:5500",
    ]


def test_mock_attack_response_echoes_uploaded_image():
    install_fake_modal()
    module = import_fresh("modal_run.web_api")

    payload = module._mock_attack_response("What is the graph about?", "42", b"fake-image")

    assert payload["mode"] == "mock"
    assert payload["clean_answer"].startswith("[mock-clean]")
    assert payload["adversarial_answer"].startswith("[mock-adv]")
    assert payload["original_image"] == payload["adversarial_image"]


def test_image_bytes_to_data_url_prefix():
    install_fake_modal()
    module = import_fresh("modal_run.web_api")

    value = module._image_bytes_to_data_url(b"abc")

    assert value.startswith("data:image/png;base64,")
