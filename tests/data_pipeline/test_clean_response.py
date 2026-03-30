from unittest.mock import ANY, MagicMock, patch

import pytest


@pytest.fixture
def mock_aws():
    with patch("agdg.data_pipeline.clean_response.rds.get_db_connection") as mock_db, patch(
        "agdg.data_pipeline.clean_response.s3.get_image", return_value=b"png"
    ) as mock_get_image:
        conn = MagicMock()
        cur = MagicMock()
        cur.rowcount = 1
        conn.cursor.return_value.__enter__.return_value = cur
        mock_db.return_value.__enter__.return_value = conn
        yield {"conn": conn, "cur": cur, "get_image": mock_get_image}


def test_generate_clean_responses_no_rows(mock_aws):
    from agdg.data_pipeline.clean_response import generate_clean_responses

    mock_aws["cur"].fetchall.return_value = []

    with patch("agdg.data_pipeline.clean_response.load_vlm", return_value=(MagicMock(), MagicMock(), "cpu", "float32")):
        result = generate_clean_responses()

    assert result == {"processed": 0}
    mock_aws["cur"].execute.assert_called_with(ANY, ("llava-hf/llava-1.5-7b-hf",))


def test_generate_clean_responses_success(mock_aws):
    from agdg.data_pipeline.clean_response import generate_clean_responses

    mock_aws["cur"].fetchall.return_value = [(1, "uuid-1")]

    with patch("agdg.data_pipeline.clean_response.load_vlm", return_value=(MagicMock(), MagicMock(), "cpu", "float32")) as mock_load, patch(
        "agdg.data_pipeline.clean_response.generate_image_response",
        return_value="A bar chart showing data.",
    ) as mock_generate:
        result = generate_clean_responses(model_id="llava-1.5")

    assert result == {"processed": 1}
    mock_load.assert_called_once_with("llava-1.5")
    mock_generate.assert_called_once()


def test_generate_clean_responses_generic_model_success(mock_aws):
    from agdg.data_pipeline.clean_response import generate_clean_responses

    mock_aws["cur"].fetchall.return_value = [(1, "uuid-1")]

    with patch("agdg.data_pipeline.clean_response.load_vlm", return_value=(MagicMock(), MagicMock(), "cpu", "float32")) as mock_load, patch(
        "agdg.data_pipeline.clean_response.generate_image_response",
        return_value="A chart.",
    ):
        result = generate_clean_responses(model_id="generic-vqa-model")

    assert result == {"processed": 1}
    mock_load.assert_called_once_with("generic-vqa-model")


def test_generate_clean_responses_with_error(mock_aws):
    from agdg.data_pipeline.clean_response import generate_clean_responses

    mock_aws["cur"].fetchall.return_value = [(1, "uuid-1"), (2, "uuid-2")]

    with patch("agdg.data_pipeline.clean_response.load_vlm", return_value=(MagicMock(), MagicMock(), "cpu", "float32")), patch(
        "agdg.data_pipeline.clean_response.generate_image_response",
        side_effect=[Exception("S3 error"), "Recovered answer"],
    ):
        with patch("agdg.data_pipeline.clean_response.BATCH_SIZE", 1):
            result = generate_clean_responses()

    assert result == {"processed": 1}
    assert mock_aws["conn"].commit.call_count == 1
