from unittest.mock import MagicMock, patch

from PIL import Image


def _make_conn(cursor):
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn


def test_attack_all_returns_zero_when_no_rows():
    from agdg.data_pipeline.attack import attack_all

    select_cur = MagicMock()
    select_cur.fetchall.return_value = []
    select_conn = _make_conn(select_cur)

    with patch("agdg.data_pipeline.attack.rds.get_db_connection") as mock_db:
        mock_db.return_value.__enter__.return_value = select_conn
        result = attack_all()

    assert result == {"attacked": 0, "rows_updated": 0}
    query, params = select_cur.execute.call_args[0]
    assert "LEFT JOIN adversarial_charts ac" in query
    assert params == ("targeted_text", "llava")


def test_attack_all_applies_limit_and_strategy_filters():
    from agdg.data_pipeline.attack import attack_all

    select_cur = MagicMock()
    select_cur.fetchall.return_value = []
    select_conn = _make_conn(select_cur)

    with patch("agdg.data_pipeline.attack.rds.get_db_connection") as mock_db:
        mock_db.return_value.__enter__.return_value = select_conn
        attack_all(max_rows=3, target_strategy="smoke-fast")

    query, params = select_cur.execute.call_args[0]
    assert "AND ta.target_strategy = %s" in query
    assert "LIMIT %s" in query
    assert params == ("targeted_text", "llava", "smoke-fast", 3)


def test_attack_all_success_and_error_paths():
    from agdg.data_pipeline.attack import attack_all

    select_cur = MagicMock()
    select_cur.fetchall.return_value = [
        (11, "clean answer 1", "chart-1", "target 1"),
        (12, "clean answer 2", "chart-2", "target 2"),
    ]
    insert_cur = MagicMock()
    insert_cur.rowcount = 1

    select_conn = _make_conn(select_cur)
    insert_conn = _make_conn(insert_cur)

    attack_method = MagicMock()
    attack_method.attack.side_effect = [
        Image.new("RGB", (8, 8), "red"),
        RuntimeError("boom"),
    ]

    with patch("agdg.data_pipeline.attack.rds.get_db_connection") as mock_db, patch(
        "agdg.data_pipeline.attack.build_attack_method",
        return_value=attack_method,
    ), patch(
        "agdg.data_pipeline.attack.s3.get_image",
        return_value=b"png-bytes",
    ), patch(
        "agdg.data_pipeline.attack.s3.put_image",
        return_value="adv-uuid",
    ), patch(
        "PIL.Image.open",
        return_value=Image.new("RGB", (8, 8), "white"),
    ):
        mock_db.return_value.__enter__.side_effect = [select_conn, insert_conn]
        with patch("agdg.data_pipeline.attack.BATCH_SIZE", 1):
            result = attack_all(method="targeted_text", surrogate="llava", steps=20)

    assert result == {"attacked": 1, "rows_updated": 1}
    _, kwargs = attack_method.attack.call_args_list[0]
    assert kwargs["target"] == "target 1"
    assert kwargs["strength"] == 1.0
    assert kwargs["hyperparameters"] == {"source_text": "clean answer 1", "steps": 20}
    insert_conn.commit.assert_called_once()
    insert_query, insert_params = insert_cur.execute.call_args[0]
    assert "INSERT INTO adversarial_charts" in insert_query
    assert insert_params[:4] == (11, "adv-uuid", "targeted_text", "llava")
