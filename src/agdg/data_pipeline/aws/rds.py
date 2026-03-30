import os
from typing import Dict, Optional
from pathlib import Path
from uuid import UUID, uuid4
from contextlib import contextmanager

import psycopg2
try:
    from psycopg2.extras import Json
except Exception:  # pragma: no cover
    def Json(value):  # pragma: no cover
        return value

from agdg.data_pipeline.chart_type import ChartType

AWS_REGION = os.environ.get("AGDG_AWS_REGION", "ca-central-1")
RDS_HOST = os.environ.get("AGDG_DB_HOST", "agd-dev-postgres.cdsyi46ammw7.ca-central-1.rds.amazonaws.com")
RDS_PORT = int(os.environ.get("AGDG_DB_PORT", "5432"))
RDS_USER = os.environ.get("AGDG_DB_USER", "modal_user")
RDS_DB = os.environ.get("AGDG_DB_NAME", "postgres")
RDS_SSLMODE = os.environ.get("AGDG_DB_SSLMODE", "require")

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _get_password() -> str:
    """Return a DB password from env var or generate an IAM auth token."""
    static_pw = os.environ.get("AGDG_DB_PASSWORD")
    if static_pw is not None:
        return static_pw
    import boto3
    client = boto3.client("rds", region_name=AWS_REGION)
    return client.generate_db_auth_token(
        DBHostname=RDS_HOST,
        Port=RDS_PORT,
        DBUsername=RDS_USER,
    )


@contextmanager
def get_db_connection():
    """
    Make a connection to the RDS PostgreSQL database.
    Use `with get_db_connection() as conn:` and it will automatically
    close when the `with` block ends.

    When AGDG_DB_PASSWORD is set, uses that directly (for local Postgres).
    Otherwise falls back to AWS IAM token authentication.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            database=RDS_DB,
            user=RDS_USER,
            password=_get_password(),
            sslmode=RDS_SSLMODE,
        )
        yield conn
        conn.commit()
    finally:
        if conn:
            conn.close()

def create_table_if_not_exists() -> None:
    """
    Create the SQL table for this project's test data, and do nothing
    if it already exists.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            with open(SCHEMA_PATH) as f:
                cur.execute(f.read())

def insert_sample(
    cursor,
    source: str,
    chart_type: ChartType,
    question: str,
    answer: str,
    chart: UUID,
) -> None:
    cursor.execute(
        """
        INSERT INTO samples (
            chart_source,
            chart_type,
            sample_question,
            sample_answer,
            raw_chart
        )
        VALUES (%s, %s, %s, %s, %s);
        """,
        (source, str(chart_type), question, answer, str(chart))
    )

def insert_preprocessing(
    cursor,
    sample_id: int,
    chart: UUID,
    original_width: int,
    original_height: int,
    meta: Optional[Dict] = None,
) -> None:
    cursor.execute(
        """
        UPDATE samples
        SET
            original_width = %s,
            original_height = %s,
            preprocess_meta = %s,
            clean_chart = %s
        WHERE id = %s;
        """,
        (original_width, original_height, Json(meta) if meta is not None else None, str(chart), sample_id)
    )

def insert_clean_answer(
    cursor,
    sample_id: int,
    clean_answer: str,
    model_name: str,
) -> None:
    cursor.execute(
        """
        INSERT INTO clean_answers (
            sample_id,
            clean_answer_model,
            clean_answer
        )
        VALUES (%s, %s, %s);
        """,
        (sample_id, model_name, clean_answer)
    )

def insert_target_answer(
    cursor,
    clean_answer_id: int,
    target_answer: str,
    strategy: str,
) -> None:
    cursor.execute(
        """
        INSERT INTO target_answers (
            clean_answer_id,
            target_answer,
            target_strategy
        )
        VALUES (%s, %s, %s);
        """,
        (clean_answer_id, target_answer, strategy)
    )

def insert_adversarial_chart(
    cursor,
    target_id: int,
    chart: UUID,
    method: str,
    surrogate: str,
    meta: Optional[Dict] = None,
) -> None:
    cursor.execute(
        """
        INSERT INTO adversarial_charts (
            target_answer_id,
            adversarial_chart,
            attack_method,
            attack_surrogate,
            attack_meta
        )
        VALUES (%s, %s, %s, %s, %s);
        """,
        (target_id, str(chart), method, surrogate, Json(meta))
    )

def insert_adversarial_answer(
    cursor,
    chart_id: int,
    answer: str,
    model: str,
    success: bool,
    meta: Optional[Dict] = None,
) -> None:
    cursor.execute(
        """
        INSERT INTO adversarial_answers (
            adversarial_chart_id,
            adversarial_answer_model,
            answer_text,
            attack_succeeded,
            eval_meta
        )
        VALUES (%s, %s, %s, %s, %s);
        """,
        (chart_id, model, answer, success, meta)
    )

def iter_preprocessor_inputs(conn, batch_size=100):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, raw_chart
            FROM samples
            WHERE clean_chart IS NULL
        """)

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break

            for sample_id, raw_chart in rows:
                yield {
                    "sample_id": sample_id,
                    "raw_chart": raw_chart,
                }

def iter_clean_answer_inputs(conn, model_name, batch_size=100):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.id, s.clean_chart
            FROM samples s
            LEFT JOIN clean_answers ca
              ON ca.sample_id = s.id
             AND ca.clean_answer_model = %s
            WHERE s.clean_chart IS NOT NULL
              AND ca.id IS NULL
        """, (model_name,))

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break

            for sample_id, clean_chart in rows:
                yield {
                    "sample_id": sample_id,
                    "clean_chart": clean_chart,
                }

def iter_target_inputs(conn, strategy, source=None, batch_size=100):
    source_clause = "AND s.chart_source = %s" if source else ""
    params = [strategy]
    if source:
        params.append(source)

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT ca.id, ca.clean_answer, s.clean_chart
            FROM clean_answers ca
            JOIN samples s ON s.id = ca.sample_id
            LEFT JOIN target_answers ta
              ON ta.clean_answer_id = ca.id
             AND ta.target_strategy = %s
            WHERE ta.id IS NULL
              {source_clause}
        """, tuple(params))

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break

            for clean_answer_id, clean_answer, clean_chart in rows:
                yield {
                    "clean_answer_id": clean_answer_id,
                    "clean_answer": clean_answer,
                    "clean_chart": clean_chart,
                }


DATASET_SOURCES = ("ChartBench", "ChartX", "ChartQA-X")


def iter_target_inputs_sampled(conn, strategy, per_source: int = 10):
    """
    Yield up to *per_source* rows from each dataset source that do not yet
    have a target answer for *strategy*.  Rows are returned source-by-source
    in insertion order (ORDER BY ca.id).
    """
    union_parts = []
    params: list = []
    for src in DATASET_SOURCES:
        union_parts.append("""
            (SELECT ca.id, ca.clean_answer, s.clean_chart, s.chart_source
             FROM clean_answers ca
             JOIN samples s ON s.id = ca.sample_id
             LEFT JOIN target_answers ta
               ON ta.clean_answer_id = ca.id
              AND ta.target_strategy = %s
             WHERE ta.id IS NULL
               AND s.chart_source = %s
             ORDER BY ca.id
             LIMIT %s)
        """)
        params.extend([strategy, src, per_source])

    query = " UNION ALL ".join(union_parts)

    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        for clean_answer_id, clean_answer, clean_chart, chart_source in cur.fetchall():
            yield {
                "clean_answer_id": clean_answer_id,
                "clean_answer": clean_answer,
                "clean_chart": clean_chart,
                "chart_source": chart_source,
            }

def iter_attack_inputs(conn, attack_method, target_surrogate, batch_size=100):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                ta.id,
                ca.clean_answer,
                s.clean_chart,
                ta.target_answer
            FROM target_answers ta
            JOIN clean_answers ca ON ca.id = ta.clean_answer_id
            JOIN samples s ON s.id = ca.sample_id
            LEFT JOIN adversarial_charts ag
              ON ag.target_answer_id = ta.id
             AND ag.attack_method = %s
             AND ag.attack_surrogate = %s
            WHERE ag.id IS NULL
        """, (attack_method, target_surrogate))

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break

            for target_answer_id, clean_answer, clean_chart, target_answer in rows:
                yield {
                    "target_answer_id": target_answer_id,
                    "clean_answer": clean_answer,
                    "clean_chart": clean_chart,
                    "target_answer": target_answer,
                }

def iter_eval_inputs(conn, model_name, batch_size=100):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                ag.id,
                ca.clean_answer,
                s.clean_chart,
                ta.target_answer,
                ag.adversarial_chart
            FROM adversarial_charts ag
            JOIN target_answers ta ON ta.id = ag.target_answer_id
            JOIN clean_answers ca ON ca.id = ta.clean_answer_id
            JOIN samples s ON s.id = ca.sample_id
            LEFT JOIN adversarial_answers aa
              ON aa.adversarial_chart_id = ag.id
             AND aa.adversarial_answer_model = %s
            WHERE aa.id IS NULL
        """, (model_name,))

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                ag_id, clean_answer, clean_chart, target_answer, adv_chart = row
                yield {
                    "adversarial_chart_id": ag_id,
                    "clean_answer": clean_answer,
                    "clean_chart": clean_chart,
                    "target_answer": target_answer,
                    "adversarial_chart": adv_chart,
                }

def wipe_rds() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DROP SCHEMA public CASCADE;")
            cursor.execute("CREATE SCHEMA public;")
