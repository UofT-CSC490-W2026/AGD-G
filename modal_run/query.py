"""
Query adversarial evaluation results from RDS.

    modal run modal_run/query.py
    modal run modal_run/query.py --limit 10
"""
import modal
from modal_run.image import build_data_pipeline_image

app = modal.App(
    "agd-query",
    image=build_data_pipeline_image(),
    secrets=[modal.Secret.from_name("aws")],
)


@app.function()
def query(limit: int = 0):
    from agdg.data_pipeline.aws.rds import get_db_connection

    limit_clause = "LIMIT %s" if limit > 0 else ""
    params = (limit,) if limit > 0 else ()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    aa.id,
                    aa.answer_text,
                    aa.attack_succeeded,
                    aa.eval_meta,
                    ca.clean_answer,
                    ta.target_answer,
                    s.chart_source
                FROM adversarial_answers aa
                JOIN adversarial_charts ac ON ac.id = aa.adversarial_chart_id
                JOIN target_answers ta ON ta.id = ac.target_answer_id
                JOIN clean_answers ca ON ca.id = ta.clean_answer_id
                JOIN samples s ON s.id = ca.sample_id
                ORDER BY aa.id
                {limit_clause}
                """,
                params,
            )
            return cur.fetchall()


@app.local_entrypoint()
def main(limit: int = 0):
    rows = query.remote(limit=limit)
    if not rows:
        print("No results found.")
        return

    for row in rows:
        id_, answer, succeeded, meta, clean, target, source = row
        print(f"\n{'='*70}")
        print(f"Row {id_} | source={source} | succeeded={succeeded}")
        print(f"  clean:        {clean}")
        print(f"  target:       {target}")
        print(f"  output:       {answer}")
        if meta:
            print(f"  score_clean:  {meta['score_clean']:.4f}")
            print(f"  score_target: {meta['score_target']:.4f}")
            print(f"  winner:       {meta['winner']}")
