from agdg.data_pipeline.chart_type import ChartType

from . import rds, s3
from .rds import create_table_if_not_exists, get_db_connection, insert_sample, wipe_rds
from .s3 import BUCKET, IMAGE_POSTFIX, IMAGE_PREFIX, get_image, put_image, wipe_s3

GraphType = ChartType
RDS_USER = rds.RDS_USER


def add_sample(cursor, source, graph_type, question, answer, chart):
    return insert_sample(cursor, source, graph_type, question, answer, chart)

__all__ = [
    "BUCKET",
    "ChartType",
    "GraphType",
    "IMAGE_POSTFIX",
    "IMAGE_PREFIX",
    "RDS_USER",
    "add_sample",
    "create_table_if_not_exists",
    "get_db_connection",
    "get_image",
    "insert_sample",
    "put_image",
    "rds",
    "s3",
    "wipe_rds",
    "wipe_s3",
]
