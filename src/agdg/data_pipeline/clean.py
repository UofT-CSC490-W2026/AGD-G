"""
Wipe S3 images and RDS tables, then recreate the schema.
"""
import logging

from agdg.data_pipeline.aws import rds, s3


def clean():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("clean")

    s3.wipe_s3(logger=log)
    rds.wipe_rds()
    rds.create_table_if_not_exists()
    log.info("Clean done — schema recreated")
