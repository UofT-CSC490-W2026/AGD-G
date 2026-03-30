"""
Wipe S3 images and RDS tables, then recreate the schema.
"""
import logging

from agdg.data_pipeline import aws


def clean():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("clean")

    aws.wipe_s3(logger=log)
    aws.wipe_rds()
    aws.create_table_if_not_exists()
    log.info("Clean done — schema recreated")
