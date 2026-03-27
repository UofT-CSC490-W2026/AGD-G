import sys
import types

from tests.helpers import ensure_modal_root, import_fresh, install_fake_modal


def test_clean_runs_expected_aws_steps():
    ensure_modal_root()
    install_fake_modal()
    calls = []
    fake_aws = types.ModuleType("aws")
    fake_aws.wipe_s3 = lambda logger=None: calls.append("wipe_s3")
    fake_aws.wipe_rds = lambda: calls.append("wipe_rds")
    fake_aws.create_table_if_not_exists = lambda: calls.append("create_table")
    sys.modules["aws"] = fake_aws

    module = import_fresh("data_pipeline.clean")
    module.clean()

    assert calls == ["wipe_s3", "wipe_rds", "create_table"]


def test_clean_main_calls_remote():
    ensure_modal_root()
    install_fake_modal()
    fake_aws = types.ModuleType("aws")
    fake_aws.wipe_s3 = lambda logger=None: None
    fake_aws.wipe_rds = lambda: None
    fake_aws.create_table_if_not_exists = lambda: None
    sys.modules["aws"] = fake_aws

    module = import_fresh("data_pipeline.clean")
    calls = []
    module.clean.remote = lambda: calls.append("remote")

    module.main()

    assert calls == ["remote"]
