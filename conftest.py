import re
import json
from pathlib import Path

import pytest


def extract_test_number(name: str) -> int | None:
    match = re.match(r"test_(\d+)_", name)
    if not match:
        return None
    return int(match.group(1))


def pytest_addoption(parser):
    parser.addoption("--stop-at", default=None, help="Stop after this test number (e.g. --stop-at 15)")
    parser.addoption(
        "--reuse-passed",
        action="store_true",
        help="Skip tests that previously passed (from pass log file).",
    )
    parser.addoption(
        "--pass-log",
        default=".pytest_passed_tests.json",
        help="Path to persisted passed-test registry.",
    )


def pytest_configure(config):
    for i in range(1, 10):
        config.addinivalue_line("markers", f"phase{i}: test phase {i}")
    config._pass_log_path = Path(config.getoption("--pass-log"))
    config._passed_registry = set()
    config._failure_seen = False
    pass_log = config._pass_log_path
    if pass_log.exists():
        try:
            config._passed_registry = set(json.loads(pass_log.read_text()))
        except Exception:
            config._passed_registry = set()


def pytest_runtest_setup(item):
    stop_at = item.config.getoption("--stop-at")
    if stop_at:
        test_num = extract_test_number(item.name)
        if test_num and test_num > int(stop_at):
            pytest.skip(f"Skipping (--stop-at {stop_at})")
    if item.config.getoption("--reuse-passed"):
        if item.nodeid in item.config._passed_registry:
            pytest.skip("Previously passed (reuse-passed enabled)")


def _write_pass_registry(config):
    config._pass_log_path.write_text(
        json.dumps(sorted(config._passed_registry), ensure_ascii=False, indent=2)
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    config = item.config
    # Only mark pass if no earlier failure has occurred in this run.
    if report.when == "call" and report.failed:
        config._failure_seen = True
        return
    if report.when == "call" and report.passed and not config._failure_seen:
        config._passed_registry.add(item.nodeid)


def pytest_sessionfinish(session, exitstatus):
    _write_pass_registry(session.config)
