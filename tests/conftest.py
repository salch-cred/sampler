"""Load the plugin as a package with a throwaway Hermes home."""

import importlib.util
import os
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

if "hermes_constants" not in sys.modules:
    try:
        import hermes_constants  # noqa: F401
    except ImportError:
        stub = types.ModuleType("hermes_constants")
        stub.get_hermes_home = lambda: pathlib.Path(os.environ["HERMES_HOME"])
        sys.modules["hermes_constants"] = stub

spec = importlib.util.spec_from_file_location(
    "sampler", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
module = importlib.util.module_from_spec(spec)
sys.modules["sampler"] = module
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    module._SESSIONS.clear()
    module._COUNTS.update(set=0, left=0)
    yield
    module._SESSIONS.clear()
