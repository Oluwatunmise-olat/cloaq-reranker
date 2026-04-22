import importlib
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

APP_MODULES = [
    "protos.reranker_pb2",
    "protos.reranker_pb2_grpc",
    "reranker.config",
    "reranker.model",
    "reranker.service",
    "reranker.server",
]


def _read_pyproject_data() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def _read_dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text()


@pytest.mark.parametrize("module", APP_MODULES)
def test_runtime_modules_importable(module):
    importlib.import_module(module)


def test_hatch_wheel_packages_exist():
    packages = _read_pyproject_data()["tool"]["hatch"]["build"]["targets"]["wheel"][
        "packages"
    ]
    missing = [pkg for pkg in packages if not (ROOT / pkg).exists()]
    assert not missing, f"Missing package paths configured for wheel build: {missing}"


def test_project_script_entrypoint_is_callable():
    scripts = _read_pyproject_data()["project"]["scripts"]
    entrypoint = scripts["cloaq-reranker"]
    module_path, attr = entrypoint.rsplit(":", 1)

    module = importlib.import_module(module_path)
    target = getattr(module, attr, None)
    assert callable(target), f"Entrypoint '{entrypoint}' must resolve to a callable"


def test_lockfile_exists():
    assert (ROOT / "uv.lock").exists(), "uv.lock is missing"


def test_lockfile_in_sync():
    if shutil.which("uv") is None:
        pytest.skip("uv is not installed in this environment")

    with tempfile.TemporaryDirectory() as tmp_dir:
        env = os.environ.copy()
        env["UV_CACHE_DIR"] = tmp_dir
        result = subprocess.run(
            ["uv", "lock", "--check"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        pytest.fail(f"uv lock --check failed:\n{result.stderr}")


def test_dockerfile_uses_locked_sync():
    content = _read_dockerfile()
    assert "uv sync --locked" in content, (
        "Dockerfile must install with a locked dependency graph"
    )


def test_dockerfile_cmd_matches_project_script():
    content = _read_dockerfile()
    assert 'CMD ["cloaq-reranker"]' in content, (
        "Dockerfile CMD must invoke cloaq-reranker"
    )
