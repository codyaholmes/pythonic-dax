"""The guides in docs/ run as written.

Each guide's ``python`` code blocks run top to bottom as one script, in a temporary
folder (the guides write small sample files). In the Streamlit guide, the block that
starts with ``# model.py`` is saved as model.py and the others make up app.py, which
runs through Streamlit's own test runner.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def blocks(guide: str) -> list[str]:
    found = BLOCK.findall((DOCS / guide).read_text(encoding="utf-8"))
    assert found, f"{guide} has no python code blocks"
    return found


@pytest.fixture
def workdir(tmp_path: Path) -> Iterator[Path]:
    # Guides refer to repository files such as tests/fixtures by relative path.
    for folder in ("tests", "examples"):
        (tmp_path / folder).symlink_to(ROOT / folder, target_is_directory=True)
    old = Path.cwd()
    os.chdir(tmp_path)
    # The Streamlit guide's app imports a module named "model", as the demo app does.
    sys.modules.pop("model", None)
    yield tmp_path
    sys.modules.pop("model", None)
    os.chdir(old)


def test_building_a_model(workdir: Path) -> None:
    script = "\n".join(blocks("building-a-model.md"))
    exec(compile(script, "building-a-model.md", "exec"), {})


def test_streamlit_guide(workdir: Path) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("altair")
    found = blocks("streamlit.md")
    modules = [b for b in found if b.startswith("# model.py")]
    assert len(modules) == 1
    (workdir / "model.py").write_text(modules[0], encoding="utf-8")
    app_code = "\n".join(b for b in found if b not in modules)
    (workdir / "app.py").write_text(app_code, encoding="utf-8")
    app = testing.AppTest.from_file(str(workdir / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    assert [m.label for m in app.metric] == ["Sales", "Units", "Margin %"]
    assert app.session_state["visits"] == 1

    # A rerun, as after a click: the model isn't built again and session state stays.
    app.multiselect(key="regions").select("West").run()
    assert not app.exception, app.exception
    assert app.session_state["visits"] == 2
