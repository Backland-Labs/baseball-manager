# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0"]
# ///
"""Tests for the single_file_uv_scripts feature.

Validates that all Python scripts are single-file UV executables:
  1. Each .py file begins with a PEP 723 inline metadata block (/// script)
     declaring its dependencies
  2. Scripts can be executed directly via 'uv run <script>.py' with no prior
     install step
  3. No requirements.txt, pyproject.toml, or setup.py is needed
  4. All dependencies including anthropic, pydantic are declared inline
  5. The main entry point can be run with 'uv run game.py'
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PEP723_OPEN = "# /// script"
PEP723_CLOSE = "# ///"

# All Python files in the project (excluding __pycache__, agent_logs, etc.)
def _all_py_files():
    """Return all .py files in the project tree."""
    files = []
    for f in sorted(PROJECT_ROOT.rglob("*.py")):
        rel = f.relative_to(PROJECT_ROOT)
        if any(part.startswith(".") or part == "__pycache__" or part == "agent_logs"
               for part in rel.parts):
            continue
        files.append(f)
    return files


def _parse_pep723_block(text: str) -> str | None:
    """Extract the PEP 723 metadata block content from a script, or None."""
    m = re.search(
        r"^# /// script\s*\n((?:#[^\n]*\n)*?)# ///",
        text,
        re.MULTILINE,
    )
    if m:
        return m.group(1)
    return None


def _extract_dependencies(block: str) -> list[str]:
    """Extract dependency strings from a PEP 723 metadata block.

    The block contains raw # prefixed lines from the PEP 723 section.
    First strip the leading '# ' from each line to get clean TOML-like content,
    then parse dependencies from it.
    """
    # Strip comment prefixes to get clean content
    clean_lines = []
    for line in block.split("\n"):
        # Remove leading '#' and optional single space
        stripped = re.sub(r"^#\s?", "", line)
        clean_lines.append(stripped)
    clean = "\n".join(clean_lines)

    deps = []
    # Try single-line: dependencies = ["foo", "bar"]
    m = re.search(r'dependencies\s*=\s*\[([^\]]*)\]', clean)
    if m:
        raw = m.group(1)
        deps = [d.strip().strip('"').strip("'") for d in raw.split(",") if d.strip().strip('"').strip("'")]
        return deps
    # Multi-line: dependencies = [\n  "foo",\n]
    lines = clean.split("\n")
    in_deps = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("dependencies"):
            in_deps = True
            if "[" in stripped:
                after_bracket = stripped.split("[", 1)[1]
                if "]" in after_bracket:
                    inner = after_bracket.split("]")[0]
                    deps = [d.strip().strip('"').strip("'") for d in inner.split(",") if d.strip().strip('"').strip("'")]
                    return deps
            continue
        if in_deps:
            if "]" in stripped:
                break
            cleaned = stripped.strip(",").strip('"').strip("'")
            if cleaned:
                deps.append(cleaned)
    return deps


# All Python files for parametrization
ALL_PY_FILES = _all_py_files()
ALL_PY_FILE_IDS = [str(f.relative_to(PROJECT_ROOT)) for f in ALL_PY_FILES]


# ---------------------------------------------------------------------------
# Step 1: Each .py file begins with a PEP 723 inline metadata block
# ---------------------------------------------------------------------------

class TestStep1PEP723MetadataBlocks:
    """Every .py file must begin with a PEP 723 inline metadata block."""

    @pytest.mark.parametrize("py_file", ALL_PY_FILES, ids=ALL_PY_FILE_IDS)
    def test_has_pep723_open_tag(self, py_file):
        text = py_file.read_text()
        assert PEP723_OPEN in text, (
            f"{py_file.relative_to(PROJECT_ROOT)} missing '# /// script' marker"
        )

    @pytest.mark.parametrize("py_file", ALL_PY_FILES, ids=ALL_PY_FILE_IDS)
    def test_has_pep723_close_tag(self, py_file):
        text = py_file.read_text()
        block = _parse_pep723_block(text)
        assert block is not None, (
            f"{py_file.relative_to(PROJECT_ROOT)} missing complete PEP 723 block "
            f"(needs both '# /// script' and closing '# ///')"
        )

    @pytest.mark.parametrize("py_file", ALL_PY_FILES, ids=ALL_PY_FILE_IDS)
    def test_pep723_block_is_at_start(self, py_file):
        text = py_file.read_text()
        # The PEP 723 block should start at line 1
        first_line = text.split("\n")[0]
        assert first_line.strip() == PEP723_OPEN, (
            f"{py_file.relative_to(PROJECT_ROOT)}: PEP 723 block must be the "
            f"very first line, got: {first_line!r}"
        )

    @pytest.mark.parametrize("py_file", ALL_PY_FILES, ids=ALL_PY_FILE_IDS)
    def test_declares_requires_python(self, py_file):
        text = py_file.read_text()
        block = _parse_pep723_block(text)
        assert block is not None
        assert "requires-python" in block, (
            f"{py_file.relative_to(PROJECT_ROOT)}: PEP 723 block must declare "
            f"requires-python"
        )

    @pytest.mark.parametrize("py_file", ALL_PY_FILES, ids=ALL_PY_FILE_IDS)
    def test_declares_dependencies(self, py_file):
        text = py_file.read_text()
        block = _parse_pep723_block(text)
        assert block is not None
        assert "dependencies" in block, (
            f"{py_file.relative_to(PROJECT_ROOT)}: PEP 723 block must declare "
            f"dependencies"
        )


# ---------------------------------------------------------------------------
# Step 2: Scripts can be executed with uv run
# ---------------------------------------------------------------------------

class TestStep2UVRunExecution:
    """Key scripts can be executed directly via 'uv run' with no prior install."""

    def test_game_py_dry_run(self):
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--dry-run"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"uv run game.py --dry-run failed:\n{result.stderr}"

    def test_game_py_sim_mode(self):
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--sim", "--seed", "42"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"uv run game.py --sim failed:\n{result.stderr}"

    def test_models_py_importable_via_uv(self):
        """models.py can be imported when run via uv."""
        result = subprocess.run(
            ["uv", "run", "--script", str(PROJECT_ROOT / "models.py"),
             "-c", ""],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        # models.py isn't directly executable with -c, so just verify the
        # import works via a Python one-liner
        result = subprocess.run(
            ["uv", "run", "--with", "pydantic>=2.0", "python", "-c",
             "import sys; sys.path.insert(0, '.'); from models import MatchupState; print('OK')"],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"models import failed:\n{result.stderr}"
        assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# Step 3: No separate requirements file or package config needed
# ---------------------------------------------------------------------------

class TestStep3NoSeparateRequirements:
    """No requirements.txt, pyproject.toml, or setup.py should exist."""

    def test_no_requirements_txt(self):
        assert not (PROJECT_ROOT / "requirements.txt").exists(), (
            "requirements.txt should not exist -- dependencies are declared inline"
        )

    def test_no_pyproject_toml(self):
        assert not (PROJECT_ROOT / "pyproject.toml").exists(), (
            "pyproject.toml should not exist -- dependencies are declared inline"
        )

    def test_no_setup_py(self):
        assert not (PROJECT_ROOT / "setup.py").exists(), (
            "setup.py should not exist -- dependencies are declared inline"
        )

    def test_no_setup_cfg(self):
        assert not (PROJECT_ROOT / "setup.cfg").exists(), (
            "setup.cfg should not exist -- dependencies are declared inline"
        )

    def test_no_pip_conf(self):
        assert not (PROJECT_ROOT / "Pipfile").exists(), (
            "Pipfile should not exist -- dependencies are declared inline"
        )


# ---------------------------------------------------------------------------
# Step 4: All third-party dependencies declared inline
# ---------------------------------------------------------------------------

# Map of file-path patterns to the packages they must declare
CORE_FILES_DEPS = {
    "game.py": ["anthropic", "pydantic"],
    "models.py": ["pydantic"],
    "simulation.py": ["pydantic"],
}

TOOL_FILES = [
    "tools/__init__.py",
    "tools/get_batter_stats.py",
    "tools/get_pitcher_stats.py",
    "tools/get_matchup_data.py",
    "tools/get_run_expectancy.py",
    "tools/get_win_probability.py",
    "tools/evaluate_stolen_base.py",
    "tools/evaluate_sacrifice_bunt.py",
    "tools/get_bullpen_status.py",
    "tools/get_pitcher_fatigue_assessment.py",
    "tools/get_defensive_positioning.py",
    "tools/get_defensive_replacement_value.py",
    "tools/get_platoon_comparison.py",
]


class TestStep4DependenciesDeclaredInline:
    """All third-party dependencies must be declared in PEP 723 metadata."""

    @pytest.mark.parametrize("relpath,expected_deps", list(CORE_FILES_DEPS.items()),
                             ids=list(CORE_FILES_DEPS.keys()))
    def test_core_file_declares_deps(self, relpath, expected_deps):
        fpath = PROJECT_ROOT / relpath
        text = fpath.read_text()
        block = _parse_pep723_block(text)
        assert block is not None
        deps = _extract_dependencies(block)
        dep_names = [d.split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip()
                     for d in deps]
        for pkg in expected_deps:
            assert pkg in dep_names, (
                f"{relpath} must declare '{pkg}' in its PEP 723 dependencies, "
                f"found: {dep_names}"
            )

    @pytest.mark.parametrize("relpath", TOOL_FILES, ids=TOOL_FILES)
    def test_tool_file_declares_anthropic(self, relpath):
        fpath = PROJECT_ROOT / relpath
        text = fpath.read_text()
        block = _parse_pep723_block(text)
        assert block is not None
        deps = _extract_dependencies(block)
        dep_names = [d.split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip()
                     for d in deps]
        assert "anthropic" in dep_names, (
            f"{relpath} must declare 'anthropic' in its PEP 723 dependencies, "
            f"found: {dep_names}"
        )

    def test_game_py_declares_anthropic_with_version(self):
        text = (PROJECT_ROOT / "game.py").read_text()
        block = _parse_pep723_block(text)
        assert block is not None
        deps = _extract_dependencies(block)
        anthropic_deps = [d for d in deps if d.startswith("anthropic")]
        assert len(anthropic_deps) >= 1
        assert any(">=" in d for d in anthropic_deps), (
            "anthropic dependency should specify a minimum version"
        )

    def test_game_py_declares_pydantic_with_version(self):
        text = (PROJECT_ROOT / "game.py").read_text()
        block = _parse_pep723_block(text)
        assert block is not None
        deps = _extract_dependencies(block)
        pydantic_deps = [d for d in deps if d.startswith("pydantic")]
        assert len(pydantic_deps) >= 1
        assert any(">=" in d for d in pydantic_deps), (
            "pydantic dependency should specify a minimum version"
        )


# ---------------------------------------------------------------------------
# Step 5: Main entry point runs with uv run game.py
# ---------------------------------------------------------------------------

class TestStep5MainEntryPoint:
    """The main entry point can be run with 'uv run game.py'."""

    def test_entry_point_is_game_py(self):
        assert (PROJECT_ROOT / "game.py").exists()

    def test_game_py_has_argparse_or_main(self):
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "if __name__" in text or "argparse" in text, (
            "game.py should have an entry point guard or argparse"
        )

    def test_uv_run_game_dry_run_shows_validation(self):
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--dry-run"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "ALL VALIDATIONS PASSED" in result.stdout

    def test_uv_run_game_sim_produces_output(self):
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--sim", "--seed", "99"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert len(result.stdout) > 50, "Sim mode should produce substantial output"

    def test_test_files_have_pep723(self):
        """All test files also have PEP 723 blocks for consistency."""
        test_dir = PROJECT_ROOT / "tests"
        for tf in sorted(test_dir.glob("test_*.py")):
            text = tf.read_text()
            block = _parse_pep723_block(text)
            assert block is not None, (
                f"{tf.name} missing PEP 723 metadata block"
            )
