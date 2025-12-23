import pytest
from prepare_annotations.cli import app
from typer.testing import CliRunner

runner = CliRunner()

def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "prepare-annotations version" in result.stdout

