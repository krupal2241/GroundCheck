import pytest

from groundcheck import __version__
from groundcheck.cli import main


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage: groundcheck" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
