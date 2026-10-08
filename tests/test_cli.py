"""
Tests for SHP Command Line Interface (shp_core/cli.py).
"""

import sys
from io import StringIO
from unittest.mock import patch

import pytest
from shp_core.cli import main


def test_cli_info(capsys):
    """Test 'shp info' command."""
    ret = main(["info"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "SILENT HOPE PROTOCOL" in captured.out
    assert "System Status" in captured.out or "SYSTEM STATUS" in captured.out


def test_cli_no_args(capsys):
    """Test 'shp' with no args prints info."""
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "SILENT HOPE PROTOCOL" in captured.out


def test_cli_version(capsys):
    """Test 'shp --version' command."""
    with pytest.raises(SystemExit):
        main(["--version"])
    captured = capsys.readouterr()
    assert "Silent Hope Protocol v" in captured.out or "Silent Hope Protocol v" in captured.err


def test_cli_remember_and_recall(capsys, tmp_path):
    """Test 'shp remember' and 'shp recall' commands."""
    with patch("shp_core.cli.Path.home", return_value=tmp_path):
        ret_rem = main(["remember", "Test memory string for CLI test"])
        assert ret_rem == 0
        cap_rem = capsys.readouterr()
        assert "Memory stored successfully" in cap_rem.out

        ret_rec = main(["recall", "CLI test"])
        assert ret_rec == 0
        cap_rec = capsys.readouterr()
        assert "found 1 block" in cap_rec.out
