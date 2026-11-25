from types import SimpleNamespace

import pytest

from oprb_lab.__main__ import show_worker_logs


def test_launcher_preserves_driver_paths_for_workers(tmp_path, monkeypatch):
    from oprb_lab import __main__ as cli
    monkeypatch.setenv("LD_LIBRARY_PATH", "/runtime/driver")
    monkeypatch.setattr(cli, "collect", lambda *args: [])
    monkeypatch.setattr(cli, "summarize", lambda root: {"complete": False})
    environments = []
    def spawn(command, **kwargs):
        environments.append(kwargs["env"])
        return SimpleNamespace(returncode=1, poll=lambda: 1, wait=lambda: 1)
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    args = SimpleNamespace(data=tmp_path, limit=5, seed=42, gpus="0,1", model="paddle",
                           revision="main", max_tokens=10, max_pixels=100, max_tiles=1, output=tmp_path)
    with pytest.raises(SystemExit, match="Incomplete"):
        cli.launch(args)
    assert [env["CUDA_VISIBLE_DEVICES"] for env in environments] == ["0", "1"]
    assert all(env["LD_LIBRARY_PATH"] == "/runtime/driver" for env in environments)


def test_failed_worker_log_tail_is_visible(tmp_path, capsys):
    (tmp_path / "rank0.log").write_text("old\nloading\nRuntimeError: CUDA startup failed\n")
    show_worker_logs(tmp_path, [SimpleNamespace(returncode=1)], tail=2)
    output = capsys.readouterr().out
    assert "exit code 1" in output
    assert "CUDA startup failed" in output
    assert "old\n" not in output


def test_worker_missing_log_still_reports_exit_code(tmp_path, capsys):
    show_worker_logs(tmp_path, [SimpleNamespace(returncode=-9)])
    assert "exit code -9" in capsys.readouterr().out
