import sys
from pathlib import Path

from koopman_sbi.cli import main
from koopman_sbi.config import resolve_task_config_path
from run_benchmark_suite import main as benchmark_suite_main


def _run_cli(command: str, config_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["python", command, "--config", str(config_path)])
    main()


def _run_benchmark_suite_script(config_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["python", "--config", str(config_path)])
    benchmark_suite_main()


def test_train_flow_cli_smoke(tiny_config_path, monkeypatch):
    _run_cli("train-flow", tiny_config_path, monkeypatch)
    assert list((tiny_config_path.parent / "logs" / "two_moons" / "train_flow").glob("test_run/checkpoints/best_model.pt"))


def test_distill_koopman_cli_smoke(tiny_config_path, monkeypatch):
    _run_cli("distill-koopman", tiny_config_path, monkeypatch)
    assert list((tiny_config_path.parent / "logs" / "two_moons" / "distill_koopman").glob("test_run/checkpoints/best_model.pt"))


def test_benchmark_compare_cli_smoke(tiny_config_path, monkeypatch):
    _run_cli("benchmark-compare", tiny_config_path, monkeypatch)
    benchmark_dir = tiny_config_path.parent / "logs" / "two_moons" / "benchmark_compare" / "test_run" / "benchmark"
    assert (benchmark_dir / "comparison_summary.csv").exists()
    assert (benchmark_dir / "benchmark_spider.png").exists()
    assert (benchmark_dir / "worth_it_curve.png").exists()
    assert (benchmark_dir / "worth_it_inference_curve.png").exists()


def test_benchmark_suite_cli_smoke(tiny_config_path, monkeypatch):
    _run_cli("benchmark-suite", tiny_config_path, monkeypatch)
    benchmark_dir = tiny_config_path.parent / "logs" / "two_moons" / "benchmark_suite" / "test_run" / "benchmark"
    assert (benchmark_dir / "comparison_summary.csv").exists()
    assert (benchmark_dir / "benchmark_manifest.json").exists()


def test_benchmark_suite_script_smoke(tiny_config_path, monkeypatch):
    _run_benchmark_suite_script(tiny_config_path, monkeypatch)
    benchmark_dir = tiny_config_path.parent / "logs" / "two_moons" / "benchmark_suite" / "test_run" / "benchmark"
    assert (benchmark_dir / "comparison_summary.csv").exists()
    assert (benchmark_dir / "benchmark_manifest.json").exists()


def test_resolve_task_config_path(tmp_path: Path):
    task_config_path = tmp_path / "two_moons.yaml"
    task_config_path.write_text("task:\n  name: two_moons\n", encoding="utf-8")
    assert resolve_task_config_path("two_moons", str(tmp_path)) == task_config_path
    assert resolve_task_config_path("two-moons", str(tmp_path)) == task_config_path
