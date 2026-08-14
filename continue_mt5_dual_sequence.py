from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_CPU = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON_MODELS = PROJECT_ROOT / ".venv-gpu" / "Scripts" / "python.exe"
FULL_POWER_CPU_CEILING = "88"
FULL_POWER_MAX_CORES = str(
    len(psutil.Process().cpu_affinity()) or (psutil.cpu_count(logical=True) or 1)
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, status: str, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updated_at_utc": now(), **extra}
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def read_status(path: Path) -> dict:
    """Read a frequently replaced Windows status file without killing the sequence."""
    last_error: OSError | json.JSONDecodeError | None = None
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read status file: {path}")


def wait_for_job(
    path: Path,
    timeout_hours: float,
    sequence_status: Path,
    expected_summary: Path | None = None,
) -> dict:
    deadline = time.monotonic() + timeout_hours * 3600
    while time.monotonic() < deadline:
        if path.exists():
            payload = read_status(path)
            if payload.get("status") in {"completed", "failed"}:
                return payload
            pid = payload.get("pid")
            if (
                expected_summary is not None
                and expected_summary.exists()
                and isinstance(pid, int)
                and not psutil.pid_exists(pid)
            ):
                recovered = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"status", "updated_at_utc"}
                }
                recovered.update(
                    {
                        "exit_code": None,
                        "completion_observation": (
                            "artifact_present_and_attached_process_exited"
                        ),
                    }
                )
                write_status(path, "completed", **recovered)
                return {"status": "completed", "updated_at_utc": now(), **recovered}
            write_status(
                sequence_status,
                "waiting_for_m5",
                m5_resource_status=payload,
                no_trading=True,
                lockboxes="sealed",
            )
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for {path}")


def run(command: list[str]) -> int:
    print("[sequence] run", json.dumps(command), flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Continue sealed Exness M5 -> M15 baselines.")
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    args = parser.parse_args()
    logs = PROJECT_ROOT / "logs" / "training"
    status_path = logs / "mt5_dual_sequence_status.json"
    m5_resource = logs / "mt5_xauusdm_m5_resource_status.json"
    m5_artifact = (
        PROJECT_ROOT
        / "artifacts"
        / "experiments"
        / "mt5_xauusdm_m5_exact_feed_baseline_v15_triple_h12_all_cpu_models_atr1_cost3_development"
    )
    m15_artifact = (
        PROJECT_ROOT
        / "artifacts"
        / "experiments"
        / "mt5_xauusdm_m15_exact_feed_hardened_v15_triple_h4_all_cpu_models_atr1_cost3_development"
    )
    m5_hardened_artifact = (
        PROJECT_ROOT
        / "artifacts"
        / "experiments"
        / "mt5_xauusdm_m5_exact_feed_hardened_v15_triple_h12_all_cpu_models_atr1_cost3_development"
    )
    write_status(status_path, "waiting_for_m5", no_trading=True, lockboxes="sealed")
    m5_result = wait_for_job(
        m5_resource,
        args.timeout_hours,
        status_path,
        expected_summary=m5_artifact / "summary.json",
    )
    if m5_result.get("status") != "completed" or not (m5_artifact / "summary.json").exists():
        write_status(status_path, "failed", stage="m5", m5_resource_status=m5_result)
        return 1
    write_status(status_path, "validating_m5", no_trading=True, lockboxes="sealed")
    m5_validation = run(
        [str(PYTHON_CPU), "scripts/validate_triple_artifacts.py", str(m5_artifact)]
    )
    if m5_validation != 0:
        write_status(status_path, "failed", stage="validate_m5", exit_code=m5_validation)
        return m5_validation

    m15_status = logs / "mt5_xauusdm_m15_resource_status.json"
    if not (m15_artifact / "summary.json").exists():
        write_status(status_path, "training_m15_hardened", no_trading=True, lockboxes="sealed")
        m15_exit = run(
            [
                str(PYTHON_CPU),
                "scripts/run_gaming_safe.py",
                "--full-power",
                "--cpu-ceiling",
                FULL_POWER_CPU_CEILING,
                "--max-cores",
                FULL_POWER_MAX_CORES,
                "--min-cores",
                "2",
                "--status-json",
                str(m15_status),
                "--",
                str(PYTHON_MODELS),
                "scripts/experiment_mt5_snapshot_triple.py",
                "--snapshot-csv",
                str(PROJECT_ROOT / "data/raw/mt5_XAUUSDm_M15_b516bca94db8f6ad.csv"),
                "--experiment-tag",
                "exact_feed_hardened_v15",
                "--horizon",
                "4",
                "--splits",
                "5",
                "--model-profile",
                "all_cpu_models",
            ]
        )
        if m15_exit != 0:
            write_status(status_path, "failed", stage="m15", exit_code=m15_exit)
            return m15_exit
    write_status(status_path, "validating_m15", no_trading=True, lockboxes="sealed")
    m15_validation = run(
        [str(PYTHON_CPU), "scripts/validate_triple_artifacts.py", str(m15_artifact)]
    )
    if m15_validation != 0:
        write_status(status_path, "failed", stage="validate_m15", exit_code=m15_validation)
        return m15_validation
    m5_hardened_status = logs / "mt5_xauusdm_m5_hardened_resource_status.json"
    if not (m5_hardened_artifact / "summary.json").exists():
        write_status(status_path, "training_m5_hardened", no_trading=True, lockboxes="sealed")
        m5_hardened_exit = run(
            [
                str(PYTHON_CPU),
                "scripts/run_gaming_safe.py",
                "--full-power",
                "--cpu-ceiling",
                FULL_POWER_CPU_CEILING,
                "--max-cores",
                FULL_POWER_MAX_CORES,
                "--min-cores",
                "2",
                "--status-json",
                str(m5_hardened_status),
                "--",
                str(PYTHON_MODELS),
                "scripts/experiment_mt5_snapshot_triple.py",
                "--snapshot-csv",
                str(PROJECT_ROOT / "data/raw/mt5_XAUUSDm_M5_42dca77237c0e62b.csv"),
                "--experiment-tag",
                "exact_feed_hardened_v15",
                "--horizon",
                "12",
                "--splits",
                "5",
                "--model-profile",
                "all_cpu_models",
            ]
        )
        if m5_hardened_exit != 0:
            write_status(status_path, "failed", stage="m5_hardened", exit_code=m5_hardened_exit)
            return m5_hardened_exit
    write_status(status_path, "validating_m5_hardened", no_trading=True, lockboxes="sealed")
    m5_hardened_validation = run(
        [str(PYTHON_CPU), "scripts/validate_triple_artifacts.py", str(m5_hardened_artifact)]
    )
    if m5_hardened_validation != 0:
        write_status(
            status_path,
            "failed",
            stage="validate_m5_hardened",
            exit_code=m5_hardened_validation,
        )
        return m5_hardened_validation
    write_status(status_path, "running_full_tests", no_trading=True, lockboxes="sealed")
    tests_exit = run(
        [
            str(PYTHON_CPU),
            "scripts/run_gaming_safe.py",
            "--full-power",
            "--cpu-ceiling",
            FULL_POWER_CPU_CEILING,
            "--max-cores",
            FULL_POWER_MAX_CORES,
            "--min-cores",
            "2",
            "--status-json",
            str(logs / "post_mt5_tests_resource_status.json"),
            "--",
            str(PYTHON_CPU),
            "-m",
            "pytest",
            "-q",
        ]
    )
    write_status(
        status_path,
        "completed" if tests_exit == 0 else "failed",
        stage="complete" if tests_exit == 0 else "tests",
        test_exit_code=tests_exit,
        m5_artifact=str(m5_artifact),
        m15_artifact=str(m15_artifact),
        m5_hardened_artifact=str(m5_hardened_artifact),
        no_trading=True,
        lockboxes="sealed",
    )
    return tests_exit


if __name__ == "__main__":
    raise SystemExit(main())
