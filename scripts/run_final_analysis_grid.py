from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs/final_analysis"
LOG_ROOT = PROJECT_ROOT / "results/final_analysis/logs"
BASELINE = {"frontier": "gamma_10", "aum": "multiplier_1"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded locked Core10 grid")
    parser.add_argument("--stage", choices=["frontier", "aum"], required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--time-budget-hours", type=float, default=8.0)
    parser.add_argument("--hard-stop-hours", type=float, default=24.0)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 2:
        raise ValueError("max-workers must be 1 or 2 to keep aggregate memory below 12 GiB")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/prepare_final_analysis_configs.py"), "--stage", args.stage],
        cwd=PROJECT_ROOT,
        check=True,
    )
    configs = sorted(CONFIG_ROOT.glob(f"{args.stage}_*.yaml"))
    configs = [path for path in configs if not _is_baseline_config(path, args.stage)]
    skipped_complete: list[str] = []
    incomplete_configs: list[Path] = []
    for path in configs:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        checkpoint_root = PROJECT_ROOT / values["checkpoint"]["root"]
        status_path = checkpoint_root / "run_status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8")).get("status")
            if status_path.exists()
            else None
        )
        if status == "complete":
            skipped_complete.append(path.stem)
        else:
            incomplete_configs.append(path)
    configs = incomplete_configs
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    pending = list(configs)
    running: dict[subprocess.Popen[str], tuple[Path, object, object]] = {}
    completed: list[dict[str, object]] = []
    stage_started = time.perf_counter()
    peak_aggregate_working_set = 0
    try:
        while pending or running:
            while pending and len(running) < args.max_workers:
                config = pending.pop(0)
                stdout = (LOG_ROOT / f"{config.stem}.stdout.log").open("w", encoding="utf-8")
                stderr = (LOG_ROOT / f"{config.stem}.stderr.log").open("w", encoding="utf-8")
                worker_python = getattr(sys, "_base_executable", sys.executable)
                worker_environment = os.environ.copy()
                python_path = [
                    str(Path(sys.prefix) / "Lib/site-packages"),
                    str(PROJECT_ROOT / "src"),
                ]
                if worker_environment.get("PYTHONPATH"):
                    python_path.append(worker_environment["PYTHONPATH"])
                worker_environment["PYTHONPATH"] = os.pathsep.join(python_path)
                process = subprocess.Popen(
                    [
                        worker_python,
                        str(PROJECT_ROOT / "scripts/run_final_analysis_scenario.py"),
                        "--config",
                        str(config),
                        "--time-budget-hours",
                        str(args.time_budget_hours),
                    ],
                    cwd=PROJECT_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    env=worker_environment,
                )
                running[process] = (config, stdout, stderr)
                print(f"started {config.stem} pid={process.pid}", flush=True)
            elapsed = time.perf_counter() - stage_started
            if elapsed > args.hard_stop_hours * 3600:
                raise RuntimeError(f"{args.stage} exceeded {args.hard_stop_hours} hours")
            process_table = _process_parent_table()
            aggregate_working_set = sum(
                _tree_working_set_bytes(process.pid, process_table)
                for process in running
            )
            peak_aggregate_working_set = max(
                peak_aggregate_working_set, aggregate_working_set
            )
            if aggregate_working_set > 12 * 1024**3:
                raise RuntimeError(
                    f"{args.stage} aggregate working set exceeded 12 GiB"
                )
            for process, (config, stdout, stderr) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                stdout.close()
                stderr.close()
                status = "paused_time_budget" if code == 75 else "complete"
                completed.append(
                    {"config": str(config), "returncode": code, "status": status}
                )
                del running[process]
                print(f"finished {config.stem} returncode={code}", flush=True)
                if code not in {0, 75}:
                    raise RuntimeError(f"scenario failed: {config}; see {LOG_ROOT}")
            progress = {
                "stage": args.stage,
                "elapsed_seconds": elapsed,
                "pending": [path.stem for path in pending],
                "running": [values[0].stem for values in running.values()],
                "completed": completed,
                "skipped_complete": skipped_complete,
                "aggregate_working_set_gib": aggregate_working_set / 1024**3,
                "peak_aggregate_working_set_gib": peak_aggregate_working_set / 1024**3,
            }
            (LOG_ROOT / f"{args.stage}_progress.json").write_text(
                json.dumps(progress, indent=2), encoding="utf-8"
            )
            if running:
                time.sleep(15)
    except Exception:
        for process, (_, stdout, stderr) in running.items():
            _terminate_process_tree(process)
            stdout.close()
            stderr.close()
        raise
    payload = {
        "stage": args.stage,
        "passed": True,
        "elapsed_seconds": time.perf_counter() - stage_started,
        "max_workers": args.max_workers,
        "time_budget_hours": args.time_budget_hours,
        "peak_aggregate_working_set_bytes": peak_aggregate_working_set,
        "peak_aggregate_working_set_gib": peak_aggregate_working_set / 1024**3,
        "baseline_reused": BASELINE[args.stage],
        "skipped_complete": skipped_complete,
        "completed": completed,
    }
    (LOG_ROOT / f"{args.stage}_runtime.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _is_baseline_config(path: Path, stage: str) -> bool:
    return path.stem == f"{stage}_{BASELINE[stage]}"


def _working_set_bytes(pid: int) -> int | None:
    if sys.platform != "win32":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    process_query_limited_information = 0x1000
    process_vm_read = 0x0010
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(
        process_query_limited_information | process_vm_read, False, int(pid)
    )
    if not handle:
        return None
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        succeeded = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if succeeded else None
    finally:
        kernel32.CloseHandle(handle)


def _tree_working_set_bytes(pid: int, parents: dict[int, int]) -> int:
    pids = {int(pid)}
    changed = True
    while changed:
        changed = False
        for child, parent in parents.items():
            if parent in pids and child not in pids:
                pids.add(child)
                changed = True
    return sum(_working_set_bytes(value) or 0 for value in pids)


def _process_parent_table() -> dict[int, int]:
    if sys.platform != "win32":
        return {}

    class ProcessEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        return {}
    parents: dict[int, int] = {}
    try:
        entry = ProcessEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        succeeded = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while succeeded:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            succeeded = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return parents


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
