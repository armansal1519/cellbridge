"""Immutable artifacts, resumable stage receipts and laptop resource limits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value, *, immutable=False):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and immutable:
        if path.read_text() != payload:
            raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    tmp.write_text(payload)
    os.replace(tmp, path)


def freeze_protocol(source, run):
    source, run = Path(source), Path(run)
    protocol = json.loads(source.read_text())
    path = run / "protocol.json"
    write_json(path, protocol, immutable=True)
    write_json(run / "protocol_lock.json", {"sha256": sha256(path), "original_source": str(source.resolve()),
        "meaning": "Frozen computational protocol. Not a public preregistration or proof of untouched data."}, immutable=True)
    return protocol


def freeze_implementation(run):
    """Resume only with exactly the same scientific code and package versions."""
    import importlib.metadata
    source = Path(__file__).parent
    value = {"source_sha256": {p.name: sha256(p) for p in sorted(source.glob("*.py"))},
             "packages": {name: importlib.metadata.version(name) for name in
                          ["numpy", "scipy", "pandas", "scikit-learn"]}}
    write_json(Path(run) / "implementation_lock.json", value, immutable=True)


@contextmanager
def exclusive_run(run):
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    with (run / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another worker is using {run}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class ResourceLimit(RuntimeError):
    """A resumable stop; never delete completed outputs to make room."""


@dataclass
class ResourceGuard:
    root: Path
    max_rss_gib: float = 32
    minimum_free_gib: float = 30
    max_hours: float = 168
    started: float = 0
    extra_pids: tuple = ()

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        path = self.root / "resource_usage.json"
        self.previous_seconds = json.loads(path.read_text()).get("elapsed_seconds", 0) if path.exists() else 0

    def check(self):
        import psutil
        proc = psutil.Process()
        rss = 0
        try:
            processes = [proc, *proc.children(recursive=True)]
            rss_scope = "process_tree"
        except (psutil.AccessDenied, PermissionError):
            processes = [proc]
            rss_scope = "current_process_only_sandbox_denied_child_enumeration"
        if self.extra_pids:
            for pid in self.extra_pids:
                try: processes.append(psutil.Process(pid))
                except psutil.NoSuchProcess: pass
            processes = list({p.pid:p for p in processes}.values())
            rss_scope = "current_process_and_explicit_workers"
        unreadable = []
        for p in processes:
            try:
                rss += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, PermissionError):
                unreadable.append(p.pid)
        elapsed = self.previous_seconds + time.monotonic() - self.started
        free = shutil.disk_usage(self.root).free
        write_json(self.root / "resource_usage.json", {"elapsed_seconds": elapsed, "rss_gib": rss / 2**30, "rss_scope": rss_scope,
            "monitored_pids": [p.pid for p in processes],
            "unreadable_rss_pids": unreadable,
            "free_gib": free / 2**30, "pid": os.getpid(), "updated_unix": time.time()})
        reason = None
        if unreadable: reason = "RSS accounting unavailable for live monitored processes"
        elif (self.root / "PAUSE").exists(): reason = "PAUSE file requested a stop"
        elif rss > self.max_rss_gib * 2**30: reason = "Process tree RSS limit exceeded"
        elif free < self.minimum_free_gib * 2**30: reason = "Free-space reserve reached"
        elif elapsed > self.max_hours * 3600: reason = "Cumulative experiment time limit reached"
        if reason:
            write_json(self.root / "stopped.json", {"reason": reason, "time": time.time()})
            raise ResourceLimit(reason)


def validate_receipt(folder):
    folder = Path(folder)
    receipt = folder / "complete.json"
    if not receipt.exists(): return False
    value = json.loads(receipt.read_text())
    for name, expected in value["files"].items():
        path = folder / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact integrity failure: {path}")
    return True


def require_receipt(folder):
    if not validate_receipt(folder):
        raise ValueError(f"Prerequisite stage is incomplete: {folder}")


def validate_scvaeit(folder):
    from .scvaeit import validate_benchmark_receipt
    return validate_benchmark_receipt(folder)


def complete_stage(folder, metadata=None):
    folder = Path(folder)
    paths = [p for p in folder.rglob("*") if p.is_file() and p.name != "complete.json" and not p.name.endswith(".partial")]
    write_json(folder / "complete.json", {"files": {str(p.relative_to(folder)): sha256(p) for p in paths},
        "metadata": metadata or {}}, immutable=True)
