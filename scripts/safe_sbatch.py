# Copyright 2026 Husnain Rasool
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guarded sbatch: refuses a submission if the projected disk of every queued+running job of ours
(plus this one) would eat into free quota minus a safety margin (2026-09-18, after two
self-inflicted incidents: a name-matched scancel killed the wrong jobs, and launching several
multi-fold trainings + a cache build at once drove headroom from 25GB to 1.5GB, below the
checkpoint-write floors, silently skipping saves — the same failure mode as exp003/008/009).

Run on the HPC login node, from the code dir, with the project venv active (needs PyYAML):

    python scripts/safe_sbatch.py [--reuses JOBID] [--force] [--dry-run] [--headroom-gb 30] \\
        -- --array=0-4 slurm/train.sbatch configs/exp010_....yaml

Everything after `--` goes to sbatch verbatim (use `--opt=value` form so the script path is
unambiguous). What it does:
  1. Projects this submission's disk: `# DISK_PROJECTION_GB: N` header in the sbatch script wins
     (used for cache builds); else for train.sbatch, per-fold footprint by backbone family x fold
     count from --array; else a conservative 5GB with a warning.
  2. Sums projected disk of every still-alive job recorded in artifacts/disk_ledger.json
     (jobs launched via this wrapper; ledger entries for finished jobs drop out on their own).
     Live jobs launched some other way are counted at a conservative 5GB per array-job id.
  3. Refuses (exit 2) if sum > free quota - headroom, unless --force.
  4. Sets --job-name to the config's `experiment:` field for train jobs (unique names make logs
     attributable and name-based cancels less catastrophic — but still cancel by ID only, see
     scripts/safe_scancel.py).
  5. Submits, appends to the ledger, prints the job id.

--reuses JOBID: this submission is a resume/chain link that rewrites the SAME checkpoint files as
JOBID (e.g. an afterany chain) — projects 0GB so the same disk isn't counted twice.

Accounting for live jobs: a job's projection is reduced by what it has ALREADY written (its
`# DISK_PROJECTION_PATH:` header dir for cache builds, artifacts/<experiment>/ for train jobs), since
the free-quota reading already excludes those bytes — counting them again made this refuse a
launch while a 160GB cache build was 90% done (2026-09-18 false positive). Still conservative
(never below 0, jobs of unknown location count in full). --force exists for the deliberate exception.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HEADROOM_DEFAULT_GB = 30.0
UNKNOWN_JOB_GB = 5.0
# Per-fold disk (checkpoint.pt + best.pt + eval parquets), measured 2026-09-18 from real files:
# ViT-B checkpoint 1.76GB + best 0.88GB; ViT-S checkpoint ~0.44GB (2x its 0.22GB best) + 0.22GB.
PER_FOLD_GB = {"vit_base": 2.7, "vit_small": 0.8}
PER_FOLD_GB_OTHER = 3.0


def quota_free_gb() -> float:
    out = subprocess.check_output(
        ["beegfs-ctl", "--getquota", "--uid", os.environ.get("USER", ""), "--csv"], text=True
    ).strip().splitlines()[-1]
    parts = out.split(",")
    used, hard = float(parts[2]), float(parts[3])
    return (hard - used) / 1024 ** 3


def written_gb(root: Path, entry: dict) -> float:
    """Bytes a live job has already put on disk, in GB, so its projection isn't double counted."""
    path = None
    script = root / "code" / "slurm" / str(entry.get("script", ""))
    if not script.exists():
        script = Path("slurm") / str(entry.get("script", ""))
    if script.exists():
        m = re.search(r"^#\s*DISK_PROJECTION_PATH:\s*(\S+)", script.read_text(), re.MULTILINE)
        if m:
            path = root / m.group(1)
    if path is None and entry.get("experiment") and str(entry.get("script", "")).startswith("train"):
        path = root / "artifacts" / entry["experiment"]
    if path is None or not path.exists():
        return 0.0
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True, timeout=120)
        return int(out.split()[0]) / 1024 ** 3
    except Exception:  # noqa: BLE001 - unmeasurable -> count in full (conservative)
        return 0.0


def alive_array_ids() -> set:
    out = subprocess.check_output(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%F"], text=True)
    return {line.strip() for line in out.splitlines() if line.strip()}


def array_size(sbatch_args: list, script: Path) -> int:
    spec = None
    for a in sbatch_args:
        m = re.match(r"--array=(.+)", a)
        if m:
            spec = m.group(1)
    if spec is None and script.exists():
        m = re.search(r"^#SBATCH\s+--array=(\S+)", script.read_text(), re.MULTILINE)
        spec = m.group(1) if m else None
    if spec is None:
        return 1
    spec = spec.split("%")[0]
    total = 0
    for chunk in spec.split(","):
        if "-" in chunk:
            lo, hi = chunk.split("-")[:2]
            total += int(hi) - int(lo) + 1
        else:
            total += 1
    return total


def project_gb(script: Path, script_args: list, sbatch_args: list, reuses: str | None):
    """Returns (projected_gb, note, experiment_name_or_None)."""
    experiment = None
    cfg_path = next((a for a in script_args if a.endswith((".yaml", ".yml"))), None)
    if cfg_path and Path(cfg_path).exists():
        import yaml
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        experiment = cfg.get("experiment")
    if reuses:
        return 0.0, f"reuses disk of job {reuses} (chain/resume link)", experiment
    if script.exists():
        m = re.search(r"^#\s*DISK_PROJECTION_GB:\s*([0-9.]+)", script.read_text(), re.MULTILINE)
        if m:
            return float(m.group(1)), "from DISK_PROJECTION_GB header", experiment
    if script.name.startswith("train") and cfg_path and Path(cfg_path).exists():
        import yaml
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        backbone = str(cfg.get("model", {}).get("backbone", ""))
        family = next((k for k in PER_FOLD_GB if k in backbone), None)
        per_fold = PER_FOLD_GB.get(family, PER_FOLD_GB_OTHER)
        n = array_size(sbatch_args, script)
        return per_fold * n, f"{n} fold(s) x {per_fold}GB ({family or 'other/unknown backbone'})", experiment
    return UNKNOWN_JOB_GB, "UNKNOWN script type — conservative default", experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuses", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headroom-gb", type=float, default=HEADROOM_DEFAULT_GB)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    rest = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    if not rest:
        parser.error("nothing to submit — pass sbatch args after `--`")

    script_idx = next((i for i, a in enumerate(rest) if not a.startswith("-") and a.endswith(".sbatch")), None)
    if script_idx is None:
        parser.error("no .sbatch script found in the sbatch args")
    sbatch_args, script, script_args = rest[:script_idx], Path(rest[script_idx]), rest[script_idx + 1:]

    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    ledger_path = root / "artifacts" / "disk_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []

    this_gb, note, experiment = project_gb(script, script_args, sbatch_args, args.reuses)
    free_gb = quota_free_gb()
    alive = alive_array_ids()

    rows, live_total = [], 0.0
    known_ids = set()
    for e in ledger:
        if e["job_id"] in alive:
            known_ids.add(e["job_id"])
            written = written_gb(root, e)
            remaining = max(0.0, e["projected_gb"] - written)
            live_total += remaining
            rows.append((e["job_id"], e.get("experiment") or e.get("script"), remaining,
                         f"ledger; {written:.1f}GB of {e['projected_gb']:.1f}GB already on disk"))
    for jid in sorted(alive - known_ids):
        live_total += UNKNOWN_JOB_GB
        rows.append((jid, "(not launched via safe_sbatch)", UNKNOWN_JOB_GB, "conservative default"))

    limit = free_gb - args.headroom_gb
    total = live_total + this_gb
    print(f"free quota:            {free_gb:8.1f} GB")
    print(f"required headroom:     {args.headroom_gb:8.1f} GB")
    print(f"budget for projections:{limit:8.1f} GB")
    for jid, name, gb, src in rows:
        print(f"  live  {jid:>10}  {gb:7.1f} GB  {name}  [{src}]")
    print(f"  THIS  {'':>10}  {this_gb:7.1f} GB  {experiment or script.name}  [{note}]")
    print(f"projected total:       {total:8.1f} GB  ->  {'OK' if total <= limit else 'EXCEEDS BUDGET'}")

    if total > limit and not args.force:
        print("REFUSING to submit (pass --force to override deliberately).", file=sys.stderr)
        return 2
    if args.dry_run:
        print("dry-run: not submitting.")
        return 0

    cmd = ["sbatch", "--parsable"]
    if experiment and script.name.startswith("train") and not any(a.startswith("--job-name") for a in sbatch_args):
        cmd.append(f"--job-name={experiment}")
    cmd += sbatch_args + [str(script)] + script_args
    job_id = subprocess.check_output(cmd, text=True).strip().split(";")[0]
    ledger.append({"job_id": job_id, "script": script.name, "experiment": experiment,
                   "projected_gb": this_gb, "note": note, "submitted": time.strftime("%Y-%m-%d %H:%M:%S")})
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=1))
    print(f"submitted job {job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
