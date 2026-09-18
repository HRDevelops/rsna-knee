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

"""Cancel SLURM jobs by explicit job ID only, listing exactly what will be cancelled first
(2026-09-18: `scancel --name=rsna_train` matched every job launched from train.sbatch — including
an experiment we meant to keep — because they all share one job name).

    python scripts/safe_scancel.py 25344 25353_2          # LIST only — cancels nothing
    python scripts/safe_scancel.py --confirm 25344 25353_2  # cancels exactly those, after listing

Rejects anything that is not a plain job id / array id / array-task id (no names, no wildcards, no
flags, no `--user`). An array id (`25344`) cancels the whole array; `25344_2` one task.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ID_RE = re.compile(r"^\d+(_\d+)?$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm", action="store_true", help="actually cancel (default: list only)")
    parser.add_argument("job_ids", nargs="+")
    args = parser.parse_args()

    bad = [j for j in args.job_ids if not ID_RE.match(j)]
    if bad:
        print(f"REFUSING: not explicit job ids: {bad}. Job ids only (e.g. 25344 or 25344_2) — never "
              f"names, patterns, or --user.", file=sys.stderr)
        return 2

    user = os.environ.get("USER", "")
    to_cancel = []
    for jid in args.job_ids:
        out = subprocess.run(["squeue", "-h", "-u", user, "-j", jid.split("_")[0], "-o", "%i|%j|%T|%M"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        rows = [r for r in out if r]
        if "_" in jid:
            rows = [r for r in rows if r.split("|")[0].split("_")[0] == jid.split("_")[0]
                    and (r.split("|")[0] == jid or "[" in r.split("|")[0])]
        if not rows:
            print(f"  {jid}: not found among your queued/running jobs (already finished?) — skipping")
            continue
        for r in rows:
            print(f"  WILL CANCEL  {jid:>12}  <- {r}")
        to_cancel.append(jid)

    if not to_cancel:
        print("nothing to cancel.")
        return 0
    if not args.confirm:
        print("list only — re-run with --confirm to cancel exactly the jobs above.")
        return 0
    for jid in to_cancel:
        subprocess.check_call(["scancel", jid])
        print(f"cancelled {jid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
