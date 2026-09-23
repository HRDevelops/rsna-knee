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

"""Shared final-cell tail for the submission notebooks: re-read + fsync submission.csv, then os._exit(0).

Added 2026-09-23 after probe #4 (blend018) hung 9h (v2) and 10h (v3) AFTER every cell had finished --
v3 already carried the thread-shutdown fix proven on HPC, so the remaining hypothesis is CUDA-context /
interpreter teardown (the blend is the only notebook holding 20 models on 2 GPUs). os._exit(0) skips
all Python-level cleanup (thread joins, atexit, CUDA destructors). It does NOT skip the OS releasing
the process's GPU file handles, so a hang inside the driver itself would still reproduce -- which is
itself diagnostic. Expects `os`, `pd`, `OUT_PATH`, `test_df`, `TARGET_COLUMNS` in the notebook scope.
"""

HARD_EXIT_TAIL = '''
# ---- Hard exit (2026-09-23) ----------------------------------------------------------------------
# Probe #4 v2/v3 hung 9-10h AFTER all cell code had finished (submission.csv and __results__.html were
# both written), despite the HPC-proven thread-shutdown fix. os._exit(0) bypasses ALL remaining cleanup
# (thread joins, atexit, CUDA context teardown). It must stay the LAST statement of the notebook: the
# file has already been written and closed above; here it is re-read, validated and fsync'd first.
import sys as _sys
_final = pd.read_csv(OUT_PATH)
assert len(_final) == len(test_df), f"re-read row count mismatch: {len(_final)} vs {len(test_df)}"
assert list(_final.columns) == ["StudyInstanceUID"] + TARGET_COLUMNS, "re-read header mismatch"
assert not _final[TARGET_COLUMNS].isna().any().any(), "re-read NaNs"
try:  # append-mode handle (writes nothing): fsync on a read-only fd raises EBADF on some platforms
    with open(OUT_PATH, "ab") as _f:
        os.fsync(_f.fileno())
except OSError as _e:
    print(f"[hard-exit] fsync skipped ({_e}); file was already closed by to_csv", flush=True)
print(_final.head().to_string())
print(f"[hard-exit] {OUT_PATH} verified on disk ({OUT_PATH.stat().st_size} bytes, {len(_final)} rows); "
      f"calling os._exit(0).", flush=True)
_sys.stdout.flush()
_sys.stderr.flush()
os._exit(0)
'''
