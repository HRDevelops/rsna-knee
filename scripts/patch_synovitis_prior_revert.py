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

"""One-off correction: revert the Synovitis effusion-tiered conditional prior to a flat 0.35.

2026-09-07 strategy review, part B step 4. The conditional prior was applied POST-HOC IN CODE
(scripts/label_reports.py's apply_conditional_priors), not baked into the LLM prompt — so this
patches the already-generated parquet directly rather than re-running the LLM. Only rows tagged
source="conditional_prior" (i.e., the LLM found nothing to say about Synovitis at all) are
touched; any row where the LLM gave an actual judgement is left untouched.

Usage:
    python scripts/patch_synovitis_prior_revert.py \
        --labels artifacts/labels/qwen2.5-72b-instruct-awq_v2.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FLAT_PRIOR = 0.35
CONFIDENCE_WEIGHT = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.labels)
    mask = df["Synovitis__source"] == "conditional_prior"
    n_before = df.loc[mask, "Synovitis__probability"].value_counts()
    print(f"{mask.sum()} rows tagged conditional_prior. Value counts before:\n{n_before}")

    df.loc[mask, "Synovitis__probability"] = FLAT_PRIOR
    df.loc[mask, "Synovitis__confidence_weight"] = CONFIDENCE_WEIGHT

    df.to_parquet(args.labels, index=False)
    print(f"Reverted {mask.sum()} rows to flat probability={FLAT_PRIOR}, "
          f"confidence_weight={CONFIDENCE_WEIGHT}. Wrote back to {args.labels}")


if __name__ == "__main__":
    main()
