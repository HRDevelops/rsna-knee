"""LLM report-to-label extraction via vLLM offline batch inference (EXECUTION_PLAN Phase 1.3).

Runs one instruction-tuned LLM (Qwen2.5-72B-Instruct[-AWQ], Llama-3.3-70B-Instruct, etc.) over
every study's free-text report and extracts a structured judgement for all 12 targets, following
the severity-thresholding rubric in configs/rubric_v1.yaml. Output is a WIDE parquet, one row per
study, with per-target present/probability/severity_phrase/evidence_quote columns plus the raw
model response (kept for the Phase 1.4 error ledger).

Designed to survive an 8-12h SLURM walltime: writes checkpoints every --chunk-size studies and
skips already-labeled studies on resume, so a preempted/timed-out job just re-run picks up where
it left off. A single bad report must never kill the run — parse failures are logged, not raised.

Usage (see slurm/label_gen.sbatch):
    python scripts/label_reports.py \
        --model Qwen/Qwen2.5-72B-Instruct-AWQ \
        --reports $RSNA_ROOT/data/raw/train.csv \
        --rubric configs/rubric_v1.yaml \
        --out $RSNA_ROOT/artifacts/labels/qwen2.5-72b-awq.parquet

Smoke test on a handful of studies first:
    python scripts/label_reports.py --model ... --reports ... --rubric ... --out /tmp/smoke.parquet \
        --max-studies 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


def load_rubric(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        rubric = yaml.safe_load(f)
    return rubric


def build_system_prompt(rubric: dict) -> str:
    return rubric["system_prompt"].replace(
        "${calibration_instruction}", rubric["calibration_instruction"]
    )


def build_few_shot_block(rubric: dict) -> str:
    blocks = []
    for i, ex in enumerate(rubric.get("few_shot_examples", []), start=1):
        blocks.append(
            f"--- Example {i} ---\nReport:\n{ex['report'].strip()}\n\n"
            f"Correct reasoning (do not copy verbatim, this is guidance not a template):\n"
            f"{ex['expected_pattern'].strip()}\n"
        )
    return "\n".join(blocks)


def build_user_message(rubric: dict, few_shot_block: str, report_text: str) -> str:
    columns = [t["column"] for t in rubric["targets"]]
    columns_json_hint = ", ".join(f'"{c}"' for c in columns)
    return (
        f"{few_shot_block}\n\n"
        f"--- Now extract for THIS report ---\n"
        f"Report:\n{report_text.strip()}\n\n"
        f"Respond with ONLY a JSON object whose top-level keys are exactly: {columns_json_hint}. "
        f"No markdown fences, no explanation outside the JSON."
    )


def build_prompts(rubric: dict, reports: pd.DataFrame, tokenizer) -> list[str]:
    system_prompt = build_system_prompt(rubric)
    few_shot_block = build_few_shot_block(rubric)
    prompts = []
    for report_text in reports["Report"]:
        user_msg = build_user_message(rubric, few_shot_block, report_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_VALID_PRESENT = {"yes", "no", "uncertain"}
_UNCERTAIN_DEFAULT_PROB = 0.275  # midpoint of the rubric's negative-leaning uncertain band


def extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJ_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


_IMPUTED_FAILURE_CONFIDENCE_WEIGHT = 0.1  # rubric v2 strategy review: residual parse failures
                                            # get label priors + this low confidence weight, not
                                            # discarded from training entirely.


def coerce_target_value(raw: dict | None) -> dict:
    if not isinstance(raw, dict):
        return {
            "present": "uncertain",
            "probability": _UNCERTAIN_DEFAULT_PROB,
            "severity_phrase": "",
            "evidence_quote": "",
            "imputed": True,
            "confidence_weight": _IMPUTED_FAILURE_CONFIDENCE_WEIGHT,
            "source": "imputed_failure",
        }
    present = str(raw.get("present", "uncertain")).strip().lower()
    if present not in _VALID_PRESENT:
        present = "uncertain"
    try:
        prob = float(raw.get("probability", _UNCERTAIN_DEFAULT_PROB))
    except (TypeError, ValueError):
        prob = _UNCERTAIN_DEFAULT_PROB
    prob = min(max(prob, 0.0), 1.0)
    return {
        "present": present,
        "probability": prob,
        "severity_phrase": str(raw.get("severity_phrase", "") or ""),
        "evidence_quote": str(raw.get("evidence_quote", "") or ""),
        "imputed": False,
        "confidence_weight": 1.0,
        "source": "llm",
    }


def is_silent(coerced: dict) -> bool:
    """True when the LLM said nothing about this target at all (not an explicit negative
    statement, just absence of any mention) — the signal conditional_priors keys off."""
    return (
        coerced["present"] == "no"
        and coerced["severity_phrase"] == ""
        and coerced["evidence_quote"] == ""
        and not coerced["imputed"]
    )


def apply_conditional_priors(rubric: dict, row: dict) -> None:
    """Deterministic cross-field overrides (rubric v2) applied after parsing, in-place on `row`.
    Only fires when the target's own result looks silent (see is_silent) — an explicit negative
    statement from the LLM is left alone, this only overrides "found nothing to say" cases."""
    priors = rubric.get("conditional_priors", {})
    for target_key, cfg in priors.items():
        target_col = next(
            (t["column"] for t in rubric["targets"] if t["key"] == target_key), None
        )
        based_on_col = next(
            (t["column"] for t in rubric["targets"] if t["key"] == cfg["based_on_target"]), None
        )
        if target_col is None or based_on_col is None:
            continue

        target_state = {
            "present": row[f"{target_col}__present"],
            "severity_phrase": row[f"{target_col}__severity_phrase"],
            "evidence_quote": row[f"{target_col}__evidence_quote"],
            "imputed": row[f"{target_col}__imputed"],
        }
        if not is_silent(target_state):
            continue

        based_on_prob = row[f"{based_on_col}__probability"]
        for tier in cfg["tiers"]:
            if based_on_prob >= tier["effusion_probability_gte"]:
                row[f"{target_col}__probability"] = tier["prior_probability"]
                row[f"{target_col}__confidence_weight"] = cfg["confidence_weight"]
                row[f"{target_col}__source"] = cfg["source_tag"]
                break


def parse_response(rubric: dict, study_uid: str, raw_text: str) -> dict:
    columns = [t["column"] for t in rubric["targets"]]
    parsed = extract_json(raw_text)
    row = {"study_uid": study_uid, "raw_response": raw_text, "parse_ok": parsed is not None}
    for col in columns:
        target_raw = parsed.get(col) if isinstance(parsed, dict) else None
        coerced = coerce_target_value(target_raw)
        row[f"{col}__present"] = coerced["present"]
        row[f"{col}__probability"] = coerced["probability"]
        row[f"{col}__severity_phrase"] = coerced["severity_phrase"]
        row[f"{col}__evidence_quote"] = coerced["evidence_quote"]
        row[f"{col}__imputed"] = coerced["imputed"]
        row[f"{col}__confidence_weight"] = coerced["confidence_weight"]
        row[f"{col}__source"] = coerced["source"]
    if "conditional_priors" in rubric and parsed is not None:
        apply_conditional_priors(rubric, row)
    return row


def load_checkpoint(out_path: Path) -> pd.DataFrame | None:
    if out_path.exists():
        return pd.read_parquet(out_path)
    return None


def run(args: argparse.Namespace) -> None:
    rubric = load_rubric(args.rubric)
    reports_df = pd.read_csv(args.reports)
    reports_df = reports_df[["StudyInstanceUID", "Report"]].rename(
        columns={"StudyInstanceUID": "study_uid", "Report": "Report"}
    )
    if args.max_studies is not None:
        reports_df = reports_df.head(args.max_studies)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_checkpoint(out_path)
    done_uids = set(existing["study_uid"]) if existing is not None else set()
    todo_df = reports_df[~reports_df["study_uid"].isin(done_uids)].reset_index(drop=True)
    print(f"{len(done_uids)} studies already labeled, {len(todo_df)} remaining "
          f"(of {len(reports_df)} total).", file=sys.stderr)
    if todo_df.empty:
        print("Nothing to do.", file=sys.stderr)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    quantization = "awq" if "awq" in args.model.lower() else None
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=quantization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    gen_cfg = rubric.get("generation", {})
    sampling_params = SamplingParams(
        temperature=gen_cfg.get("temperature", 0.0),
        max_tokens=gen_cfg.get("max_tokens", 1400),
    )
    retry_sampling_params = SamplingParams(
        temperature=gen_cfg.get("temperature", 0.0),
        max_tokens=gen_cfg.get("retry_max_tokens", gen_cfg.get("max_tokens", 1400)),
    )
    max_retries = gen_cfg.get("max_retries", 1) if gen_cfg.get("retry_on_parse_failure", True) else 0

    all_rows: list[dict] = [] if existing is None else existing.to_dict("records")
    chunk_size = args.chunk_size

    for start in range(0, len(todo_df), chunk_size):
        chunk = todo_df.iloc[start:start + chunk_size].reset_index(drop=True)
        t0 = time.time()
        prompts = build_prompts(rubric, chunk, tokenizer)
        outputs = llm.generate(prompts, sampling_params)
        texts = [o.outputs[0].text for o in outputs]

        chunk_rows = []
        n_parse_fail = 0
        for study_uid, raw_text in zip(chunk["study_uid"], texts):
            row = parse_response(rubric, study_uid, raw_text)
            if not row["parse_ok"]:
                n_parse_fail += 1
            chunk_rows.append(row)

        # Retry parse failures once with a stricter reminder appended to the same prompt.
        for retry_i in range(max_retries):
            fail_idx = [i for i, r in enumerate(chunk_rows) if not r["parse_ok"]]
            if not fail_idx:
                break
            retry_prompts = [prompts[i] + "\n\nREMINDER: start your response immediately with "
                              "\"{\" — no preamble text, no markdown fences, output ONLY valid JSON."
                              for i in fail_idx]
            retry_outputs = llm.generate(retry_prompts, retry_sampling_params)
            for i, o in zip(fail_idx, retry_outputs):
                raw_text = o.outputs[0].text
                chunk_rows[i] = parse_response(rubric, chunk["study_uid"].iloc[i], raw_text)

        all_rows.extend(chunk_rows)
        pd.DataFrame(all_rows).to_parquet(out_path, index=False)
        elapsed = time.time() - t0
        n_still_failed = sum(1 for r in chunk_rows if not r["parse_ok"])
        print(f"chunk {start}-{start + len(chunk)}: {elapsed:.1f}s, "
              f"{n_parse_fail} initial parse failures, {n_still_failed} unresolved after retry. "
              f"Checkpoint written to {out_path} ({len(all_rows)} studies total).", file=sys.stderr)

    print(f"Done. {len(all_rows)} studies labeled -> {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--reports", type=Path, required=True, help="train.csv (or test.csv) path")
    parser.add_argument("--rubric", type=Path, required=True, help="configs/rubric_v1.yaml")
    parser.add_argument("--out", type=Path, required=True, help="Output parquet path")
    parser.add_argument("--max-studies", type=int, default=None, help="Limit for a smoke test")
    parser.add_argument("--chunk-size", type=int, default=200,
                         help="Studies per checkpoint (resumable)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
