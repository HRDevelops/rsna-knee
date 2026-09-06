"""Decode-audit: census DICOM headers against the 86-tag metadata allowlist.

Answers the one question that decides the laterality route (CLAUDE.md §3, §5):
did Laterality (0020,0060) / ImageLaterality (0020,0062) / BodyPartExamined survive
the RSNA metadata strip? If yes, side detection is free. If no, we need the
report-supervised side classifier (EXECUTION_PLAN Phase 2.3).

Reads one representative DICOM per series (stop_before_pixels for speed; a small
subsample is fully decoded to confirm pylibjpeg/gdcm handle the transfer syntaxes
present and to record image shape). Never raises on a single bad file — per-file
errors are logged and skipped, since a crash here would blow the 1 GPU-hour budget
for nothing.

Usage (on HPC, after data download):
    python scripts/decode_audit.py \
        --raw "$RSNA_ROOT/data/raw" \
        --out-data artifacts/decode_audit.parquet \
        --out-report docs/DECODE_AUDIT.md \
        --workers 16

    # quick smoke test before the full run:
    python scripts/decode_audit.py --raw "$RSNA_ROOT/data/raw" --sample-studies 50
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

# Tags whose presence directly decides the side-detection route (EXECUTION_PLAN 0.5).
LATERALITY_TAGS = [
    "Laterality",           # (0020,0060)
    "ImageLaterality",      # (0020,0062)
    "BodyPartExamined",     # (0018,0015)
    "PatientPosition",      # (0018,5100)
]

# Other tags worth calling out explicitly even if not decisive.
WATCH_TAGS = [
    "StudyDescription",
    "SeriesDescription",
    "ViewPosition",
    "ProtocolName",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "SliceThickness",
    "Rows",
    "Columns",
]


@dataclass
class SeriesAudit:
    study_uid: str
    series_uid: str
    sop_uid: str = ""
    n_files: int = 0
    ok: bool = False
    error: str = ""
    transfer_syntax_uid: str = ""
    transfer_syntax_name: str = ""
    rows: int | None = None
    columns: int | None = None
    pixel_spacing: str = ""
    slice_thickness: str = ""
    image_orientation_patient: str = ""
    present_tags: list[str] = field(default_factory=list)
    laterality_values: dict[str, str] = field(default_factory=dict)
    pixel_decode_ok: bool | None = None  # only set for the pixel-decode subsample


def find_series_dirs(raw_root: Path) -> list[Path]:
    train_series_dir = raw_root / "train_series"
    if not train_series_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {train_series_dir} to exist (StudyUID/SeriesUID/*.dcm layout)"
        )
    series_dirs = [p for p in train_series_dir.glob("*/*") if p.is_dir()]
    return sorted(series_dirs)


def audit_one_series(series_dir: Path, decode_pixels: bool) -> SeriesAudit:
    study_uid = series_dir.parent.name
    series_uid = series_dir.name
    audit = SeriesAudit(study_uid=study_uid, series_uid=series_uid)

    dcm_files = sorted(series_dir.glob("*.dcm"))
    audit.n_files = len(dcm_files)
    if not dcm_files:
        audit.error = "no .dcm files in series dir"
        return audit

    rep_file = dcm_files[len(dcm_files) // 2]  # middle slice, avoids scout/localizer bias
    audit.sop_uid = rep_file.stem

    try:
        ds = pydicom.dcmread(str(rep_file), stop_before_pixels=not decode_pixels, force=True)
    except (InvalidDicomError, Exception) as exc:  # noqa: BLE001 - must never crash the audit
        audit.error = f"{type(exc).__name__}: {exc}"
        return audit

    audit.ok = True
    try:
        ts = ds.file_meta.TransferSyntaxUID
        audit.transfer_syntax_uid = str(ts)
        audit.transfer_syntax_name = str(ts.name) if hasattr(ts, "name") else ""
    except Exception:  # noqa: BLE001
        pass

    audit.rows = int(ds.Rows) if "Rows" in ds else None
    audit.columns = int(ds.Columns) if "Columns" in ds else None
    audit.pixel_spacing = str(ds.get("PixelSpacing", ""))
    audit.slice_thickness = str(ds.get("SliceThickness", ""))
    audit.image_orientation_patient = str(ds.get("ImageOrientationPatient", ""))

    audit.present_tags = sorted({elem.keyword for elem in ds if elem.keyword})
    for tag in LATERALITY_TAGS:
        if tag in ds:
            audit.laterality_values[tag] = str(ds.get(tag, ""))

    if decode_pixels:
        try:
            _ = ds.pixel_array  # forces the actual JPEG/JPEG2000/RLE decode path
            audit.pixel_decode_ok = True
        except Exception as exc:  # noqa: BLE001
            audit.pixel_decode_ok = False
            audit.error = f"pixel decode failed: {type(exc).__name__}: {exc}"

    return audit


def run_audit(
    raw_root: Path,
    sample_studies: int | None,
    pixel_decode_sample: int,
    workers: int,
) -> pd.DataFrame:
    series_dirs = find_series_dirs(raw_root)
    if sample_studies is not None:
        studies_seen: set[str] = set()
        limited = []
        for d in series_dirs:
            studies_seen.add(d.parent.name)
            if len(studies_seen) > sample_studies:
                break
            limited.append(d)
        series_dirs = limited

    print(f"Auditing {len(series_dirs)} series across "
          f"{len({d.parent.name for d in series_dirs})} studies...", file=sys.stderr)

    decode_flags = [False] * len(series_dirs)
    for i in range(min(pixel_decode_sample, len(series_dirs))):
        decode_flags[i] = True  # deterministic subsample (first N) is fine here

    results: list[SeriesAudit] = []
    errors: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(audit_one_series, d, flag): d
            for d, flag in zip(series_dirs, decode_flags)
        }
        done = 0
        for fut in as_completed(futures):
            d = futures[fut]
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(series_dirs)}", file=sys.stderr)
            try:
                audit = fut.result()
            except Exception as exc:  # noqa: BLE001 - a worker crash must not kill the run
                errors.append((str(d), f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
                continue
            results.append(audit)
            if audit.error:
                errors.append((str(d), audit.error))

    if errors:
        print(f"{len(errors)} series had errors (see error log column in output).",
              file=sys.stderr)

    rows = []
    for a in results:
        rows.append({
            "study_uid": a.study_uid,
            "series_uid": a.series_uid,
            "sop_uid": a.sop_uid,
            "n_files": a.n_files,
            "ok": a.ok,
            "error": a.error,
            "transfer_syntax_uid": a.transfer_syntax_uid,
            "transfer_syntax_name": a.transfer_syntax_name,
            "rows": a.rows,
            "columns": a.columns,
            "pixel_spacing": a.pixel_spacing,
            "slice_thickness": a.slice_thickness,
            "image_orientation_patient": a.image_orientation_patient,
            "present_tags": "|".join(a.present_tags),
            **{f"tag_{t}": a.laterality_values.get(t, "") for t in LATERALITY_TAGS},
            "pixel_decode_ok": a.pixel_decode_ok,
        })
    return pd.DataFrame(rows)


def write_markdown_report(df: pd.DataFrame, out_path: Path) -> None:
    n_series = len(df)
    n_studies = df["study_uid"].nunique()
    n_ok = int(df["ok"].sum())

    tag_counter: Counter[str] = Counter()
    for tags in df.loc[df["ok"], "present_tags"]:
        tag_counter.update(t for t in tags.split("|") if t)

    lines = []
    lines.append("# Decode Audit — DICOM tag census\n")
    lines.append(f"Series scanned: **{n_series}** across **{n_studies}** studies "
                  f"({n_ok} decoded successfully, {n_series - n_ok} errors).\n")

    lines.append("## Laterality tags — decides the side-detection route\n")
    lines.append("| Tag | Present (n series) | Present (%) | Sample values |")
    lines.append("|---|---|---|---|")
    for tag in LATERALITY_TAGS:
        col = f"tag_{tag}"
        present = df.loc[df["ok"] & (df[col] != ""), col]
        pct = 100.0 * len(present) / max(n_ok, 1)
        samples = ", ".join(sorted(present.unique())[:5])
        lines.append(f"| {tag} | {len(present)} | {pct:.1f}% | {samples} |")
    lines.append("")

    any_laterality = any(
        len(df.loc[df["ok"] & (df[f"tag_{t}"] != ""), f"tag_{t}"]) / max(n_ok, 1) > 0.5
        for t in ("Laterality", "ImageLaterality")
    )
    if any_laterality:
        lines.append("**Verdict: Laterality/ImageLaterality survived** the metadata strip "
                      "for most series → use the DICOM tag directly, no side classifier needed "
                      "(still spot-check against report-stated side per CLAUDE.md risk register).\n")
    else:
        lines.append("**Verdict: Laterality/ImageLaterality did NOT survive** (or is present on "
                      "<50% of series) → side detection must fall back to "
                      "ImageOrientationPatient + anatomy or the report-supervised side classifier "
                      "(EXECUTION_PLAN Phase 2.3).\n")

    lines.append("## Full tag presence census (top 100 by frequency)\n")
    lines.append("| Tag | Present (n series) | Present (%) |")
    lines.append("|---|---|---|")
    for tag, count in tag_counter.most_common(100):
        pct = 100.0 * count / max(n_ok, 1)
        lines.append(f"| {tag} | {count} | {pct:.1f}% |")
    lines.append(f"\nTotal distinct tags observed: **{len(tag_counter)}** "
                  f"(CLAUDE.md states the allowlist is 86 tags — compare against this count).\n")

    lines.append("## Transfer syntax distribution\n")
    ts_counts = df.loc[df["ok"], "transfer_syntax_name"].value_counts()
    lines.append("| Transfer syntax | n series |")
    lines.append("|---|---|")
    for name, count in ts_counts.items():
        lines.append(f"| {name or '(unknown)'} | {count} |")
    lines.append("")

    decode_sub = df.loc[df["pixel_decode_ok"].notna()]
    if len(decode_sub):
        n_pixel_ok = int(decode_sub["pixel_decode_ok"].sum())
        lines.append(f"## Pixel decode subsample\n")
        lines.append(f"{n_pixel_ok}/{len(decode_sub)} sampled series decoded pixel data "
                      "successfully with the installed codecs (pylibjpeg/gdcm).\n")
        failures = decode_sub.loc[~decode_sub["pixel_decode_ok"].astype(bool)]
        if len(failures):
            lines.append("Failures (transfer syntax, error):\n")
            for _, row in failures.iterrows():
                lines.append(f"- `{row['transfer_syntax_name']}`: {row['error']}")
            lines.append("")

    lines.append("## Shape / spacing sanity\n")
    ok_df = df.loc[df["ok"]]
    lines.append(f"- Rows/Columns distinct values: "
                  f"{sorted(ok_df[['rows', 'columns']].drop_duplicates().itertuples(index=False))[:10]}")
    lines.append(f"- Distinct PixelSpacing values (top 10): "
                  f"{ok_df['pixel_spacing'].value_counts().head(10).to_dict()}")
    lines.append(f"- Distinct SliceThickness values (top 10): "
                  f"{ok_df['slice_thickness'].value_counts().head(10).to_dict()}")
    lines.append("")

    if (~df["ok"]).any():
        lines.append("## Errors (first 50)\n")
        for _, row in df.loc[~df["ok"]].head(50).iterrows():
            lines.append(f"- `{row['study_uid']}/{row['series_uid']}`: {row['error']}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True,
                         help="Path to data/raw containing train_series/")
    parser.add_argument("--out-data", type=Path, default=Path("artifacts/decode_audit.parquet"))
    parser.add_argument("--out-report", type=Path, default=Path("docs/DECODE_AUDIT.md"))
    parser.add_argument("--sample-studies", type=int, default=None,
                         help="Limit to first N studies for a quick smoke test")
    parser.add_argument("--pixel-decode-sample", type=int, default=200,
                         help="Number of series to fully decode (pixel data) to verify codecs")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    df = run_audit(
        raw_root=args.raw,
        sample_studies=args.sample_studies,
        pixel_decode_sample=args.pixel_decode_sample,
        workers=args.workers,
    )

    args.out_data.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_data, index=False)
    print(f"Wrote {len(df)} series records to {args.out_data}", file=sys.stderr)

    write_markdown_report(df, args.out_report)


if __name__ == "__main__":
    main()
