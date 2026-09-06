# Decode Audit — DICOM tag census

Series scanned: **24371** across **4407** studies (24371 decoded successfully, 0 errors).

## Laterality tags — decides the side-detection route

| Tag | Present (n series) | Present (%) | Sample values |
|---|---|---|---|
| Laterality | 12004 | 49.3% | B, L, LEFT, R, RIGHT |
| ImageLaterality | 0 | 0.0% |  |
| BodyPartExamined | 22049 | 90.5% | ABDOMEN, ADRENAL, ANKLE, BRAIN, BRONCHUS |
| PatientPosition | 23165 | 95.1% | FFP, FFS |

**Verdict: Laterality/ImageLaterality did NOT survive** (or is present on <50% of series) → side detection must fall back to ImageOrientationPatient + anatomy or the report-supervised side classifier (EXECUTION_PLAN Phase 2.3).

### Reviewed findings (manual follow-up, 2026-09-06 — read this, the auto-verdict above undersells it)

The per-series number above hides the decision-relevant number. Laterality is a per-**study**
property, so what matters is: does at least one series in the study carry a usable value?

- **Exactly 50.0% of studies (2204/4407) have >=1 series with a non-blank Laterality tag**
  (values seen: blank 50.7% of series, `R` 24.6%, `L` 24.3%, `RIGHT`/`LEFT` spelled out ~0.4%,
  `B` (bilateral?) 4 series). This is a clean ~50/50 split, not a sparse-but-recoverable pattern —
  strongly suggests a **site/scanner-dependent metadata practice** (some sites populate it, others
  never do), which is also a useful signal for the report-cluster/site-proxy CV grouping
  (CLAUDE.md §7, EXECUTION_PLAN 2.4).
- **25 of the 2204 tag-covered studies (~1.1%) have genuinely conflicting L/R values across their
  own series** (e.g. one series tagged `R`, another tagged `L` in the same study). This needs
  explicit handling before trusting the tag: majority-vote across series within a study, and
  hand-review the disagreeing studies (candidates: truly bilateral exams, mislabeled series, or a
  scanner protocol quirk) rather than silently picking one value.
- **Recommendation — hybrid side detection, not a binary tag-vs-classifier choice:**
  1. For the 50% of studies with a tag: majority-vote across series, flag the 25 conflicting
     studies for manual review (or default to running both flips + label-swap-averaged inference
     per CLAUDE.md's risk-register mitigation for exactly this uncertain case).
  2. For the other 50%: use the report-supervised side classifier already planned in
     EXECUTION_PLAN Phase 2.3 (reports give free side labels in every language per CLAUDE.md §5).
  3. Bonus: the 50% WITH a DICOM-tag side label is a ready-made validation set for that
     classifier, in addition to report-stated-side supervision — train/validate the classifier
     on this half, then trust it on the untagged half.

`BodyPartExamined` also needs one caveat: the sample-values column above shows an alphabetically-
sorted slice (ABDOMEN/ADRENAL/ANKLE/...), which is misleading on its own — the real distribution
(`artifacts/decode_audit.parquet`, column `tag_BodyPartExamined`) is dominated by `KNEE` (19360,
79.4%), blank (2322, 9.5%), and `EXTREMITY` (2265, 9.3%); everything else (WRIST, ANKLE, HEAD,
BRAIN, LIVER, ...) is under 0.5% each and reads as inherited PACS metadata noise from the source
archives, not evidence of wrong-body-part images in this competition's data — not a blocker, but
worth a spot-check of those ~150 oddly-labeled series later if BodyPartExamined is ever used as a
QC filter.

## Full tag presence census (top 100 by frequency)

| Tag | Present (n series) | Present (%) |
|---|---|---|
| AcquisitionNumber | 24371 | 100.0% |
| BitsAllocated | 24371 | 100.0% |
| BitsStored | 24371 | 100.0% |
| Columns | 24371 | 100.0% |
| FrameOfReferenceUID | 24371 | 100.0% |
| HighBit | 24371 | 100.0% |
| ImageOrientationPatient | 24371 | 100.0% |
| ImagePositionPatient | 24371 | 100.0% |
| ImageType | 24371 | 100.0% |
| InstanceNumber | 24371 | 100.0% |
| Manufacturer | 24371 | 100.0% |
| ManufacturerModelName | 24371 | 100.0% |
| Modality | 24371 | 100.0% |
| PatientID | 24371 | 100.0% |
| PhotometricInterpretation | 24371 | 100.0% |
| PixelRepresentation | 24371 | 100.0% |
| PixelSpacing | 24371 | 100.0% |
| Rows | 24371 | 100.0% |
| SOPClassUID | 24371 | 100.0% |
| SOPInstanceUID | 24371 | 100.0% |
| SamplesPerPixel | 24371 | 100.0% |
| SeriesInstanceUID | 24371 | 100.0% |
| SeriesNumber | 24371 | 100.0% |
| SliceLocation | 24371 | 100.0% |
| SliceThickness | 24371 | 100.0% |
| StudyInstanceUID | 24371 | 100.0% |
| WindowCenter | 24371 | 100.0% |
| WindowWidth | 24371 | 100.0% |
| EchoTime | 23165 | 95.1% |
| EchoTrainLength | 23165 | 95.1% |
| FlipAngle | 23165 | 95.1% |
| MRAcquisitionType | 23165 | 95.1% |
| MagneticFieldStrength | 23165 | 95.1% |
| NumberOfAverages | 23165 | 95.1% |
| PatientPosition | 23165 | 95.1% |
| PatientSex | 23165 | 95.1% |
| RepetitionTime | 23165 | 95.1% |
| ScanOptions | 23165 | 95.1% |
| ScanningSequence | 23165 | 95.1% |
| SequenceVariant | 23165 | 95.1% |
| SoftwareVersions | 23165 | 95.1% |
| AcquisitionMatrix | 23163 | 95.0% |
| InPlanePhaseEncodingDirection | 23104 | 94.8% |
| PixelBandwidth | 23035 | 94.5% |
| ImagingFrequency | 22978 | 94.3% |
| PercentPhaseFieldOfView | 22978 | 94.3% |
| SeriesDescription | 22928 | 94.1% |
| SpacingBetweenSlices | 22792 | 93.5% |
| BodyPartExamined | 22462 | 92.2% |
| PercentSampling | 22291 | 91.5% |
| EchoNumbers | 20669 | 84.8% |
| Laterality | 19271 | 79.1% |
| ReceiveCoilName | 18705 | 76.8% |
| NumberOfPhaseEncodingSteps | 18614 | 76.4% |
| TransmitCoilName | 13829 | 56.7% |
| NumberOfTemporalPositions | 13593 | 55.8% |
| VariableFlipAngleFlag | 12176 | 50.0% |
| SequenceName | 11412 | 46.8% |
| RescaleIntercept | 9907 | 40.7% |
| RescaleSlope | 9907 | 40.7% |
| ContrastBolusAgent | 8596 | 35.3% |
| DiffusionGradientOrientation | 7656 | 31.4% |
| DiffusionBValue | 7611 | 31.2% |
| InversionTime | 4858 | 19.9% |
| NumberOfFrames | 4402 | 18.1% |
| InStackPositionNumber | 4236 | 17.4% |
| StackID | 4236 | 17.4% |
| RescaleType | 4184 | 17.2% |
| PlanarConfiguration | 3062 | 12.6% |
| ParallelAcquisition | 1301 | 5.3% |
| ParallelAcquisitionTechnique | 1301 | 5.3% |
| ParallelReductionFactorInPlane | 994 | 4.1% |
| TemporalPositionIndex | 687 | 2.8% |
| DiffusionDirectionality | 666 | 2.7% |
| MRAcquisitionPhaseEncodingStepsInPlane | 666 | 2.7% |
| MRAcquisitionPhaseEncodingStepsOutOfPlane | 666 | 2.7% |
| ParallelReductionFactorOutOfPlane | 666 | 2.7% |
| PartialFourier | 666 | 2.7% |
| PartialFourierDirection | 666 | 2.7% |
| PulseSequenceName | 666 | 2.7% |
| RFEchoTrainLength | 666 | 2.7% |
| ReceiveCoilType | 666 | 2.7% |
| TransmitCoilType | 666 | 2.7% |
| PixelData | 300 | 1.2% |

Total distinct tags observed: **84** (CLAUDE.md states the allowlist is 86 tags — compare against this count).

## Transfer syntax distribution

| Transfer syntax | n series |
|---|---|
| Explicit VR Little Endian | 24371 |

## Pixel decode subsample

300/300 sampled series decoded pixel data successfully with the installed codecs (pylibjpeg/gdcm).

## Shape / spacing sanity

- Rows/Columns distinct values: [Pandas(rows=160, columns=160), Pandas(rows=192, columns=192), Pandas(rows=216, columns=256), Pandas(rows=224, columns=224), Pandas(rows=256, columns=204), Pandas(rows=256, columns=216), Pandas(rows=256, columns=232), Pandas(rows=256, columns=256), Pandas(rows=288, columns=288), Pandas(rows=300, columns=320)]
- Distinct PixelSpacing values (top 10): {'[0.3125, 0.3125]': 2388, '[0.25, 0.25]': 1993, '[0.416667, 0.416667]': 1405, '[0.332, 0.332]': 919, '[0.3516, 0.3516]': 820, '[0.5625, 0.5625]': 671, '[0.22727273404598, 0.22727273404598]': 560, '[0.5, 0.5]': 530, '[0.390625, 0.390625]': 400, '[0.1563, 0.1563]': 357}
- Distinct SliceThickness values (top 10): {'3': 13605, '4': 4099, '3.5': 1559, '2.5': 887, '2': 632, '3.0': 600, '3.5000': 469, '4.0000': 354, '3.0000': 315, '0.6': 263}
