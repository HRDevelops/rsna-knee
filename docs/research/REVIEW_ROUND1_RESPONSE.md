Thank you, this was a useful review. I checked every point against the sources before changing anything. Summary: I
accept 14 of your 16 points outright and 2 with an adjustment (#3 and the endpoint hierarchy), and I found 3 new data
problems that change Phase 0.
The revised plan is attached as `PLAN_v1_1.md`.

## Your points: verdict and evidence

| # | Point | Verdict | What I checked / changed |
|---|---|---|---|
| 1 | Re-keyed FrameOfReferenceUIDs mean the common frame is unproven | **Accept** | Wording changed to "candidate common coordinates". Phase 0.1 is now a blocking registration check with your gates (median ≤ 3 mm, p95 ≤ 5 mm, gross failures < 3% of pairs, MI peak at the DICOM transform) plus an injected 10 mm shift as a positive control. If it fails, a per-series rigid $T_s$ is stored in the cache. |
| 2 | Pixel→world must go through the preprocessing affine | **Accept (blocking)** | Our chain is: resample to 0.417 mm with `scipy.ndimage.zoom` (with rounding), centred 140 mm crop/pad to 336, then upsample to 384. Cache v2 stores the exact per-series affine $A_s$, with a unit test that it reprojects to DICOM within 0.5 px. Token cell size now comes from $A_s$, not PixelSpacing. |
| 3 | Canonical frame too simple | **Accept, narrower scope** | The core attention bias uses only relative Mahalanobis distances, so it does not depend on the frame. The knee basis (orthonormalised from the sagittal normal and coronal orientation) is used for interpretation and side labels only. |
| 4 | Triplet covariance should use the actual three slice centres | **Accept** | Moment-matched mixture of the three actual footprints. |
| 5 | $\log\mathcal N$ bias hides a $\log\lvert\Sigma\rvert$ resolution preference | **Accept** | Mahalanobis term only; the resolution term is a separate ablation, off by default. |
| 6 | Attention is not localization | **Accept** | Renamed "evidence centroid"; claimed only if validation passes. We already measured that the attention is broad (16–40 effective windows on gold-58), which supports caution. |
| 7 | $\mathcal L_{\text{geo}}$ can collapse onto the notch | **Accept** | First model is classification only. $\mathcal L_{\text{geo}}$ comes later, with your safeguards. |
| 8 | ACL and MCL are not point lesions | **Accept** | Point centroids only for meniscal tear, contusion, fracture, Baker's cyst. Ligaments become principal-axis regions later. |
| 9 | CoPAS is closer than admitted; different 12 categories | **Accept, verified in the full text (PMC11368947)** | CoPAS argues interpolation adds nothing and uses orthogonal planes. Its "cross-plane volumes" are rotated voxel arrays with learned cross-attention and a meniscus-segmentation crop, with no patient coordinates. Its categories differ (PCL, LCL, plica, fat pad, retinaculum). New fact: **its code is public (github.com/zqiuak/CoPAS)**, so it becomes a run baseline, not a re-implementation. Our claim is narrowed to geometry-determined correspondence with acquisition support. |
| 10 | fastMRI+ validates only projected 2D localization | **Accept** | H2 reworded. 3D validation needs a hand-annotated RSNA subset (plus 3DReasonKnee, arXiv:2510.20967, if OAI access works). |
| 11 | Signal-equation embedding is underdetermined | **Accept** | Replaced by measured metadata + missingness masks → MLP. The signal model is kept only as an ablation. |
| 12 | INR arm is a separate problem | **Accept** | Journal backlog. |
| B1 | Public "SWA" = selected epochs averaged | **Accept, verified** | The public author's post (`discussion/746792`) says the checkpoint averages three selected epochs (chosen on gold-58). We already average our top 3 by fold validation. Dropped. |
| B2 | Gap attribution too strong | **Accept, verified** | Their own table shows small, confounded steps (density +0.001, OAI +0.005, mirror +0.001). Their starting point was already 0.939 public with blended public LLM label sets, one tuned on gold-58. Reworded to "plausible, confounded contributors". |
| S1 | Report-label OOF should not be the primary endpoint; DeLong needs hard labels | **Accept, one adjustment** | Endpoint hierarchy adopted. Adjustment: for H1 the most relevant confirmatory set is the RSNA hidden test (expert image-review labels, ~1,300 studies) **if** late scoring works. External cohorts test generalization, because their labels and protocols differ. DeLong only on expert binary labels; cluster bootstrap for proxy labels. |
| S2 | Late submissions not a safe endpoint | **Accept, now partly better** | The competition page shows a "Late Submission" button today (2026-10-23). Still conditional: there is no host statement, only aggregate macro-AUC comes back, and submissions are rationed. |

## New findings since your review (metadata audit, `scripts/audit_data_usage.py`)

Our current cache uses only **32.5% of the 819,078 slices** and **84.3% of the series**: one series per slot, five slots.

- **Slot subsampling:** native-in-span ÷ kept is Sag-FS 1.44, Sag 1.86, Cor-FS 2.17, Cor 3.25, Ax 2.33.
- **Consequence:** our "adjacent-slice" triplets are adjacent only after subsampling. In coronal non-FS one window spans
  roughly 40 mm.
- **No anti-aliasing:** 76% of series are downsampled to the 0.417 mm grid (median 1.29×, p90 1.85×) with bilinear
  `zoom` and no prefilter.
- **Crop:** the fixed centred 140 mm crop drops a median 23% of image area (median field of view 160 mm); 5.7% of series
  are padded.

**Change:** Phase 0 now builds **cache v2**:
- every series and every native slice in span (≈ 106 GB at 384² uint8; quota fits);
- anti-aliased resampling;
- per-slice geometry, affine and acquisition metadata;
- windows of three native-adjacent slices.

The matched baseline **B\*** uses cache v2 without geometry. Comparing exp018's cache with cache v2 on fold 0 isolates
the data-usage effect, so it cannot be mistaken for a geometry effect later.

## Added controls, to make H1 causal

1. **Capacity-matched no-geometry control:** same transformer, tokens and neighbour budget (random neighbours,
   slice-index positions).
2. **Geometry shuffle:** each series gets the geometry of a random other study; the gain must vanish.
3. **Dose–response:** inject 2 / 5 / 10 mm test-time misregistration into one plane; AUC must fall monotonically.

## Questions for this round

1. Is the Phase 0.1 registration design adequate, given that 3 mm slices limit precision along the normal? Would you
   register orthogonal pairs only, so each plane constrains the other's normal?
2. Are controls 1–3 sufficient for a causal reading of H1, or is something missing (for example a "geometry from the
   wrong plane" control)?
3. Is the endpoint hierarchy acceptable for MICCAI when the confirmatory set returns only one aggregate macro-AUC per
   submission?
4. For efficiency (~1–3k tokens per study at inference): per-plane k-NN attention, or sampling along cross-plane
   intersection lines?
5. Novelty: given CoPAS (now a run baseline) and PETR-style 3D position encodings, is "DICOM-geometry correspondence with
   acquisition support" enough for MICCAI on its own, or does it need H2 (validated evidence centroids) to pass?
6. Cache v2 design: any objection to using every series without a slot cap? (The geometry model takes a bag of series;
   B\* would get the same bag through extra slots.)
