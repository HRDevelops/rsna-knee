"""Unit tests for per-series laterality resolution (CLAUDE.md §5, 2026-09-07 strategy review
step 8): a study flagged as a genuine cross-series L/R tag conflict (almost certainly a bilateral
exam — both knees imaged in one study) must be flipped PER-SERIES using that series' own DICOM
tag, not given one study-wide side decision the way every other study is.

Run with: python -m unittest tests/test_laterality.py -v
(or via pytest if installed; plain unittest is used here to avoid adding a new pip dependency)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.dataset import normalize_tag_side, resolve_side  # noqa: E402


class TestNormalizeTagSide(unittest.TestCase):
    def test_right_variants(self):
        self.assertEqual(normalize_tag_side("R"), "R")
        self.assertEqual(normalize_tag_side("RIGHT"), "R")
        self.assertEqual(normalize_tag_side("right"), "R")

    def test_left_variants(self):
        self.assertEqual(normalize_tag_side("L"), "L")
        self.assertEqual(normalize_tag_side("LEFT"), "L")

    def test_bilateral(self):
        self.assertEqual(normalize_tag_side("B"), "BILATERAL")

    def test_blank_or_unknown(self):
        self.assertIsNone(normalize_tag_side(""))
        self.assertIsNone(normalize_tag_side(None))
        self.assertIsNone(normalize_tag_side("XYZ"))


class TestResolveSideOrdinaryStudy(unittest.TestCase):
    """A normal (non-conflict) study: every series shares the single per-study side, regardless
    of what that series' own tag says — this is the pre-existing, non-bilateral-aware behavior
    that must be preserved for the other ~4380 studies."""

    def setUp(self):
        self.bilateral_conflict_studies = set()  # empty — no conflicts in this scenario
        self.series_tag_side = {("study_A", "series_1"): "L"}  # deliberately disagrees with final_side
        self.final_side_map = {"study_A": "R"}

    def test_uses_study_level_side_even_if_series_tag_disagrees(self):
        side = resolve_side("study_A", "series_1", self.bilateral_conflict_studies,
                             self.series_tag_side, self.final_side_map)
        self.assertEqual(side, "R", "non-conflict studies must use the per-study final_side")

    def test_unknown_study_returns_none(self):
        side = resolve_side("study_unknown", "series_x", self.bilateral_conflict_studies,
                             self.series_tag_side, self.final_side_map)
        self.assertIsNone(side)


class TestResolveSideSyntheticBilateralStudy(unittest.TestCase):
    """The core case this test file exists for: a synthetic bilateral study with a genuine
    cross-series L/R tag conflict (docs/LATERALITY_RESOLUTION.md — real examples in the dataset
    have exactly this shape: some series tagged L, others R, within the same study)."""

    def setUp(self):
        self.study_uid = "bilateral_study_1"
        self.bilateral_conflict_studies = {self.study_uid}
        self.series_tag_side = {
            (self.study_uid, "series_left_knee"): "L",
            (self.study_uid, "series_right_knee"): "R",
            (self.study_uid, "series_no_tag"): None,
        }
        # A study-level side WOULD exist in final_side.csv (e.g. "R" from conflict_resolved_by_report
        # per resolve_laterality.py) but must be IGNORED for this study once it's flagged bilateral.
        self.final_side_map = {self.study_uid: "R"}

    def test_left_series_resolves_to_left(self):
        side = resolve_side(self.study_uid, "series_left_knee", self.bilateral_conflict_studies,
                             self.series_tag_side, self.final_side_map)
        self.assertEqual(side, "L")

    def test_right_series_resolves_to_right(self):
        side = resolve_side(self.study_uid, "series_right_knee", self.bilateral_conflict_studies,
                             self.series_tag_side, self.final_side_map)
        self.assertEqual(side, "R")

    def test_two_series_in_same_study_resolve_differently(self):
        """The actual regression this test guards against: per-STUDY resolution would give both
        series the same side (whatever final_side_map says) and silently corrupt one of them."""
        left = resolve_side(self.study_uid, "series_left_knee", self.bilateral_conflict_studies,
                             self.series_tag_side, self.final_side_map)
        right = resolve_side(self.study_uid, "series_right_knee", self.bilateral_conflict_studies,
                              self.series_tag_side, self.final_side_map)
        self.assertNotEqual(left, right,
                             "a bilateral study's two knee series must NOT resolve to the same side")

    def test_series_with_no_own_tag_falls_back_to_none_not_study_side(self):
        """Must not silently inherit the (ignored) per-study side as a fallback — that would
        reintroduce the exact bug this whole mechanism exists to avoid."""
        side = resolve_side(self.study_uid, "series_no_tag", self.bilateral_conflict_studies,
                             self.series_tag_side, self.final_side_map)
        self.assertIsNone(side)


if __name__ == "__main__":
    unittest.main()
