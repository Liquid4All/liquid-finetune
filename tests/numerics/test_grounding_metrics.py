from __future__ import annotations

import sys

import pytest


# === Grounding Metric Tests ===


class TestStrictParser:
    """Internal contract of ``_parse_bboxes`` (strict, reward-aligned).
    F1 e2e tests below always use in-range valid coords, so they don't
    catch a regression where strict checks get relaxed — pin them here.
    """

    def test_rejects_out_of_range_coords(self):
        from liquid_finetune.evaluation.metrics import _parse_bboxes

        assert _parse_bboxes('[{"label":"x","bbox":[0,0,1,1.5]}]') == []
        assert _parse_bboxes('[{"label":"x","bbox":[-0.1,0,1,1]}]') == []

    def test_rejects_zero_or_inverted_area(self):
        from liquid_finetune.evaluation.metrics import _parse_bboxes

        assert _parse_bboxes('[{"label":"x","bbox":[0.5,0,0.5,1]}]') == []  # x2==x1
        assert _parse_bboxes('[{"label":"x","bbox":[1,0,0,1]}]') == []  # x2<x1


class TestHungarianMatching:
    """Hungarian matcher contracts not covered by F1 e2e tests."""

    def test_picks_optimal_assignment_not_greedy(self, monkeypatch):
        """When scipy is available, the matcher must pick the OPTIMAL
        assignment, not greedy. Pin this on an adversarial sim matrix
        that strictly distinguishes the two algorithms.

        Matrix ``[[1.0, 0.9], [0.9, 0.1]]``:
          * Greedy: picks (0,0)=1.0 → forced to (1,1)=0.1 → sum 1.1
          * Optimal: cross-assigns (0,1)+(1,0) = 0.9+0.9 = 1.8

        The 2D IoU geometry can't reproduce this matrix from real
        bboxes (triangle-inequality-ish constraints), so we mock
        ``_compute_iou`` to inject it directly. The test is skipped
        if scipy is not installed — in that environment the greedy
        fallback IS the implementation, and the optimal-vs-greedy
        contract doesn't apply.
        """
        pytest.importorskip("scipy")

        from liquid_finetune.evaluation import metrics

        sim_values = iter([1.0, 0.9, 0.9, 0.1])  # row-major: (0,0),(0,1),(1,0),(1,1)
        monkeypatch.setattr(metrics, "_compute_iou", lambda a, b: next(sim_values))

        pred = [[0, 0, 1, 1], [0, 0, 1, 1]]  # values irrelevant; mock intercepts
        gt = [[0, 0, 1, 1], [0, 0, 1, 1]]
        ious = metrics._hungarian_match_iou(pred, gt)
        # Greedy would give 1.1; scipy gives 1.8.
        assert sum(ious) == pytest.approx(1.8)


class TestGroundingIouF1:
    """Multi-bbox F1 metric — the new contract used by mgrounding_test."""

    def test_empty_pred_empty_gt_is_one(self):
        from liquid_finetune.evaluation.metrics import score_grounding_iou_f1

        # Correct abstention: model emits no boxes when GT has no boxes.
        assert score_grounding_iou_f1("[]", "[]") == 1.0

    def test_empty_pred_nonempty_gt_is_zero(self):
        from liquid_finetune.evaluation.metrics import score_grounding_iou_f1

        assert score_grounding_iou_f1("[]", '[{"label":"x","bbox":[0,0,1,1]}]') == 0.0

    def test_multi_permuted_still_matches(self):
        """Hungarian matching is order-invariant."""
        from liquid_finetune.evaluation.metrics import score_grounding_iou_f1

        s = score_grounding_iou_f1(
            '[{"label":"b","bbox":[0.5,0.5,1,1]},{"label":"a","bbox":[0,0,0.5,0.5]}]',
            '[{"label":"a","bbox":[0,0,0.5,0.5]},{"label":"b","bbox":[0.5,0.5,1,1]}]',
        )
        assert s == pytest.approx(1.0)

    def test_extra_pred_drags_precision(self):
        """2 preds vs 1 gt, one matches perfectly → F1=2/3."""
        from liquid_finetune.evaluation.metrics import score_grounding_iou_f1

        s = score_grounding_iou_f1(
            '[{"label":"a","bbox":[0,0,0.5,0.5]},{"label":"b","bbox":[0.6,0.6,1,1]}]',
            '[{"label":"a","bbox":[0,0,0.5,0.5]}]',
        )
        assert s == pytest.approx(2 / 3, abs=1e-6)

    def test_registered_in_dispatch(self):
        from liquid_finetune.evaluation.metrics import compute_metric

        s = compute_metric(
            "grounding_iou_f1",
            prediction='[{"label":"x","bbox":[0,0,1,1]}]',
            ground_truth='[{"label":"x","bbox":[0,0,1,1]}]',
        )
        assert s == pytest.approx(1.0)


class TestGroundingIouLegacyFormats:
    """Permissive parser kept for refcoco-style baselines. A previous
    refactor silently broke these by routing through the strict parser —
    pins the formats that must keep working.
    """

    def test_accepts_bare_4_list(self):
        from liquid_finetune.evaluation.metrics import score_grounding_iou

        assert score_grounding_iou("[0, 0, 1, 1]", "[0, 0, 1, 1]") == 1.0

    def test_accepts_prose_embedded_json(self):
        from liquid_finetune.evaluation.metrics import score_grounding_iou

        assert (
            score_grounding_iou("Sure! The bbox is [0, 0, 1, 1].", "[0, 0, 1, 1]")
            == 1.0
        )

    def test_rescales_0_1000_coords(self):
        """MGrounding-native 0-1000 coord space auto-scales to 0-1."""
        from liquid_finetune.evaluation.metrics import score_grounding_iou

        assert score_grounding_iou("[0, 0, 1000, 1000]", "[0, 0, 1, 1]") == 1.0


class TestGroundingIouMalformedDoesNotInflate:
    """``Benchmark.evaluate`` excludes per-sample failures from the count,
    so any parser exception silently inflates the mean. Pin the regression
    paths Codex caught.
    """

    def test_boolean_coords_rejected_not_coerced(self):
        """``[false, false, true, true]`` must NOT coerce to ``[0,0,1,1]``.

        ``bool`` is a subclass of ``int``, so float() would turn booleans
        into 0.0/1.0 and a JSON-bool prediction would score IoU=1.0 against
        a full-image GT — a free perfect score from gibberish output.
        """
        from liquid_finetune.evaluation.metrics import _parse_bbox, score_grounding_iou

        assert _parse_bbox("[false, false, true, true]") is None
        assert score_grounding_iou("[false, false, true, true]", "[0, 0, 1, 1]") == 0.0

    def test_garbage_input_never_raises(self):
        """Pathological strings must return None, never raise — a raised
        exception would drop the sample from the count and inflate."""
        from liquid_finetune.evaluation.metrics import _parse_bbox

        for junk in [
            "",
            '{"bbox": 42}',  # int bbox; used to crash on len(int)
            '{"bbox": "hello"}',
            '{"bbox": null}',
            '[{"bbox": 42}]',
            "[0, 0, NaN, 1]",  # NaN coord
        ]:
            assert _parse_bbox(junk) is None, f"{junk!r} should parse to None"


class TestHungarianFallback:
    """The scipy-or-greedy fallback is an environment contract: customers
    on minimal installs without scipy must still get correct multi-bbox
    matching."""

    def test_greedy_fallback_when_scipy_missing(self, monkeypatch):
        import builtins

        from liquid_finetune.evaluation.metrics import _hungarian_match_iou

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("scipy disabled for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for mod in list(sys.modules):
            if mod.startswith("scipy"):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        pred = [[0, 0, 1, 1], [0, 0, 0.5, 0.5]]
        gt = [[0, 0, 0.5, 0.5], [0, 0, 1, 1]]
        ious = _hungarian_match_iou(pred, gt)
        assert len(ious) == 2
        assert sum(ious) == pytest.approx(2.0)
