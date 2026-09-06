from __future__ import annotations

import json
import math
from typing import Sequence

from liquid_finetune.rl.rewards import Recipe


# === VLM grounding box rewards ===
#
# Completions and solutions are JSON arrays of {"label", "bbox"} objects with
# normalized [0, 1] bounding boxes in [x1, y1, x2, y2] format.


def _iou(pred: Sequence[float], gt: Sequence[float]) -> float:
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    pa = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    ga = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pa + ga - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _ciou(pred: Sequence[float], gt: Sequence[float]) -> float:
    """Complete IoU: IoU minus center-distance and aspect-ratio penalties."""
    iou = _iou(pred, gt)

    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt

    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    center_dist_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    ex1, ey1 = min(px1, gx1), min(py1, gy1)
    ex2, ey2 = max(px2, gx2), max(py2, gy2)
    diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    if diag_sq <= 0:
        return iou

    pw = max(1e-6, px2 - px1)
    ph = max(1e-6, py2 - py1)
    gw = max(1e-6, gx2 - gx1)
    gh = max(1e-6, gy2 - gy1)
    v = (4.0 / (math.pi**2)) * (math.atan(gw / gh) - math.atan(pw / ph)) ** 2
    alpha = v / (1.0 - iou + v + 1e-6) if iou >= 0.5 else 0.0

    return iou - (center_dist_sq / diag_sq) - alpha * v


def _get_text(completion) -> str:
    if isinstance(completion, list):
        if completion and isinstance(completion[0], dict):
            return completion[0].get("content", "") or ""
        return ""
    return str(completion) if completion is not None else ""


def _is_finite(value) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _validate_bbox(bbox) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if not all(_is_finite(v) for v in bbox):
        return None
    coords = [float(v) for v in bbox]
    x1, y1, x2, y2 = coords
    if not (x2 > x1 and y2 > y1):
        return None
    return coords


def _extract_bboxes(text: str) -> tuple[list[list[float]], int] | None:
    """Parse completion as a JSON array of ``{bbox: [...]}`` dicts."""
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, list) or not parsed:
        return None

    valid: list[list[float]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        coords = _validate_bbox(item.get("bbox"))
        if coords is not None:
            valid.append(coords)

    if not valid:
        return None
    return valid, len(parsed)


def _parse_gt_bboxes(solution_text) -> list[list[float]]:
    if not solution_text:
        return []
    try:
        parsed = json.loads(str(solution_text).strip())
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    out: list[list[float]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        coords = _validate_bbox(item.get("bbox"))
        if coords is not None:
            out.append(coords)
    return out


def _hungarian_match(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
    similarity=_iou,
) -> list[tuple[int, int, float]]:
    """1-to-1 bipartite matching that maximizes total similarity.

    Uses ``scipy.optimize.linear_sum_assignment`` when available, else a
    greedy fallback seeded at ``-inf`` so negative similarities still
    match. Returns ``[(pred_idx, gt_idx, similarity), ...]``.
    """
    if not pred_boxes or not gt_boxes:
        return []

    sim_matrix = [[similarity(p, g) for g in gt_boxes] for p in pred_boxes]

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        cost = -np.asarray(sim_matrix)
        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c), sim_matrix[r][c]) for r, c in zip(rows, cols)]
    except ImportError:
        pass

    matches: list[tuple[int, int, float]] = []
    used_p: set[int] = set()
    used_g: set[int] = set()
    max_matches = min(len(pred_boxes), len(gt_boxes))
    while len(matches) < max_matches:
        best = -math.inf
        best_p = best_g = -1
        for i in range(len(pred_boxes)):
            if i in used_p:
                continue
            for j in range(len(gt_boxes)):
                if j in used_g:
                    continue
                if sim_matrix[i][j] > best:
                    best = sim_matrix[i][j]
                    best_p, best_g = i, j
        if best_p < 0:
            break
        matches.append((best_p, best_g, best))
        used_p.add(best_p)
        used_g.add(best_g)
    return matches


def strict_format_reward(completions, **kwargs) -> list[float]:
    """1.0 if the completion is a parseable JSON bbox array, else 0.0."""
    return [
        1.0 if _extract_bboxes(_get_text(c)) is not None else 0.0 for c in completions
    ]


def _f1_reward(completions, solution, similarity) -> list[float]:
    if solution is None:
        return [0.0] * len(completions)

    rewards: list[float] = []
    for completion, gt_text in zip(completions, solution, strict=False):
        parsed = _extract_bboxes(_get_text(completion))
        pred_boxes = parsed[0] if parsed is not None else []
        gt_boxes = _parse_gt_bboxes(gt_text)

        num_pred = len(pred_boxes)
        num_gt = len(gt_boxes)

        if num_gt == 0 and num_pred == 0:
            rewards.append(1.0)
            continue
        if num_gt == 0 or num_pred == 0:
            rewards.append(0.0)
            continue

        matches = _hungarian_match(pred_boxes, gt_boxes, similarity=similarity)
        score = sum(max(0.0, s) for _, _, s in matches)
        precision = score / num_pred
        recall = score / num_gt
        if precision + recall <= 0.0:
            rewards.append(0.0)
            continue
        rewards.append(2.0 * precision * recall / (precision + recall))
    return rewards


def iou_f1_reward(completions, solution=None, **kwargs) -> list[float]:
    """Soft F1 over Hungarian-matched predicted/GT box pairs scored by IoU.

    Precision is the sum of matched IoUs divided by the number of
    predicted boxes; recall divides by the number of GT boxes; the
    reward is their F1, in ``[0, 1]``. Multi-object safe; abstention
    (no preds, no GT) scores 1.0; format failures score 0.0 when any
    GT is present.
    """
    return _f1_reward(completions, solution, similarity=_iou)


def ciou_f1_reward(completions, solution=None, **kwargs) -> list[float]:
    """Same F1 shape as :func:`iou_f1_reward` but the per-pair metric is CIoU.

    CIoU is IoU minus a center-distance penalty and an aspect-ratio
    penalty, so pairings that share overlap *and* center alignment
    *and* aspect ratio score higher. The Hungarian matcher also runs
    on CIoU, so assignments prefer center-aligned pairs even when raw
    overlap ties.
    """
    return _f1_reward(completions, solution, similarity=_ciou)


class VLMGroundingIoURecipe(Recipe):
    description = (
        "VLM grounding - JSON bbox array with strict format check and "
        "soft F1 of Hungarian-matched IoUs."
    )

    required_columns = ("prompt", "solution")

    system_prompt = (
        "You are a visual grounding assistant. Given an image and an "
        "instruction, reply with a JSON array of objects: "
        '[{"label": "...", "bbox": [x1, y1, x2, y2]}, ...]. '
        "Bounding boxes use normalized [0, 1] coordinates in "
        "[x1, y1, x2, y2] format (top-left, bottom-right). "
        "Do not output any text outside the JSON array."
    )

    def rewards(self):
        return [
            (strict_format_reward, 0.1),
            (iou_f1_reward, 1.0),
        ]


class VLMGroundingCIoURecipe(VLMGroundingIoURecipe):
    description = (
        "VLM grounding - JSON bbox array with strict format check and "
        "soft F1 of Hungarian-matched CIoUs."
    )

    def rewards(self):
        return [
            (strict_format_reward, 0.1),
            (ciou_f1_reward, 1.0),
        ]
