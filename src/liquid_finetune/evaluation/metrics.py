import ast
import json
import math
import re
from collections import Counter


# === Built-in scoring functions ===


def score_grounding_iou(
    prediction: str, ground_truth: str, iou_threshold: float = 0.5, **_
) -> float:
    """1.0 if predicted bbox overlaps ground truth above threshold, else 0.0.

    Single-bbox metric: only the first bbox in either side is read. For
    multi-bbox samples (Group_Grounding, Object_Tracking, …) use
    ``grounding_iou_f1`` instead.
    """
    pred_bbox = _parse_bbox(prediction)
    gt_bbox = _parse_bbox(ground_truth)

    if pred_bbox is None or gt_bbox is None:
        return 0.0

    return 1.0 if _compute_iou(pred_bbox, gt_bbox) >= iou_threshold else 0.0


def score_grounding_iou_f1(prediction: str, ground_truth: str, **_) -> float:
    """Soft F1 over Hungarian-matched (pred, gt) bbox pairs scored by IoU.

    Multi-bbox safe — required for Group_Grounding and Object_Tracking
    samples whose ground truth is a list of bboxes (one per object or per
    input frame). Returns a continuous score in [0, 1]:
      * (no preds, no gt) → 1.0 (correct abstention)
      * (preds without gt, or vice versa) → 0.0
      * else: F1 = 2·P·R/(P+R) where P = Σ(matched IoUs)/n_pred,
        R = Σ(matched IoUs)/n_gt; unmatched preds drag precision,
        unmatched gt boxes drag recall.

    Mirrors the GRPO reward's iou_f1 semantics so train and eval signals
    are aligned.
    """
    pred_boxes = _parse_bboxes(prediction)
    gt_boxes = _parse_bboxes(ground_truth)

    if not pred_boxes and not gt_boxes:
        return 1.0
    if not pred_boxes or not gt_boxes:
        return 0.0

    matches = _hungarian_match_iou(pred_boxes, gt_boxes)
    score = sum(max(0.0, iou) for iou in matches)
    precision = score / len(pred_boxes)
    recall = score / len(gt_boxes)
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def score_short_answer(
    prediction: str, ground_truth: str, match_mode: str = "contains", **_
) -> float:
    """1.0 if ground truth appears in the prediction (case-insensitive).

    match_mode="any_in_array": ground truth is a JSON array — match if any element
    appears in the prediction.
    """
    if (
        match_mode == "any_in_array"
        and ground_truth.startswith("[")
        and ground_truth.endswith("]")
    ):
        gt_clean = ground_truth.replace('""', '"')
        pred_lower = prediction.lower().strip().replace("\n", " ")
        try:
            gt_array = ast.literal_eval(gt_clean)
        except (ValueError, SyntaxError):
            gt_array = [ground_truth]

        for gt_item in gt_array:
            if str(gt_item).lower().strip().replace("\n", " ") in pred_lower:
                return 1.0
        return 0.0

    return 1.0 if ground_truth.lower().strip() in prediction.lower().strip() else 0.0


_GSM8K_MARKER = re.compile(r"####\s*([\-\+]?[\d,\.]+)")
_GSM8K_ANY_NUMBER = re.compile(r"([\-\+]?[\d,\.]+)")


def _normalize_gsm8k_number(s: str) -> str | None:
    if s is None:
        return None
    s = str(s).strip().rstrip(".").replace(",", "").replace("$", "").replace(" ", "")
    if not s or not re.search(r"\d", s):
        return None
    try:
        return str(float(s))
    except (ValueError, TypeError):
        return s or None


def _extract_gsm8k_answer(text: str) -> str | None:
    if not text:
        return None
    tail = text[-1024:]
    matches = _GSM8K_MARKER.findall(tail)
    if matches:
        return _normalize_gsm8k_number(matches[-1])
    matches = _GSM8K_ANY_NUMBER.findall(tail)
    if matches:
        return _normalize_gsm8k_number(matches[-1])
    return None


def score_gsm8k(prediction: str, ground_truth: str, **_) -> float:
    """1.0 if the numeric answer extracted from prediction matches ground truth.

    Accepts either a clean number or a full chain-of-thought ending in
    ``#### N`` on either side.
    """
    gt = _extract_gsm8k_answer(ground_truth)
    pred = _extract_gsm8k_answer(prediction)
    if gt is None or pred is None:
        return 0.0
    return 1.0 if gt == pred else 0.0


def score_mcq_gen(prediction: str, ground_truth: str, **_) -> float:
    """1.0 if extracted MCQ letter matches ground truth."""
    if prediction.strip() == ground_truth.strip():
        return 1.0

    gt_letter = _extract_mcq_answer(ground_truth)
    pred_letter = _extract_mcq_answer(prediction)

    if gt_letter is None or pred_letter is None:
        return 0.0

    return 1.0 if gt_letter == pred_letter else 0.0


def score_bleu(prediction: str, ground_truth: str, max_n: int = 4, **_) -> float:
    """Sentence-level BLEU score (0-1) up to max_n-grams with brevity penalty."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(ground_truth)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = (
        min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))
        if pred_tokens
        else 0.0
    )

    # n-gram precisions with +1 smoothing (smoothing method 1)
    log_avg = 0.0
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = max(sum(pred_ngrams.values()), 1)
        # Add-1 smoothing for n > 1 to avoid zero precision killing the score
        if n > 1:
            clipped += 1
            total += 1
        precision = clipped / total
        if precision == 0:
            return 0.0
        log_avg += math.log(precision) / max_n

    return bp * math.exp(log_avg)


def score_rouge_l(prediction: str, ground_truth: str, **_) -> float:
    """ROUGE-L F1 score (0-1) based on longest common subsequence."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(ground_truth)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# === Metric dispatch (add new metrics here) ===

_METRIC_DISPATCH: dict[str, callable] = {
    "grounding_iou": score_grounding_iou,
    "grounding_iou_f1": score_grounding_iou_f1,
    "short_answer": score_short_answer,
    "mcq_gen": score_mcq_gen,
    "gsm8k": score_gsm8k,
    "bleu": score_bleu,
    "rouge_l": score_rouge_l,
}


def compute_metric(
    metric_type: str, prediction: str, ground_truth: str, **kwargs
) -> float:
    """Dispatch to the named scoring function and return a 0-1 score."""
    fn = _METRIC_DISPATCH.get(metric_type)
    if fn is None:
        raise ValueError(
            f"Unknown metric: {metric_type!r}. Available: {sorted(_METRIC_DISPATCH)}"
        )
    return fn(prediction, ground_truth, **kwargs)


# === Internal helpers ===

_VALID_MCQ_LETTERS = set("abcdef")

_MCQ_PATTERNS = [
    r"\b([a-f])[\.\(\):, ]?",
    r"(?:option |choice |answer: |the answer is )\s*([a-f])\b",
    r"^([a-f])[:).]",
    r"\b([a-f])[\.\s]*(?:is correct|is the answer|appears to be right)\b",
]


def _extract_mcq_answer(text: str) -> str | None:
    """Extract a single MCQ letter (A-F) from free-form text."""
    text = text.strip().lower()

    if len(text) == 1 and text in _VALID_MCQ_LETTERS:
        return text.upper()

    for pattern in _MCQ_PATTERNS:
        matches = re.findall(pattern, text)
        valid = [m for m in matches if m in _VALID_MCQ_LETTERS]
        if len(valid) == 1:
            return valid[0].upper()
    return None


def _parse_bbox(text: str) -> list[float] | None:
    """Permissive single-bbox parser for the legacy ``grounding_iou`` metric.

    Accepts the wider set of formats refcoco-style baselines emit:
      * prose-embedded JSON (regex-extracted)
      * bare ``[x,y,x,y]`` and list-of-lists
      * list-of-dicts with ``bbox`` field
      * 0-1000 coord space (autoscaled to 0-1)
      * ``ast.literal_eval`` fallback when ``json.loads`` fails

    Returns ``None`` on ANY malformed input — never raises. Hardening
    matters: ``Benchmark.evaluate`` excludes per-sample failures from
    the count, so a parser exception would silently drop the sample
    instead of scoring it as 0 — inflating the mean.

    ``grounding_iou_f1`` uses ``_parse_bboxes`` directly: it must mirror
    the GRPO reward's strict rules. Loosening this legacy path is
    intentional — ``grounding_iou`` scores published checkpoints whose
    output predates the cookbook recipe; strict parsing would silently
    deflate baseline numbers.
    """
    if not isinstance(text, str):
        return None
    try:
        return _parse_bbox_unchecked(text)
    except Exception:
        # Any malformed input we didn't anticipate (TypeError on len() of an
        # int-shaped "bbox", recursion depth on pathological JSON, ...) maps
        # to "no valid bbox" rather than letting the exception escape.
        return None


def _parse_bbox_unchecked(text: str) -> list[float] | None:
    text = text.strip()
    json_match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

    bbox = None
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            for item in data:
                if isinstance(item, dict) and "bbox" in item:
                    bbox = item["bbox"]
                    break
        elif isinstance(data[0], list) and len(data[0]) == 4:
            bbox = data[0]
        elif len(data) == 4 and all(isinstance(x, (int, float)) for x in data):
            bbox = data
    elif isinstance(data, dict) and "bbox" in data:
        bbox = data["bbox"]

    # isinstance check covers cases where bbox came from {"bbox": <non-list>}
    # — without this, ``len(bbox)`` raises TypeError on int/float/dict values
    # and the failure inflates the running mean.
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    # Reject bool entries before they coerce to 0.0/1.0: ``bool`` subclasses
    # ``int``, so ``[false, false, true, true]`` would otherwise become a
    # clean-looking ``[0, 0, 1, 1]`` and score IoU=1.0 against the full-image
    # GT — a free perfect score from gibberish output.
    if any(isinstance(v, bool) for v in bbox):
        return None

    try:
        bbox = [float(x) for x in bbox]
    except (ValueError, TypeError):
        return None
    if any(not math.isfinite(c) for c in bbox):
        return None  # NaN/Inf would poison _compute_iou downstream.

    # 0-1000 coord space (MGrounding native) autoscaled to 0-1.
    if max(abs(c) for c in bbox) > 1.5:
        bbox = [c / 1000.0 for c in bbox]

    return bbox


def _parse_bboxes(text: str) -> list[list[float]]:
    """Strict twin of the GRPO reward's bbox parser (rewards/tasks/
    vlm_grounding/recipe.py: _extract_bboxes / _parse_gt_bboxes / _validate_bbox).

    A completion that scores 0.0 on the reward must also score 0.0 here, so
    we deliberately do NOT relax any of the reward's rules:
      * ``json.loads`` only — no ast.literal_eval, no regex JSON-extraction
        from prose. Any preamble around the JSON kills the parse.
      * Top-level must be a list. Each item must be a dict with a ``bbox``
        field; bare ``[x,y,x,y]`` and list-of-lists are rejected.
      * No 0-1000 → 0-1 rescaling: the reward computes IoU against 0-1 GT
        with whatever scale the model emitted, so we do too.
      * NaN/Inf and inverted/zero-area boxes (``x2<=x1`` or ``y2<=y1``)
        are rejected (matches ``_validate_bbox``).
    """
    if not isinstance(text, str):
        return []
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []

    out: list[list[float]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        b = item.get("bbox")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in b):
            continue
        if not all(math.isfinite(float(v)) for v in b):
            continue
        box = [float(v) for v in b]
        # Range check: coords must be normalized to [0, 1]. The model is
        # trained on bboxes in that range, so anything outside is a format
        # violation — give it 0 instead of partial credit for accidentally
        # overlapping the GT.
        if any(c < 0.0 or c > 1.0 for c in box):
            continue
        x1, y1, x2, y2 = box
        if not (x2 > x1 and y2 > y1):
            continue
        out.append(box)
    return out


def _hungarian_match_iou(
    pred_boxes: list[list[float]], gt_boxes: list[list[float]]
) -> list[float]:
    """Return the matched-pair IoUs from a 1-to-1 max-similarity bipartite
    matching of ``pred_boxes`` against ``gt_boxes``. Uses scipy when
    available; greedy otherwise. The list length is ``min(n_pred, n_gt)``.
    """
    sim = [[_compute_iou(p, g) for g in gt_boxes] for p in pred_boxes]
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        cost = -np.asarray(sim)
        rows, cols = linear_sum_assignment(cost)
        return [sim[r][c] for r, c in zip(rows, cols)]
    except ImportError:
        pass
    used_p: set[int] = set()
    used_g: set[int] = set()
    matched: list[float] = []
    max_n = min(len(pred_boxes), len(gt_boxes))
    while len(matched) < max_n:
        best = -math.inf
        best_p = best_g = -1
        for i in range(len(pred_boxes)):
            if i in used_p:
                continue
            for j in range(len(gt_boxes)):
                if j in used_g:
                    continue
                if sim[i][j] > best:
                    best = sim[i][j]
                    best_p, best_g = i, j
        if best_p < 0:
            break
        matched.append(best)
        used_p.add(best_p)
        used_g.add(best_g)
    return matched


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization for BLEU/ROUGE."""
    return text.lower().split()


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Count n-grams in a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (space-optimised)."""
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ai in a:
        curr = [0] * (len(b) + 1)
        for j, bj in enumerate(b):
            curr[j + 1] = prev[j] + 1 if ai == bj else max(prev[j + 1], curr[j])
        prev = curr
    return prev[-1]


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute intersection-over-union between two [x1, y1, x2, y2] boxes."""
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # Handle inverted coordinates
    if x1_a > x2_a:
        x1_a, x2_a = x2_a, x1_a
    if y1_a > y2_a:
        y1_a, y2_a = y2_a, y1_a
    if x1_b > x2_b:
        x1_b, x2_b = x2_b, x1_b
    if y1_b > y2_b:
        y1_b, y2_b = y2_b, y1_b

    x1 = max(x1_a, x1_b)
    y1 = max(y1_a, y1_b)
    x2 = min(x2_a, x2_b)
    y2 = min(y2_a, y2_b)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0
