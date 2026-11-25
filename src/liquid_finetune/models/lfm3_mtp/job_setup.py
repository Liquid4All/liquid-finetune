"""Build the MTP start point a job's ``mtp:`` section asks for, on the training node."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import shutil

from liquid_finetune.checkpointing.model_loading import _resolve_model_id
from liquid_finetune.config.job_config import ResolvedJobConfig
from liquid_finetune.models.lfm3_mtp.prepare import (
    MANIFEST_NAME,
    prepare_lfm3_mtp_checkpoint,
)

logger = logging.getLogger(__name__)

START_POINT_DIRNAME = "mtp_start"


def _request(job_config: ResolvedJobConfig, mtp: dict) -> dict:
    """The preparation settings; a start point is reused only if it was built with exactly these."""
    return {
        "source": _resolve_model_id(job_config.model_name),
        "revision": None,
        "mode": "joint" if mtp["train_backbone"] else "heads",
        "objective": mtp["objective"],
        "num_draft_tokens": mtp["num_draft_tokens"],
        "loss_weight": mtp["loss_weight"],
        "init_from_layer": mtp["init_from_layer"],
    }


def _matches(start: pathlib.Path, request: dict) -> bool:
    manifest = start / MANIFEST_NAME
    if not manifest.is_file() or not (start / "config.json").is_file():
        return False
    return json.loads(manifest.read_text()).get("request") == request


def prepare_mtp_job(job_config: ResolvedJobConfig) -> ResolvedJobConfig:
    """Point ``model_name`` at an MTP start point with ``mtp.num_draft_tokens`` modules."""
    mtp = job_config.mtp
    if not mtp:
        return job_config
    output_dir = pathlib.Path(job_config.training_config.value["output_dir"])
    start = output_dir / START_POINT_DIRNAME
    request = _request(job_config, mtp)
    if not _matches(start, request):
        shutil.rmtree(
            start, ignore_errors=True
        )  # a stale or unfinished earlier attempt
        summary = prepare_lfm3_mtp_checkpoint(
            request["source"],
            str(start),
            mode=request["mode"],
            loss_weight=request["loss_weight"],
            init_from_layer=request["init_from_layer"],
            objective=request["objective"],
            num_draft_tokens=request["num_draft_tokens"],
        )
        logger.info("MTP start point: %s", summary)
    print(
        f"MTP: {mtp['num_draft_tokens']} draft token(s), {mtp['objective']} objective, "
        f"{'joint' if mtp['train_backbone'] else 'frozen backbone'}; start point {start}"
    )
    return dataclasses.replace(job_config, model_name=str(start))
