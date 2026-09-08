import logging
from typing import assert_never, cast

from pylate import evaluation, losses, models, utils
from leap_finetune.distribution.ray_runtime import normalize_visible_devices  # noqa: F401

from ray.train.huggingface.transformers import prepare_trainer
from sentence_transformers import SentenceTransformerTrainer

from leap_finetune.checkpointing.callback import LeapCheckpointCallback
from leap_finetune.config.job_config import ColBERTLoss
from leap_finetune.checkpointing.model_loading import _resolve_model_id
from leap_finetune.training.retrieval_utils import (
    align_retrieval_train_shard,
    build_ir_evaluation_data,
    build_retrieval_training_args,
    canonical_retrieval_dataset,
    register_remote_model_for_checkpointing,
)
from leap_finetune.training.utils.logging import finish_tracker
from leap_finetune.training.utils.trainer_lifecycle import run_training_safely
from leap_finetune.training.utils.trainer_mixins import RayDataLoaderMixin
from leap_finetune.training.utils.worker_setup import (
    get_ray_train_eval_datasets,
    init_tracking_from_config,
    setup_training_worker,
)

logger = logging.getLogger(__name__)


class LFMColBERTTrainer(RayDataLoaderMixin, SentenceTransformerTrainer):
    """PyLate trainer over Ray-owned dataset shards."""


def _build_evaluator(eval_dataset, batch_size: int):
    if eval_dataset is None or len(eval_dataset) == 0:
        return None
    queries, corpus, relevant_docs = build_ir_evaluation_data(eval_dataset)
    evaluators = [
        evaluation.PyLateInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="retrieval",
            batch_size=batch_size,
            write_csv=False,
        )
    ]
    if "negative" in eval_dataset.column_names:
        evaluators.append(
            evaluation.ColBERTTripletEvaluator(
                anchors=eval_dataset["query"],
                positives=eval_dataset["positive"],
                negatives=eval_dataset["negative"],
                name="retrieval_triplet",
                batch_size=batch_size,
                write_csv=False,
            )
        )
    return evaluators


def colbert_run(training_config: dict) -> None:
    setup_training_worker()
    train_dataset, eval_dataset = get_ray_train_eval_datasets()
    train_dataset = canonical_retrieval_dataset(train_dataset)
    eval_dataset = canonical_retrieval_dataset(eval_dataset)

    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")
    train_config = training_config.get("train_config", {})
    train_dataset = align_retrieval_train_shard(
        train_dataset,
        per_device_batch_size=train_config["per_device_train_batch_size"],
        gather_across_devices=train_config["gather_across_devices"],
    )
    output_dir = train_config.get("output_dir", "")
    resume_from = train_config.get("resume_from_checkpoint")

    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir or None,
        resume_from_checkpoint=resume_from,
    )
    args = build_retrieval_training_args(
        train_config,
        tracker=tracker,
        job_name=job_name,
    )
    model = models.ColBERT(
        model_name_or_path=_resolve_model_id(model_name),
        trust_remote_code=True,
    )
    register_remote_model_for_checkpointing(model)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    gather = train_config["gather_across_devices"]
    temperature = train_config["temperature"]
    loss_name = cast(ColBERTLoss, train_config["loss"])
    if loss_name == "contrastive":
        loss = losses.Contrastive(
            model=model,
            gather_across_devices=gather,
            temperature=temperature,
        )
    elif loss_name == "cached_contrastive":
        loss = losses.CachedContrastive(
            model=model,
            mini_batch_size=train_config.get("mini_batch_size", 32),
            gather_across_devices=gather,
            temperature=temperature,
        )
    else:
        assert_never(loss_name)

    trainer = LFMColBERTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=_build_evaluator(
            eval_dataset,
            args.per_device_eval_batch_size,
        ),
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    trainer.add_callback(
        LeapCheckpointCallback(
            run_name_template=train_config.get("leap_run_name_template")
        )
    )
    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)
    finish_tracker(tracker)
